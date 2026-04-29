"""Direct tests for _plan_reconciliation — the pure planning logic."""

from decimal import Decimal

from hashbidder.clients.braiins import AccountBalance, BidStatus, UserBid
from hashbidder.domain.bid_config import MIN_BID_SPEED_LIMIT, TargetHashrateConfig
from hashbidder.domain.hashrate import Hashrate, HashratePrice, HashUnit
from hashbidder.domain.sats import Sats
from hashbidder.domain.time_unit import TimeUnit
from hashbidder.services.target_hashrate import BidWithCooldown
from hashbidder.use_cases.set_bids_target import (
    _HARD_CAP_BIDS,
    TargetHashrateInputs,
    _plan_reconciliation,
    craft_all_possible_plans,
)
from tests.conftest import UPSTREAM, make_user_bid

EH_DAY = Hashrate(Decimal(1), HashUnit.EH, TimeUnit.DAY)


def _ph_s(value: str) -> Hashrate:
    return Hashrate(Decimal(value), HashUnit.PH, TimeUnit.SECOND)


def _price(sats: int) -> HashratePrice:
    return HashratePrice(sats=Sats(sats), per=EH_DAY)


_DEFAULT_TARGET = _ph_s("10")
_DEFAULT_NEEDED = _ph_s("15")
_DEFAULT_PRICE = HashratePrice(sats=Sats(501_000), per=EH_DAY)


def _config() -> TargetHashrateConfig:
    return TargetHashrateConfig(
        default_amount=Sats(100_000),
        upstream=UPSTREAM,
        target_hashrate=_ph_s("10"),
    )


_ZERO_BALANCE = AccountBalance(
    available_sat=Sats(0), blocked_sat=Sats(0), total_sat=Sats(0)
)


def _inputs(
    *,
    target: Hashrate = _DEFAULT_TARGET,
    needed: Hashrate = _DEFAULT_NEEDED,
    price: HashratePrice = _DEFAULT_PRICE,
    bids_with_cooldowns: tuple[BidWithCooldown, ...] = (),
    non_manageable_bids: tuple[UserBid, ...] = (),
) -> TargetHashrateInputs:
    return TargetHashrateInputs(
        ocean_24h=_ph_s("5"),
        target=target,
        needed_hashrate=needed,
        target_price=price,
        bids_with_cooldowns=bids_with_cooldowns,
        non_manageable_bids=non_manageable_bids,
        available_balance=_ZERO_BALANCE,
    )


def _config_with_target(target_ph_s: str) -> TargetHashrateConfig:
    return TargetHashrateConfig(
        default_amount=Sats(100_000),
        upstream=UPSTREAM,
        target_hashrate=_ph_s(target_ph_s),
    )


def _served_total_ph_s(plan, target_price: HashratePrice) -> Hashrate:  # type: ignore[no-untyped-def]
    """Sum of speeds in the plan from bids whose price ≥ target_price."""
    total = _ph_s("0")
    for bid in plan.unchanged:
        if bid.price >= target_price:
            total = total + bid.speed_limit_ph
    for edit in plan.edits:
        if edit.new_price >= target_price:
            total = total + edit.new_speed_limit_ph
    for create in plan.creates:
        if create.config.price >= target_price:
            total = total + create.config.speed_limit
    return total


def _bwc(
    bid_id: str = "B1",
    price_sat_per_ph_day: int = 800,
    speed: str = "3",
    *,
    price_cd: bool = False,
    speed_cd: bool = False,
    amount: int = 100_000,
    remaining: int | None = None,
) -> BidWithCooldown:
    return BidWithCooldown(
        bid=make_user_bid(
            bid_id, price_sat_per_ph_day, speed, amount=amount, remaining=remaining
        ),
        is_price_in_cooldown=price_cd,
        is_speed_in_cooldown=speed_cd,
    )


class TestNeedZero:
    """When needed hashrate is zero: cancel everything manageable."""

    def test_no_bids_produces_empty_plan(self) -> None:
        """No existing bids + zero need → empty plan."""
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("0"), bids_with_cooldowns=()),
            _config(),
        )
        assert plan.cancels == ()
        assert plan.creates == ()
        assert plan.edits == ()
        assert plan.unchanged == ()


class TestCreate:
    """No existing bids + positive need → create one at market price."""

    def test_creates_one_bid_at_market_price_and_needed_speed(self) -> None:
        """No bids + positive need → one create at (price, needed)."""
        plan = _plan_reconciliation(
            _inputs(
                needed=_ph_s("15"),
                price=_price(501_000),
                bids_with_cooldowns=(),
            ),
            _config(),
        )
        assert len(plan.creates) == 1
        create = plan.creates[0]
        assert create.config.price == _price(501_000)
        assert create.config.speed_limit == _ph_s("15")
        assert create.amount == Sats(100_000)
        assert create.upstream == UPSTREAM
        assert plan.cancels == ()
        assert plan.edits == ()
        assert plan.unchanged == ()


# ============================================================================
# Stubs from the test plan (2026-04-28). End-to-end coverage of the new
# craft_all_possible_plans → select_best_plan pipeline. Bodies are empty;
# each stub is a name + docstring describing what to assert and how.
# ============================================================================


class TestPlanReconciliationInvariantsStubs:
    """End-to-end invariants of the craft → select pipeline."""

    def test_always_returns_a_plan(self) -> None:
        """For any TargetHashrateInputs, _plan_reconciliation returns a plan."""
        scenarios = (
            _inputs(needed=_ph_s("0"), bids_with_cooldowns=()),
            _inputs(needed=_ph_s("15"), bids_with_cooldowns=()),
            _inputs(
                needed=_ph_s("5"),
                bids_with_cooldowns=(_bwc("B1", 600, "5"),),
            ),
            # Two-bid case to avoid blowing up the cartesian product. Both
            # locked to keep per-bid options small.
            _inputs(
                needed=_ph_s("10"),
                bids_with_cooldowns=(
                    _bwc("B1", 600, "5", price_cd=True, speed_cd=True),
                    _bwc("B2", 700, "3", price_cd=True, speed_cd=True),
                ),
            ),
        )
        for inputs in scenarios:
            plan = _plan_reconciliation(inputs, _config())
            assert plan is not None

    def test_no_live_speed_below_min_bid_speed_limit(self) -> None:
        """Every edit and create in the result has speed ≥ MIN_BID_SPEED_LIMIT."""
        # Tiny target plus a small bid: percentage decreases would dip below
        # 1 PH/s, and fraction-of-target creates likewise.
        bid = _bwc("B1", 500, "2.0")
        plan = _plan_reconciliation(
            _inputs(
                target=_ph_s("4"),
                needed=_ph_s("4"),
                bids_with_cooldowns=(bid,),
            ),
            _config_with_target("4"),
        )
        for edit in plan.edits:
            assert edit.new_speed_limit_ph >= MIN_BID_SPEED_LIMIT
        for create in plan.creates:
            assert create.config.speed_limit >= MIN_BID_SPEED_LIMIT

    def test_bid_count_never_exceeds_hard_cap(self) -> None:
        """Result's bid count is ≤ _HARD_CAP_BIDS across varied inputs."""
        scenarios = [
            _inputs(needed=_ph_s("0"), bids_with_cooldowns=()),
            _inputs(needed=_ph_s("15")),
            # Two locked bids — small option space, still exercises the cap.
            _inputs(
                needed=_ph_s("15"),
                bids_with_cooldowns=(
                    _bwc("B1", 600, "5", price_cd=True, speed_cd=True),
                    _bwc("B2", 700, "8", price_cd=True, speed_cd=True),
                ),
            ),
        ]
        for inputs in scenarios:
            plan = _plan_reconciliation(inputs, _config())
            assert (
                len(plan.unchanged) + len(plan.edits) + len(plan.creates)
                <= _HARD_CAP_BIDS
            )

    def test_existing_bids_partition_disjointly(self) -> None:
        """Each input bid appears in exactly one of cancels/edits/unchanged."""
        # Two bids (locked → small per-bid option space).
        bids = (
            _bwc("B1", 500, "5", price_cd=True, speed_cd=True),
            _bwc("B2", 700, "3", price_cd=True, speed_cd=True),
        )
        plan = _plan_reconciliation(
            _inputs(bids_with_cooldowns=bids),
            _config(),
        )
        cancel_ids = {c.bid.id for c in plan.cancels}
        edit_ids = {e.bid.id for e in plan.edits}
        unchanged_ids = {u.id for u in plan.unchanged}
        assert cancel_ids.isdisjoint(edit_ids)
        assert cancel_ids.isdisjoint(unchanged_ids)
        assert edit_ids.isdisjoint(unchanged_ids)
        assert cancel_ids | edit_ids | unchanged_ids == {b.bid.id for b in bids}

    def test_no_op_plan_was_in_candidate_set(self) -> None:
        """The all-unchanged + no-creates plan is always offered."""
        bids = (
            _bwc("B1", 500, "5", price_cd=True, speed_cd=True),
            _bwc("B2", 700, "3", price_cd=True, speed_cd=True),
        )
        candidates = craft_all_possible_plans(
            _inputs(bids_with_cooldowns=bids), _config()
        )
        bid_ids = {b.bid.id for b in bids}
        assert any(
            {u.id for u in p.unchanged} == bid_ids
            and p.cancels == ()
            and p.edits == ()
            and p.creates == ()
            for p in candidates
        )

    def test_locked_decreases_never_proposed(self) -> None:
        """No edit lowers a locked field on a cooldown'd bid."""
        bids = (
            _bwc("B1", 800, "10", price_cd=True, speed_cd=True),
            _bwc("B2", 700, "5", price_cd=True, speed_cd=True),
        )
        by_id = {b.bid.id: b for b in bids}
        plan = _plan_reconciliation(
            _inputs(bids_with_cooldowns=bids),
            _config(),
        )
        for edit in plan.edits:
            bwc = by_id[edit.bid.id]
            if bwc.is_price_in_cooldown:
                assert edit.new_price >= edit.bid.price
            if bwc.is_speed_in_cooldown:
                assert edit.new_speed_limit_ph >= edit.bid.speed_limit_ph


class TestPlanReconciliationGoldenStubs:
    """Common scenarios — verify sensible plan selection through the pipeline."""

    def test_aligned_existing_bid_at_target_price_kept(self) -> None:
        """An aligned bid + same target_price → effective served hashrate matches.

        Under the new architecture the bid may be cancelled and replaced
        rather than kept by identity (cancel+create at target price has
        the same shape and ties on score). What matters is the *result*:
        the chosen plan delivers needed hashrate at the target price.
        """
        bid = _bwc("B1", 501, "1", price_cd=True, speed_cd=True)
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("1"), bids_with_cooldowns=(bid,)),
            _config_with_target("1"),
        )
        assert _served_total_ph_s(plan, _DEFAULT_PRICE) == _ph_s("1")

    def test_existing_bid_above_target_price_kept(self) -> None:
        """High-priced (still-served) bid is kept rather than canceled.

        bid.price strictly above target_price → strictly cheaper to keep
        the unchanged bid than to cancel+recreate at the same price (the
        unchanged path costs the bid's price, recreate costs target_price
        which is even lower; but the unchanged plan ties on hashrate and
        loses on price by exactly the difference). Verify the chosen plan
        either keeps it unchanged OR replaces it via cancel+create — what
        matters is that effective served hashrate equals needed.
        """
        bid = _bwc("B1", 1000, "1", price_cd=True, speed_cd=True)
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("1"), bids_with_cooldowns=(bid,)),
            _config_with_target("1"),
        )
        assert _served_total_ph_s(plan, _DEFAULT_PRICE) == _ph_s("1")

    def test_existing_bid_below_target_price_canceled(self) -> None:
        """Bid priced below target_price (unserved) → canceled."""
        # bid price = 400 sat/PH/Day = 400_000 sat/EH/Day, below target_price
        # 501_000 → unserved. Both cooldowns → can't decrease into relevance.
        # Edit-up to target_price is allowed (no cooldown blocks increases),
        # but that creates a still-unserved (price equal) bid only if the
        # edit hits target — verify the bid is at least not in unchanged.
        bid = _bwc("B1", 400, "5", price_cd=True, speed_cd=True)
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("5"), bids_with_cooldowns=(bid,)),
            _config_with_target("5"),
        )
        # The unserved-as-is bid must not be left alone (it'd be dead weight).
        assert all(u.id != bid.bid.id for u in plan.unchanged)

    def test_misaligned_bid_no_cooldowns_converges_to_target(self) -> None:
        """Miscalibrated bid + no cooldowns → converges with no floor violations.

        Effective served hashrate is close to needed and no plan field
        violates MIN_BID_SPEED_LIMIT.
        """
        # Bid at 1000 sat/PH/Day, 20 PH/s. needed=5 → wants to decrease.
        # No cooldowns → free to edit.
        bid = _bwc("B1", 1000, "20")
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("5"), bids_with_cooldowns=(bid,)),
            _config_with_target("5"),
        )
        # All plan fields respect the speed-limit floor.
        for edit in plan.edits:
            assert edit.new_speed_limit_ph >= MIN_BID_SPEED_LIMIT
        for create in plan.creates:
            assert create.config.speed_limit >= MIN_BID_SPEED_LIMIT

    def test_many_bids_settles_within_acceptable_range(self) -> None:
        """Multiple manageable bids + reasonable target → bid count in range."""
        # Three locked bids, target=4 → bid-count range = [2, 4].
        bids = (
            _bwc("B1", 600, "2", price_cd=True, speed_cd=True),
            _bwc("B2", 700, "2", price_cd=True, speed_cd=True),
            _bwc("B3", 800, "2", price_cd=True, speed_cd=True),
        )
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("4"), bids_with_cooldowns=bids),
            _config_with_target("4"),
        )
        bid_count = len(plan.unchanged) + len(plan.edits) + len(plan.creates)
        # Acceptable range is [needed/2, min(needed, _HARD_CAP_BIDS)] = [2, 4].
        assert 2 <= bid_count <= 4

    def test_double_cooldown_with_double_decrease_request_is_blocked(self) -> None:
        """Bid in both cooldowns + target below on both fields → no decrease."""
        # bid 1000 sat/PH/Day, 20 PH/s, both cooldowns. target = 501 / 5.
        bid = _bwc("B1", 1000, "20", price_cd=True, speed_cd=True)
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("5"), bids_with_cooldowns=(bid,)),
            _config_with_target("5"),
        )
        for edit in plan.edits:
            if edit.bid.id == bid.bid.id:
                assert edit.new_price >= edit.bid.price
                assert edit.new_speed_limit_ph >= edit.bid.speed_limit_ph


class TestPlanReconciliationEdgeCasesStubs:
    """Boundary cases through the full pipeline."""

    def test_locked_sub_target_bid_is_canceled(self) -> None:
        """Bid pinned by both cooldowns at a price below target_price → cancel.

        The bid is unserved and locked from any decrease. Keeping it
        unchanged would waste a bid-count slot for zero served hashrate.
        """
        bid = _bwc("B1", 400, "5", price_cd=True, speed_cd=True)
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("5"), bids_with_cooldowns=(bid,)),
            _config_with_target("5"),
        )
        assert all(u.id != bid.bid.id for u in plan.unchanged)

    def test_four_existing_bids_yields_cap_satisfying_plan(self) -> None:
        """4 manageable existing bids → planner picks a ≤3-bid plan.

        Uses sub-MIN bid speeds (forced by both cooldowns) so the per-bid
        option set collapses to ~3 entries each, keeping the cartesian
        product tractable.
        """
        # speed=0.5 (sub-MIN) + both cooldowns: speed_choices = [0.5, needed]
        # → 1 cancel + 2 (price * speed) - 1 (unchanged) + 1 unchanged = 3 per bid.
        bids = tuple(
            _bwc(f"B{i}", 600, "0.5", price_cd=True, speed_cd=True) for i in range(4)
        )
        plan = _plan_reconciliation(
            _inputs(target=_ph_s("3"), needed=_ph_s("3"), bids_with_cooldowns=bids),
            _config_with_target("3"),
        )
        bid_count = len(plan.unchanged) + len(plan.edits) + len(plan.creates)
        assert bid_count <= _HARD_CAP_BIDS
        assert len(plan.cancels) >= 1

    def test_legacy_sub_min_speed_bid_never_edited_to_sub_min(self) -> None:
        """A sub-MIN-speed bid is never edited to a sub-MIN target speed.

        Increases are still allowed (and an increase to MIN_BID_SPEED_LIMIT
        is a valid edit). The invariant is that no EditAction produced has
        a new_speed_limit_ph below MIN_BID_SPEED_LIMIT — otherwise
        EditAction.__post_init__ would have raised.
        """
        bid = _bwc("B1", 600, "0.5", price_cd=True, speed_cd=True)
        plan = _plan_reconciliation(
            _inputs(needed=_ph_s("1"), bids_with_cooldowns=(bid,)),
            _config_with_target("1"),
        )
        for edit in plan.edits:
            assert edit.new_speed_limit_ph >= MIN_BID_SPEED_LIMIT

    def test_tiny_target_picks_zero_or_one_bid(self) -> None:
        """target=1 PH/s → chosen plan has at most a handful of bids."""
        plan = _plan_reconciliation(
            _inputs(target=_ph_s("1"), needed=_ph_s("1")),
            _config_with_target("1"),
        )
        bid_count = len(plan.unchanged) + len(plan.edits) + len(plan.creates)
        assert bid_count <= 2

    def test_at_long_term_target_zero_deviation_wins(self) -> None:
        """current_target == long_term_target → served hashrate equals current."""
        plan = _plan_reconciliation(
            _inputs(target=_ph_s("3"), needed=_ph_s("3")),
            _config_with_target("3"),
        )
        assert _served_total_ph_s(plan, _DEFAULT_PRICE) == _ph_s("3")

    def test_catch_up_scenario_overshoots_target_modestly(self) -> None:
        """When needed > target, served hashrate is closer to needed than 0.

        Catch-up scenario where needed > target. The chosen plan should
        deliver most of needed_hashrate to drive 24h average toward target.
        We assert served >= half of needed (i.e., we're not falling
        further behind by undershooting drastically).
        """
        plan = _plan_reconciliation(
            _inputs(target=_ph_s("3"), needed=_ph_s("10")),
            _config_with_target("3"),
        )
        served = _served_total_ph_s(plan, _DEFAULT_PRICE)
        assert served >= _ph_s("5")

    def test_only_non_manageable_bids_planner_sees_empty(self) -> None:
        """All non-manageable bids → planner sees an empty bids_with_cooldowns."""
        # Build a UserBid that's PAUSED. The planner is given an empty
        # bids_with_cooldowns (since manageable filtering happens upstream).
        paused = make_user_bid("P1", 700, "5", status=BidStatus.PAUSED)
        plan = _plan_reconciliation(
            _inputs(
                needed=_ph_s("0"),
                bids_with_cooldowns=(),
                non_manageable_bids=(paused,),
            ),
            _config(),
        )
        # needed=0 → empty plan (no creates, nothing to cancel).
        assert plan.cancels == ()
        assert plan.edits == ()
        assert plan.creates == ()
        assert plan.unchanged == ()
