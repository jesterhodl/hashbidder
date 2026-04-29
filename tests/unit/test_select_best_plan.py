"""Tests for plan-total hashrate and best-plan selection."""

from decimal import Decimal

import pytest

from hashbidder.domain.bid_config import BidConfig
from hashbidder.domain.bid_planning import (
    CreateAction,
    EditAction,
    ReconciliationPlan,
)
from hashbidder.domain.hashrate import Hashrate, HashratePrice, HashUnit
from hashbidder.domain.sats import Sats
from hashbidder.domain.time_unit import TimeUnit
from hashbidder.use_cases.set_bids_target import select_best_plan
from tests.conftest import EH_DAY, UPSTREAM, make_user_bid


def _ph_s(value: str) -> Hashrate:
    return Hashrate(Decimal(value), HashUnit.PH, TimeUnit.SECOND)


def _price(sats: int) -> HashratePrice:
    return HashratePrice(sats=Sats(sats), per=EH_DAY)


def _create(speed_ph_s: str, price_sat_eh_day: int = 500_000) -> CreateAction:
    return CreateAction(
        config=BidConfig(price=_price(price_sat_eh_day), speed_limit=_ph_s(speed_ph_s)),
        amount=Sats(100_000),
        upstream=UPSTREAM,
    )


def _plan_with_total_speed(speed_ph_s: str) -> ReconciliationPlan:
    """Build a plan whose total hashrate equals ``speed_ph_s``."""
    return ReconciliationPlan(
        cancels=(), edits=(), creates=(_create(speed_ph_s),), unchanged=()
    )


def _empty_plan() -> ReconciliationPlan:
    return ReconciliationPlan(cancels=(), edits=(), creates=(), unchanged=())


class TestSelectBestPlan:
    """Selection picks the candidate with the highest hashrate-deviation score.

    Score formula:
        100_000_000 * (1 - dev_pct * is_deviation_the_right_way)
    where is_deviation_the_right_way is 1 (right way) or 2 (wrong way), so the
    wrong-way candidate's penalty is twice as harsh per unit of deviation.
    """

    def test_empty_candidate_set_raises(self) -> None:
        """Empty input is a precondition violation, not silent None."""
        with pytest.raises(ValueError, match="at least one candidate"):
            select_best_plan(
                candidate_plans=(),
                long_term_hashrate_target=_ph_s("10"),
                current_hashrate_target=_ph_s("5"),
                target_price=_price(500_000),
            )

    def test_picks_plan_at_current_target(self) -> None:
        """Zero-deviation candidate beats both higher and lower alternatives."""
        plan_under = _plan_with_total_speed("3")
        plan_match = _plan_with_total_speed("5")
        plan_over = _plan_with_total_speed("8")
        best = select_best_plan(
            candidate_plans=(plan_under, plan_match, plan_over),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_match

    def test_smaller_right_way_deviation_wins(self) -> None:
        """Among right-way candidates, the one closer to the target wins."""
        # current=5, long_term=10. Both above current → right way.
        plan_close = _plan_with_total_speed("6")  # dev=20%
        plan_far = _plan_with_total_speed("9")  # dev=80%
        best = select_best_plan(
            candidate_plans=(plan_far, plan_close),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_close

    def test_smaller_wrong_way_deviation_wins(self) -> None:
        """Among wrong-way candidates, the one closer to the target wins."""
        # current=5, long_term=10. Both below current → wrong way.
        plan_close = _plan_with_total_speed("4")  # dev=20%
        plan_far = _plan_with_total_speed("2")  # dev=60%
        best = select_best_plan(
            candidate_plans=(plan_far, plan_close),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_close

    def test_under_long_term_prefers_right_way_over_wrong_way(self) -> None:
        """Under long-term, plan above current target beats plan below at same |dev|."""
        # current=5, long_term=10. Right way: plan=6. Wrong way: plan=4.
        plan_right = _plan_with_total_speed("6")
        plan_wrong = _plan_with_total_speed("4")
        best = select_best_plan(
            candidate_plans=(plan_wrong, plan_right),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_right

    def test_over_long_term_prefers_right_way_over_wrong_way(self) -> None:
        """Over long-term, plan below current target beats plan above at same |dev|."""
        # current=10, long_term=5. Right way: plan=8. Wrong way: plan=12.
        plan_right = _plan_with_total_speed("8")
        plan_wrong = _plan_with_total_speed("12")
        best = select_best_plan(
            candidate_plans=(plan_wrong, plan_right),
            long_term_hashrate_target=_ph_s("5"),
            current_hashrate_target=_ph_s("10"),
            target_price=_price(500_000),
        )
        assert best is plan_right

    def test_at_long_term_smaller_deviation_wins(self) -> None:
        """At long_term, both directions are wrong-way; smallest |dev| wins."""
        plan_close_above = _plan_with_total_speed("11")  # +10%
        plan_far_below = _plan_with_total_speed("8")  # -20%
        best = select_best_plan(
            candidate_plans=(plan_far_below, plan_close_above),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("10"),
            target_price=_price(500_000),
        )
        assert best is plan_close_above

    def test_zero_current_target_zero_plan_is_optimal(self) -> None:
        """When current target is zero, the zero-speed plan is optimal."""
        plan_zero = _empty_plan()
        plan_some = _plan_with_total_speed("3")
        best = select_best_plan(
            candidate_plans=(plan_some, plan_zero),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("0"),
            target_price=_price(500_000),
        )
        assert best is plan_zero


def _multi_create_plan(*speeds_ph_s: str) -> ReconciliationPlan:
    """A plan that consists of N creates at the given speeds."""
    return ReconciliationPlan(
        cancels=(),
        edits=(),
        creates=tuple(_create(s) for s in speeds_ph_s),
        unchanged=(),
    )


class TestSelectBestPlanBidCount:
    """Bid count component: ideal range is [target/2, min(target, 3)] bids."""

    def test_in_range_count_beats_single_bid_at_same_hashrate(self) -> None:
        """At identical plan hashrate, an in-range bid count beats a single bid."""
        # current_target=4, range=[2,4]. Both totals = 4 PH/s → identical
        # hashrate-deviation score. Bid-count component differentiates.
        plan_single = _plan_with_total_speed("4")  # 1 bid
        plan_in_range = _multi_create_plan("2", "2")  # 2 bids
        best = select_best_plan(
            candidate_plans=(plan_single, plan_in_range),
            long_term_hashrate_target=_ph_s("4"),
            current_hashrate_target=_ph_s("4"),
            target_price=_price(500_000),
        )
        assert best is plan_in_range

    def test_too_many_bids_loses_to_in_range_at_same_hashrate(self) -> None:
        """Too-many-bids candidate (still within hard cap) loses to in-range."""
        # current_target=2, range=[1,2]. Both totals = 4 PH/s.
        # in-range candidate: 2 bids at 2 PH/s each.
        # over-range candidate: 4 bids at 1 PH/s each (above range, within cap).
        plan_in_range = _multi_create_plan("2", "2")
        plan_too_many = _multi_create_plan("1", "1", "1", "1")
        best = select_best_plan(
            candidate_plans=(plan_too_many, plan_in_range),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("2"),
            target_price=_price(500_000),
        )
        assert best is plan_in_range

    def test_hard_cap_disqualifies_plan_with_more_than_3_bids(self) -> None:
        """A plan with 4 bids is never selected, even when otherwise ideal."""
        # plan_four perfectly matches hashrate target; plan_three is far off
        # but only has 3 bids. plan_three must still win because plan_four is
        # disqualified by the hard cap.
        plan_four = _multi_create_plan("1", "1", "1", "1")  # 4 PH/s
        plan_three = _multi_create_plan("1", "1", "1")  # 3 PH/s
        best = select_best_plan(
            candidate_plans=(plan_four, plan_three),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("4"),
            target_price=_price(500_000),
        )
        assert best is plan_three

    def test_all_candidates_over_hard_cap_raises(self) -> None:
        """Caller must include at least one cap-satisfying candidate.

        ``craft_all_possible_plans`` upholds this by always emitting the
        all-cancel + no-create plan. If a caller bypasses that and passes
        only over-cap candidates, ``select_best_plan`` raises rather than
        silently returning None.
        """
        plan_four = _multi_create_plan("1", "1", "1", "1")
        with pytest.raises(RuntimeError, match="hard cap"):
            select_best_plan(
                candidate_plans=(plan_four,),
                long_term_hashrate_target=_ph_s("10"),
                current_hashrate_target=_ph_s("4"),
                target_price=_price(500_000),
            )

    def test_single_bid_penalized_even_when_one_is_in_range(self) -> None:
        """A 1-bid plan still incurs the single-bid penalty when 1 is in range.

        For target=2 PH/s the range is [1, 2] and a 1-bid plan is technically
        in range, but the single-bid special case still adds a unit of
        distance — so a 2-bid plan with the same total hashrate beats it.
        """
        plan_single = _plan_with_total_speed("2")  # 1 bid, 2 PH/s
        plan_two = _multi_create_plan("1", "1")  # 2 bids, 2 PH/s
        best = select_best_plan(
            candidate_plans=(plan_single, plan_two),
            long_term_hashrate_target=_ph_s("2"),
            current_hashrate_target=_ph_s("2"),
            target_price=_price(500_000),
        )
        assert best is plan_two


class TestSelectBestPlanPrice:
    """Price component: weighted-average price (sat/PH/Day) is subtracted."""

    def test_cheaper_plan_wins_at_same_hashrate_and_bid_count(self) -> None:
        """Two equally-shaped plans differ only in price → cheap one wins."""
        plan_cheap = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(
                _create("1", price_sat_eh_day=500_000),
                _create("1", price_sat_eh_day=500_000),
            ),
            unchanged=(),
        )
        plan_pricey = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(
                _create("1", price_sat_eh_day=1_000_000),
                _create("1", price_sat_eh_day=1_000_000),
            ),
            unchanged=(),
        )
        best = select_best_plan(
            candidate_plans=(plan_pricey, plan_cheap),
            long_term_hashrate_target=_ph_s("2"),
            current_hashrate_target=_ph_s("2"),
            target_price=_price(500_000),
        )
        assert best is plan_cheap

    def test_weighted_average_uses_hashrate_as_weight(self) -> None:
        """A high-priced bid with tiny hashrate is outweighed by a cheap large bid.

        plan_a: 4 PH/s @ 500 sat/PH/Day + 1 PH/s @ 5_000 sat/PH/Day
                avg = (4 * 500 + 1 * 5_000) / 5 = (2_000 + 5_000) / 5 = 1_400
        plan_b: 4 PH/s @ 1_500 sat/PH/Day + 1 PH/s @ 1_500 sat/PH/Day
                avg = 1_500
        plan_a should win on price (1_400 < 1_500), all other components tied.
        """
        plan_a = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(
                _create("4", price_sat_eh_day=500_000),
                _create("1", price_sat_eh_day=5_000_000),
            ),
            unchanged=(),
        )
        plan_b = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(
                _create("4", price_sat_eh_day=1_500_000),
                _create("1", price_sat_eh_day=1_500_000),
            ),
            unchanged=(),
        )
        best = select_best_plan(
            candidate_plans=(plan_b, plan_a),
            long_term_hashrate_target=_ph_s("5"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_a

    def test_empty_plan_carries_no_price_penalty(self) -> None:
        """An empty plan (no live bids) doesn't apply the price subtraction.

        At current_target=0, both an empty plan and a cheap-1-bid plan are
        possible candidates. The empty plan should win because the bid-count
        component favors zero bids when the target is zero, and the empty
        plan has no price penalty to drag it down.
        """
        plan_empty = _empty_plan()
        plan_priced = _multi_create_plan("1")  # would carry a 500 penalty
        best = select_best_plan(
            candidate_plans=(plan_priced, plan_empty),
            long_term_hashrate_target=_ph_s("0"),
            current_hashrate_target=_ph_s("0"),
            target_price=_price(500_000),
        )
        assert best is plan_empty


class TestSelectBestPlanCooldowns:
    """Cooldown component: each price/speed decrease in an edit is a cooldown.

    Penalty = 10_000 * total_cooldowns / bid_count
            + 10_000 per bid that triggers both cooldowns
            + 1_000 * abs(price_cooldowns - speed_cooldowns)
    """

    def test_no_cooldowns_when_no_decreases(self) -> None:
        """A plan with no decreases triggers no cooldown penalty.

        Two plans with identical final shape (1 bid @ 500 sat/PH/day, 5 PH/s)
        but one is unchanged and the other is an edit that *decreases* both
        price and speed. The unchanged plan wins on the cooldown component.
        """
        target_bid = make_user_bid("B1", 500, "5.0")
        plan_unchanged = ReconciliationPlan(
            cancels=(), edits=(), creates=(), unchanged=(target_bid,)
        )
        # Same final state, but reached via an edit that decreased both fields.
        bid_pre_decrease = make_user_bid("B2", 600, "10.0")
        decrease_edit = EditAction(
            bid=bid_pre_decrease,
            new_price=target_bid.price,  # 500 < 600 → price decrease
            new_speed_limit_ph=_ph_s("5.0"),  # 5 < 10 → speed decrease
        )
        plan_with_both_cooldowns = ReconciliationPlan(
            cancels=(), edits=(decrease_edit,), creates=(), unchanged=()
        )
        best = select_best_plan(
            candidate_plans=(plan_with_both_cooldowns, plan_unchanged),
            long_term_hashrate_target=_ph_s("5"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_unchanged

    def test_split_cooldowns_beat_both_in_same_bid(self) -> None:
        """Two cooldowns split across two bids beats both on a single bid."""
        # Plan A: one bid with both decreases (price + speed).
        bid_a = make_user_bid("A", 600, "10.0")
        edit_both = EditAction(
            bid=bid_a,
            new_price=_price(500_000),
            new_speed_limit_ph=_ph_s("5.0"),
        )
        # Plan B: two bids, each with one kind of cooldown.
        bid_b1 = make_user_bid("B1", 600, "5.0")  # only price decreases
        bid_b2 = make_user_bid("B2", 500, "10.0")  # only speed decreases
        edit_price_only = EditAction(
            bid=bid_b1,
            new_price=_price(500_000),
            new_speed_limit_ph=_ph_s("5.0"),  # unchanged
        )
        edit_speed_only = EditAction(
            bid=bid_b2,
            new_price=_price(500_000),  # unchanged
            new_speed_limit_ph=_ph_s("5.0"),
        )

        plan_both_in_one = ReconciliationPlan(
            cancels=(), edits=(edit_both,), creates=(), unchanged=()
        )
        plan_split = ReconciliationPlan(
            cancels=(),
            edits=(edit_price_only, edit_speed_only),
            creates=(),
            unchanged=(),
        )
        best = select_best_plan(
            candidate_plans=(plan_both_in_one, plan_split),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("10"),
            target_price=_price(500_000),
        )
        assert best is plan_split

    def test_symmetric_cooldowns_beat_same_feature(self) -> None:
        """Two cooldowns of different feature beat two of the same feature."""
        # Plan A: 2 price decreases, 0 speed decreases.
        bid_a1 = make_user_bid("A1", 600, "5.0")
        bid_a2 = make_user_bid("A2", 600, "5.0")
        edit_a1 = EditAction(
            bid=bid_a1, new_price=_price(500_000), new_speed_limit_ph=_ph_s("5.0")
        )
        edit_a2 = EditAction(
            bid=bid_a2, new_price=_price(500_000), new_speed_limit_ph=_ph_s("5.0")
        )
        plan_same_feature = ReconciliationPlan(
            cancels=(), edits=(edit_a1, edit_a2), creates=(), unchanged=()
        )

        # Plan B: 1 price decrease + 1 speed decrease (across separate bids).
        bid_b1 = make_user_bid("B1", 600, "5.0")
        bid_b2 = make_user_bid("B2", 500, "10.0")
        edit_b1 = EditAction(
            bid=bid_b1, new_price=_price(500_000), new_speed_limit_ph=_ph_s("5.0")
        )
        edit_b2 = EditAction(
            bid=bid_b2, new_price=_price(500_000), new_speed_limit_ph=_ph_s("5.0")
        )
        plan_split_feature = ReconciliationPlan(
            cancels=(), edits=(edit_b1, edit_b2), creates=(), unchanged=()
        )
        best = select_best_plan(
            candidate_plans=(plan_same_feature, plan_split_feature),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("10"),
            target_price=_price(500_000),
        )
        assert best is plan_split_feature


class TestSelectBestPlanBelowTargetPriceUnserved:
    """Bids priced below the market target_price contribute zero hashrate."""

    def test_below_target_bid_does_not_close_hashrate_gap(self) -> None:
        """An under-priced bid can't close the hashrate gap.

        A plan that hits the speed target with a below-target-price bid
        still scores worse than one that hits it with a served bid, since
        the unserved bid won't actually be matched.
        """
        # Market target_price = 500_000 sat/EH/Day.
        # plan_underpriced: 1 bid at 5 PH/s but at 400 sat/PH/Day (= 400_000
        #   sat/EH/Day, below target_price → unserved → effective 0 PH/s).
        # plan_served: 1 bid at 5 PH/s at the target price → effective 5 PH/s.
        plan_underpriced = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(_create("5", price_sat_eh_day=400_000),),
            unchanged=(),
        )
        plan_served = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(_create("5", price_sat_eh_day=500_000),),
            unchanged=(),
        )
        best = select_best_plan(
            candidate_plans=(plan_underpriced, plan_served),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_served

    def test_below_target_bid_does_not_pollute_weighted_average(self) -> None:
        """Under-priced bids are excluded from the weighted-average price.

        plan_a: 5 PH/s at 600 sat/PH/Day (served).
        plan_b: 5 PH/s at 600 sat/PH/Day (served) + 5 PH/s at 100 sat/PH/Day
                (unserved). If unserved bids polluted the average, plan_b
                would *appear* cheaper. They don't, so plan_b ties plan_a on
                weighted price and loses on bid count (plan_b has 2 bids,
                plan_a has 1). Among the two, plan_a should win.
        """
        # current_target=5, range=[2.5, 5] bids ⇒ both 1 and 2 are valid;
        # but plan_a has 1 bid (single-bid floor of 1 distance).
        # Make plan_a have 2 served bids (in range), plan_b have 1 served
        # + 1 unserved (effective 1 served-bid hashrate, but 2 in count).
        plan_a = ReconciliationPlan(  # 2 served bids, weighted price = 600
            cancels=(),
            edits=(),
            creates=(
                _create("2.5", price_sat_eh_day=600_000),
                _create("2.5", price_sat_eh_day=600_000),
            ),
            unchanged=(),
        )
        plan_b = ReconciliationPlan(  # 1 served + 1 unserved; bid count=2
            cancels=(),
            edits=(),
            creates=(
                _create("2.5", price_sat_eh_day=600_000),
                _create("2.5", price_sat_eh_day=100_000),  # unserved
            ),
            unchanged=(),
        )
        best = select_best_plan(
            candidate_plans=(plan_b, plan_a),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        # plan_a delivers 5 served PH/s (perfect match);
        # plan_b delivers only 2.5 served PH/s → bigger hashrate-deviation.
        assert best is plan_a


# ============================================================================
# Stubs from the test plan (2026-04-28). Some plan items already have
# implementations above; the stubs below are entries for items not yet
# covered. Each stub is a name + docstring; bodies are intentionally empty.
# ============================================================================


class TestSelectBestPlanInvariantsStubs:
    """Cross-cutting invariants of the selector — should hold over any input."""

    def test_always_returns_a_plan_in_input(self) -> None:
        """Non-empty candidate_plans → result is a member of the list."""
        p1 = _plan_with_total_speed("3")
        p2 = _plan_with_total_speed("5")
        p3 = _plan_with_total_speed("8")
        candidates = (p1, p2, p3)
        best = select_best_plan(
            candidate_plans=candidates,
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert any(best is c for c in candidates)

    def test_deterministic_across_runs(self) -> None:
        """Same inputs twice → identical (is-the-same-object) plan returned."""
        candidates = (
            _plan_with_total_speed("3"),
            _plan_with_total_speed("5"),
            _plan_with_total_speed("8"),
        )
        long_term = _ph_s("10")
        current = _ph_s("5")
        target_price = _price(500_000)
        first = select_best_plan(
            candidate_plans=candidates,
            long_term_hashrate_target=long_term,
            current_hashrate_target=current,
            target_price=target_price,
        )
        second = select_best_plan(
            candidate_plans=candidates,
            long_term_hashrate_target=long_term,
            current_hashrate_target=current,
            target_price=target_price,
        )
        assert first is second

    def test_first_wins_on_score_tie(self) -> None:
        """Ordering invariant: ties go to the earlier candidate."""
        # Two structurally identical, distinct ReconciliationPlan instances.
        p1 = _plan_with_total_speed("5")
        p2 = _plan_with_total_speed("5")
        assert p1 is not p2
        best = select_best_plan(
            candidate_plans=(p1, p2),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is p1

    def test_hard_cap_respected_when_alternatives_exist(self) -> None:
        """If at least one ≤3-bid candidate is present, the winner has ≤3 bids."""
        # 4-bid plan would be perfect on hashrate (sum=4 PH/s, target=4) but
        # is hard-capped out. 3-bid plan undershoots but is selectable.
        plan_four = _multi_create_plan("1", "1", "1", "1")
        plan_three = _multi_create_plan("1", "1", "1")
        best = select_best_plan(
            candidate_plans=(plan_four, plan_three),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("4"),
            target_price=_price(500_000),
        )
        bid_count = len(best.unchanged) + len(best.edits) + len(best.creates)
        assert bid_count <= 3

    def test_components_are_subtractive(self) -> None:
        """Adding penalty material to a candidate strictly lowers its rank."""
        # Baseline: 2 in-range bids at 2 PH/s each, total 4, served, cheap.
        baseline = _multi_create_plan("2", "2")

        # Variant A: shrinks bid count to 1 (single-bid floor + below range).
        variant_count = _plan_with_total_speed("4")

        # Variant B: same shape as baseline but one bid is reached via a
        # decrease edit (price-cooldown trigger).
        bid_pre = make_user_bid("Bx", 600, "2.0")
        decrease_edit = EditAction(
            bid=bid_pre,
            new_price=_price(500_000),  # 500 < 600 → triggers price cooldown
            new_speed_limit_ph=_ph_s("2"),
        )
        variant_cooldown = ReconciliationPlan(
            cancels=(),
            edits=(decrease_edit,),
            creates=(_create("2"),),
            unchanged=(),
        )

        # Variant C: same shape as baseline but more expensive.
        variant_price = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(
                _create("2", price_sat_eh_day=1_000_000),
                _create("2", price_sat_eh_day=1_000_000),
            ),
            unchanged=(),
        )

        # In each head-to-head, baseline wins.
        for variant in (variant_count, variant_cooldown, variant_price):
            best = select_best_plan(
                candidate_plans=(variant, baseline),
                long_term_hashrate_target=_ph_s("4"),
                current_hashrate_target=_ph_s("4"),
                target_price=_price(500_000),
            )
            assert best is baseline

    def test_below_target_price_bids_contribute_zero(self) -> None:
        """An unserved bid contributes nothing to hashrate or price scoring.

        Two plans differ only in the *price* of an unserved bid.
        Effective hashrate and weighted-avg price are identical, bid count
        is identical, cooldowns are identical → scores tie → first-wins.
        """
        plan_first = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(
                _create("4", price_sat_eh_day=500_000),
                _create("1", price_sat_eh_day=100_000),  # unserved
            ),
            unchanged=(),
        )
        plan_second = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(
                _create("4", price_sat_eh_day=500_000),
                _create("1", price_sat_eh_day=200_000),  # unserved (different price)
            ),
            unchanged=(),
        )
        best = select_best_plan(
            candidate_plans=(plan_first, plan_second),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("4"),
            target_price=_price(500_000),
        )
        assert best is plan_first


class TestSelectBestPlanGoldenStubs:
    """Hand-built candidates exercising one scoring component at a time."""

    def test_zero_deviation_candidate_outranks_others(self) -> None:
        """A candidate matching current_target exactly beats deviating ones."""
        plan_under = _plan_with_total_speed("3")
        plan_match = _plan_with_total_speed("5")
        plan_over = _plan_with_total_speed("8")
        best = select_best_plan(
            candidate_plans=(plan_under, plan_over, plan_match),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(500_000),
        )
        assert best is plan_match


class TestSelectBestPlanEdgeCasesStubs:
    """Boundary conditions for the selector."""

    def test_target_price_so_high_no_bid_clears(self) -> None:
        """No candidate has any served bid → empty plan wins.

        With target_price above every bid, every plan has effective
        hashrate 0 (so the hashrate component is identical across
        candidates). The bid-count component then breaks the tie: at
        current_target=0 the in-range slot is [0, 0], so count=0 wins.
        """
        plan_empty = _empty_plan()
        plan_some = _plan_with_total_speed("5")  # priced at 500_000, unserved
        plan_more = _multi_create_plan("2", "3")  # also unserved
        best = select_best_plan(
            candidate_plans=(plan_some, plan_more, plan_empty),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("0"),
            target_price=_price(10_000_000),  # way above any bid price
        )
        assert best is plan_empty

    def test_target_price_zero_every_bid_clears(self) -> None:
        """target_price=0 makes every bid served; ranking matches unfiltered."""
        plan_under = _plan_with_total_speed("3")
        plan_match = _plan_with_total_speed("5")
        plan_over = _plan_with_total_speed("8")
        best = select_best_plan(
            candidate_plans=(plan_under, plan_match, plan_over),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("5"),
            target_price=_price(0),
        )
        # On-target candidate still wins — the zero filter is a no-op here.
        assert best is plan_match

    def test_needed_zero_with_long_term_positive(self) -> None:
        """needed=0, long_term>0 → empty plan is optimal."""
        plan_empty = _empty_plan()
        plan_some = _plan_with_total_speed("3")
        best = select_best_plan(
            candidate_plans=(plan_some, plan_empty),
            long_term_hashrate_target=_ph_s("10"),
            current_hashrate_target=_ph_s("0"),
            target_price=_price(500_000),
        )
        assert best is plan_empty

    def test_huge_needed_with_only_sub_cap_options(self) -> None:
        """needed=1000 PH/s with only ≤3-bid candidates → returns without error."""
        plan_close = _multi_create_plan("400", "400", "200")  # 1000 PH/s
        plan_far = _multi_create_plan("200", "200", "100")  # 500 PH/s
        best = select_best_plan(
            candidate_plans=(plan_far, plan_close),
            long_term_hashrate_target=_ph_s("1000"),
            current_hashrate_target=_ph_s("1000"),
            target_price=_price(500_000),
        )
        # The exact match wins — most importantly, the call doesn't crash.
        assert best is plan_close

    def test_single_candidate_returns_regardless_of_score(self) -> None:
        """Single candidate is the winner even with maxed penalties."""
        # Build a candidate with: huge deviation, both-cooldown trigger,
        # 1 bid (single-bid floor), expensive price.
        bid = make_user_bid("X", 1000, "20.0")
        nasty_edit = EditAction(
            bid=bid,
            new_price=_price(900_000),  # 1000 → 900: price cooldown
            new_speed_limit_ph=_ph_s("1.0"),  # 20 → 1: speed cooldown
        )
        nasty = ReconciliationPlan(
            cancels=(), edits=(nasty_edit,), creates=(), unchanged=()
        )
        best = select_best_plan(
            candidate_plans=(nasty,),
            long_term_hashrate_target=_ph_s("100"),
            current_hashrate_target=_ph_s("50"),
            target_price=_price(500_000),
        )
        assert best is nasty

    def test_near_score_tie_resolves_deterministically(self) -> None:
        """Two near-tied scores resolve via first-wins on every run."""
        # Two plans that differ by an extremely small margin in price only.
        plan_a = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(_create("4", price_sat_eh_day=500_000),),
            unchanged=(),
        )
        plan_b = ReconciliationPlan(
            cancels=(),
            edits=(),
            creates=(_create("4", price_sat_eh_day=500_001),),
            unchanged=(),
        )
        # plan_a is fractionally cheaper → wins. Run twice; same on every run.
        for _ in range(2):
            best = select_best_plan(
                candidate_plans=(plan_a, plan_b),
                long_term_hashrate_target=_ph_s("4"),
                current_hashrate_target=_ph_s("4"),
                target_price=_price(500_000),
            )
            assert best is plan_a

    def test_negative_total_score_is_still_selectable(self) -> None:
        """A deeply negative aggregate score is still returned when alone."""
        # 1 bid, far below current_target, so deviation > 100% → score < 0.
        plan_terrible = _plan_with_total_speed("1")  # dev = (1-50)/50 = 98%
        best = select_best_plan(
            candidate_plans=(plan_terrible,),
            long_term_hashrate_target=_ph_s("100"),
            current_hashrate_target=_ph_s("50"),
            target_price=_price(500_000),
        )
        assert best is plan_terrible
