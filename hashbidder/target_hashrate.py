"""Pure computations for target-hashrate mode."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from hashbidder.client import MarketSettings, OrderBook, UserBid
from hashbidder.domain.bid_config import BidConfig
from hashbidder.domain.bid_history import BidHistory
from hashbidder.domain.hashrate import Hashrate, HashratePrice, HashUnit
from hashbidder.domain.price_tick import PriceTick
from hashbidder.domain.time_unit import TimeUnit


def compute_needed_hashrate(target: Hashrate, current_24h: Hashrate) -> Hashrate:
    """Hashrate to buy now so the 12h-forward 24h average equals target.

    Assumes a 12-hour horizon: if we add `needed` for the next 12 hours, the
    rolling 24h average will land at `target`. Clamped to zero when already
    at or above target.
    """
    twice = target + target
    if current_24h >= twice:
        return Hashrate(Decimal(0), HashUnit.PH, TimeUnit.SECOND)
    return (twice - current_24h).to(HashUnit.PH, TimeUnit.SECOND)


def distribute_bids(needed: Hashrate, max_bids_count: int) -> tuple[Hashrate, ...]:
    """Split a needed hashrate into per-bid speed limits in PH/s.

    Uses as many bids as possible up to `max_bids_count`, with each bid >= 1 PH/s
    and the total summing to `needed` (rounded to 0.01 PH/s precision).
    Returns an empty tuple to mean "cancel all bids".
    """
    if max_bids_count < 1:
        raise ValueError(f"max_bids_count must be >= 1, got {max_bids_count}")

    needed_ph = needed.to(HashUnit.PH, TimeUnit.SECOND).value
    if needed_ph < Decimal("0.5"):
        return ()
    if needed_ph < 1:
        return (Hashrate(Decimal(1), HashUnit.PH, TimeUnit.SECOND),)

    n = min(max_bids_count, int(needed_ph))
    share = (needed_ph / Decimal(n)).quantize(Decimal("0.01"))
    return tuple(Hashrate(share, HashUnit.PH, TimeUnit.SECOND) for _ in range(n))


@dataclass(frozen=True)
class CooldownInfo:
    """Whether a bid is still in its decrease cooldown windows.

    A True flag means the corresponding field cannot be lowered yet — the
    Braiins API enforces a minimum delay between consecutive decreases.
    Increases are always allowed.
    """

    price_cooldown: bool
    speed_cooldown: bool


@dataclass(frozen=True)
class BidWithCooldown:
    """A bid paired with its current cooldown status."""

    bid: UserBid
    is_price_in_cooldown: bool
    is_speed_in_cooldown: bool


def is_price_guaranteed_free(
    bid: UserBid, settings: MarketSettings, now: datetime
) -> bool:
    """True iff the bid's price is provably past its decrease window.

    Derived from ``UserBid.last_updated``, which is bumped by any user
    update — including increases and no-op rewrites. If the bid has not
    been touched for at least the price decrease period, no price
    decrease can be sitting inside that window.

    A False answer is non-committal ("we can't tell from this alone")
    and must be resolved by fetching the bid's history.
    """
    return now - bid.last_updated >= settings.min_bid_price_decrease_period


def is_speed_guaranteed_free(
    bid: UserBid, settings: MarketSettings, now: datetime
) -> bool:
    """True iff the bid's speed limit is provably past its decrease window.

    Derived from ``UserBid.last_updated``, which is bumped by any user
    update — including increases and no-op rewrites. If the bid has not
    been touched for at least the speed decrease period, no speed
    decrease can be sitting inside that window.

    A False answer is non-committal ("we can't tell from this alone")
    and must be resolved by fetching the bid's history.
    """
    return now - bid.last_updated >= settings.min_bid_speed_limit_decrease_period


def cooldown_from_history(
    history: BidHistory,
    settings: MarketSettings,
    now: datetime,
) -> CooldownInfo:
    """Authoritative per-field cooldown status derived from bid history.

    Each flag is True iff the last decrease of that field occurred within
    its own window in ``settings``. Missing timestamp (no decrease ever,
    or none visible in history) → flag is False.
    """
    last_price = history.last_price_decrease_at()
    last_speed = history.last_speed_decrease_at()
    return CooldownInfo(
        price_cooldown=(
            last_price is not None
            and now - last_price < settings.min_bid_price_decrease_period
        ),
        speed_cooldown=(
            last_speed is not None
            and now - last_speed < settings.min_bid_speed_limit_decrease_period
        ),
    )


def plan_with_cooldowns(
    desired_price: HashratePrice,
    hashrate_to_set: Hashrate,
    max_bids_count: int,
    bids: tuple[BidWithCooldown, ...],
) -> tuple[BidConfig, ...]:
    """Build a bid plan that respects per-bid cooldown constraints.

    Rules:
      - speed_cooldown=True: keep the bid's current speed limit (cannot lower).
        Such a bid consumes one slot from `max_bids_count` and its current
        speed is subtracted from `needed`.
      - price_cooldown=True (and not speed_cooldown): keep the bid's current
        price; speed is freely re-assigned from the remaining distribution.
      - Bids with no cooldown are treated as fresh slots at `desired_price`.

    The remaining hashrate budget is split via `distribute_bids` and assigned
    first to price-locked bids (preserving their old price), then to brand-new
    slots at `desired_price`.
    """
    speed_locked = [b for b in bids if b.is_speed_in_cooldown]
    price_locked_only = [
        b for b in bids if b.is_price_in_cooldown and not b.is_speed_in_cooldown
    ]

    locked_speed_total = Hashrate(Decimal(0), HashUnit.PH, TimeUnit.SECOND)
    for entry in speed_locked:
        locked_speed_total = locked_speed_total + entry.bid.speed_limit_ph

    locked_entries = tuple(
        BidConfig(
            price=entry.bid.price if entry.is_price_in_cooldown else desired_price,
            speed_limit=entry.bid.speed_limit_ph,
        )
        for entry in speed_locked
    )

    if hashrate_to_set > locked_speed_total:
        remaining = hashrate_to_set - locked_speed_total
    else:
        remaining = Hashrate(Decimal(0), HashUnit.PH, TimeUnit.SECOND)

    remaining_slots = max(0, max_bids_count - len(speed_locked))
    speeds = distribute_bids(remaining, remaining_slots) if remaining_slots else ()

    free_entries: list[BidConfig] = []
    for i, speed in enumerate(speeds):
        if i < len(price_locked_only):
            entry = price_locked_only[i]
            free_entries.append(BidConfig(price=entry.bid.price, speed_limit=speed))
        else:
            free_entries.append(BidConfig(price=desired_price, speed_limit=speed))

    return locked_entries + tuple(free_entries)


def find_market_price(orderbook: OrderBook, tick: PriceTick) -> HashratePrice:
    """Lowest served bid, undercut (from above) by one price tick.

    The cheapest served price is aligned down to the tick grid first to
    guarantee the result lands on a valid tick.

    Raises:
        ValueError: If no bid in the order book has hr_matched_ph > 0.
    """
    served = [b for b in orderbook.bids if b.hr_matched_ph.value > 0]
    if not served:
        raise ValueError("Order book has no served bids; cannot pick a price")
    cheapest = min(served, key=lambda b: b.price.sats)
    return tick.add_one(tick.align_down(cheapest.price))
