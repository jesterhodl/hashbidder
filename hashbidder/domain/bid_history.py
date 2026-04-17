"""Bid history domain types."""

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from hashbidder.domain.hashrate import Hashrate, HashratePrice


@dataclass(frozen=True)
class BidHistoryEntry:
    """A single point in a bid's per-field value history.

    Captures the price and speed limit at a given moment so consumers can
    walk adjacent entries to detect strictly-downward transitions.
    """

    timestamp: datetime
    price: HashratePrice
    speed_limit_ph: Hashrate


@dataclass(frozen=True)
class BidHistory:
    """A bid's history, normalised newest-first at construction.

    Sorting is done once here so every query method can walk adjacent
    entries without re-sorting or trusting the server's order.
    """

    entries: tuple[BidHistoryEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(self.entries, key=lambda e: e.timestamp, reverse=True)),
        )

    def last_price_decrease_at(self) -> datetime | None:
        """Timestamp of the most recent strict price decrease, or None.

        A decrease is a transition from an older entry to a newer entry
        whose ``price.sats`` is strictly smaller.
        """
        for newer, older in pairwise(self.entries):
            if newer.price.sats < older.price.sats:
                return newer.timestamp
        return None

    def last_speed_decrease_at(self) -> datetime | None:
        """Timestamp of the most recent strict speed decrease, or None.

        A decrease is a transition from an older entry to a newer entry
        whose ``speed_limit_ph`` is strictly smaller.
        """
        for newer, older in pairwise(self.entries):
            if newer.speed_limit_ph < older.speed_limit_ph:
                return newer.timestamp
        return None
