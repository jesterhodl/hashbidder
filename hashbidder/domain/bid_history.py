"""Bid history entry domain type."""

from dataclasses import dataclass
from datetime import datetime

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
