"""Rate of satoshi consumption over time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from hashbidder.domain.sats import Sats


@dataclass(frozen=True)
class SatsBurnRate:
    """A non-negative rate of satoshi consumption.

    Stored as an `amount` of sats consumed over a `period`. Equivalent
    rates can be expressed over different periods via `to`.
    """

    amount: Decimal
    period: timedelta

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError(
                f"SatsBurnRate amount must be non-negative, got {self.amount}"
            )
        if self.period.total_seconds() <= 0:
            raise ValueError(f"SatsBurnRate period must be positive, got {self.period}")

    @classmethod
    def zero(cls) -> SatsBurnRate:
        """The zero burn rate."""
        return cls(Decimal(0), timedelta(days=1))

    def to(self, period: timedelta) -> SatsBurnRate:
        """Return an equivalent rate expressed over a different period."""
        scale = Decimal(period.total_seconds()) / Decimal(self.period.total_seconds())
        return SatsBurnRate(amount=self.amount * scale, period=period)

    def runway(self, available: Sats) -> timedelta:
        """Time before `available` sats would be exhausted at this rate.

        Returns `timedelta.max` when the rate is zero.
        """
        if self.amount == 0:
            return timedelta.max
        sats_per_second = self.amount / Decimal(self.period.total_seconds())
        seconds = Decimal(int(available)) / sats_per_second
        return timedelta(seconds=float(seconds))

    def __add__(self, other: SatsBurnRate) -> SatsBurnRate:
        return SatsBurnRate(
            amount=self.amount + other.to(self.period).amount,
            period=self.period,
        )
