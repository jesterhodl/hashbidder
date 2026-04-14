"""Tests for the SatsBurnRate primitive."""

from datetime import timedelta
from decimal import Decimal

import pytest

from hashbidder.domain.sats import Sats
from hashbidder.domain.sats_burn_rate import SatsBurnRate


class TestSatsBurnRate:
    """Tests for SatsBurnRate."""

    def test_zero_is_zero(self) -> None:
        """The zero constructor yields a zero-amount rate."""
        rate = SatsBurnRate.zero()
        assert rate.amount == Decimal(0)

    def test_negative_amount_rejected(self) -> None:
        """Negative amounts are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            SatsBurnRate(Decimal(-1), timedelta(days=1))

    def test_non_positive_period_rejected(self) -> None:
        """A zero-length period is rejected."""
        with pytest.raises(ValueError, match="period"):
            SatsBurnRate(Decimal(100), timedelta(0))

    def test_to_converts_between_periods(self) -> None:
        """Converting between periods scales the amount linearly."""
        # 2400 sat/day = 100 sat/hour.
        rate = SatsBurnRate(Decimal(2400), timedelta(days=1))
        hourly = rate.to(timedelta(hours=1))
        assert hourly.amount == Decimal(100)
        assert hourly.period == timedelta(hours=1)

    def test_to_roundtrip(self) -> None:
        """Converting through another period and back is lossless."""
        rate = SatsBurnRate(Decimal(2400), timedelta(days=1))
        assert rate.to(timedelta(hours=1)).to(timedelta(days=1)).amount == Decimal(2400)

    def test_addition_same_period(self) -> None:
        """Rates with the same period sum by amount."""
        a = SatsBurnRate(Decimal(100), timedelta(days=1))
        b = SatsBurnRate(Decimal(250), timedelta(days=1))
        total = a + b
        assert total.amount == Decimal(350)
        assert total.period == timedelta(days=1)

    def test_addition_normalizes_periods(self) -> None:
        """Rates with different periods are normalized before summing."""
        # 24 sat/day + 1 sat/hour = 24 + 24 = 48 sat/day.
        a = SatsBurnRate(Decimal(24), timedelta(days=1))
        b = SatsBurnRate(Decimal(1), timedelta(hours=1))
        total = a + b
        assert total.amount == Decimal(48)
        assert total.period == timedelta(days=1)

    def test_runway_zero_rate_is_max(self) -> None:
        """A zero rate yields timedelta.max runway."""
        rate = SatsBurnRate.zero()
        assert rate.runway(Sats(1_000_000)) == timedelta.max

    def test_runway_simple(self) -> None:
        """Runway at a known rate matches hand calculation."""
        # 2400 sat/day = 100 sat/hour → 500 sat lasts 5 hours.
        rate = SatsBurnRate(Decimal(2400), timedelta(days=1))
        assert rate.runway(Sats(500)) == timedelta(hours=5)

    def test_runway_zero_balance(self) -> None:
        """A zero balance has zero runway against a non-zero rate."""
        rate = SatsBurnRate(Decimal(2400), timedelta(days=1))
        assert rate.runway(Sats(0)) == timedelta(0)
