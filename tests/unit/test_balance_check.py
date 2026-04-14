"""Tests for the balance check domain module."""

from datetime import timedelta
from decimal import Decimal

from hashbidder.domain.balance_check import (
    LOW_BALANCE_RUNWAY,
    BalanceStatus,
    check_balance,
)
from hashbidder.domain.bid_planning import plan_bid_changes
from hashbidder.domain.sats import Sats
from tests.conftest import make_bid_config, make_config

# Burn rate for one create at 500 sat/PH/Day and 5 PH/s:
#   speed in EH/Day  = 5 PH/s * 86400 s/day / 1000 PH/EH = 432 EH/Day
#   price in sat/EH/Day = 500 * 1000 = 500_000
#   cost/day = 432 * 500_000 = 216_000_000 sat
#   cost/hour = 9_000_000 sat
_BURN_RATE_SAT_PER_HOUR = 9_000_000
_BURN_RATE_SAT_PER_DAY = _BURN_RATE_SAT_PER_HOUR * 24


class TestCheckBalance:
    """Tests for check_balance."""

    def test_no_creates_is_sufficient(self) -> None:
        """An empty plan needs no funds and has no burn rate."""
        plan = plan_bid_changes(make_config(), ())

        result = check_balance(plan, Sats(0))

        assert result.status == BalanceStatus.SUFFICIENT
        assert result.required_sat == 0
        assert result.burn_rate.amount == Decimal(0)
        assert result.runway == timedelta.max

    def test_sufficient_balance_with_long_runway(self) -> None:
        """Funds cover creates and runway comfortably exceeds the threshold."""
        plan = plan_bid_changes(make_config(make_bid_config(500, "5.0")), ())
        # 100h of runway at 9M sat/hour.
        available = Sats(_BURN_RATE_SAT_PER_HOUR * 100)

        result = check_balance(plan, available)

        assert result.status == BalanceStatus.SUFFICIENT
        assert result.required_sat == 100_000  # default amount
        assert result.burn_rate.amount == Decimal(_BURN_RATE_SAT_PER_DAY)
        assert result.runway == timedelta(hours=100)

    def test_short_runway_is_low(self) -> None:
        """Funds cover creates but runway is under the warning threshold."""
        plan = plan_bid_changes(make_config(make_bid_config(500, "5.0")), ())
        # 71h of runway (< 72h threshold).
        available = Sats(_BURN_RATE_SAT_PER_HOUR * 71)

        result = check_balance(plan, available)

        assert result.status == BalanceStatus.LOW
        assert result.runway == timedelta(hours=71)

    def test_exactly_threshold_runway_is_sufficient(self) -> None:
        """Runway exactly equal to the threshold is not flagged as LOW."""
        plan = plan_bid_changes(make_config(make_bid_config(500, "5.0")), ())
        available = Sats(_BURN_RATE_SAT_PER_HOUR * 72)

        result = check_balance(plan, available)

        assert result.status == BalanceStatus.SUFFICIENT
        assert result.runway == LOW_BALANCE_RUNWAY

    def test_balance_below_required_is_insufficient(self) -> None:
        """Balance smaller than the creates' amount_sat is INSUFFICIENT."""
        plan = plan_bid_changes(make_config(make_bid_config(500, "5.0")), ())
        # default_amount is 100_000; 50_000 is not enough to fund the create.
        result = check_balance(plan, Sats(50_000))

        assert result.status == BalanceStatus.INSUFFICIENT
        assert result.required_sat == 100_000
        assert result.available_sat == 50_000

    def test_insufficient_takes_precedence_over_low(self) -> None:
        """INSUFFICIENT wins even when runway is also below threshold."""
        plan = plan_bid_changes(
            make_config(
                make_bid_config(500, "5.0"),
                make_bid_config(500, "5.0"),
            ),
            (),
        )
        # required = 2 * 100_000 = 200_000; not enough.
        result = check_balance(plan, Sats(100_000))

        assert result.status == BalanceStatus.INSUFFICIENT
        assert result.required_sat == 200_000

    def test_burn_rate_sums_across_creates(self) -> None:
        """Burn rate is the sum over all planned creates."""
        plan = plan_bid_changes(
            make_config(
                make_bid_config(500, "5.0"),
                make_bid_config(500, "5.0"),
            ),
            (),
        )
        available = Sats(_BURN_RATE_SAT_PER_HOUR * 2 * 1000)

        result = check_balance(plan, available)

        assert result.burn_rate.amount == Decimal(_BURN_RATE_SAT_PER_DAY * 2)
        assert result.required_sat == 200_000
