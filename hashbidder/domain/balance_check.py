"""Balance check: can the account fund the creates in a reconciliation plan?"""

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from enum import Enum

from hashbidder.domain.bid_planning import ReconciliationPlan
from hashbidder.domain.hashrate import HashUnit
from hashbidder.domain.sats import Sats
from hashbidder.domain.sats_burn_rate import SatsBurnRate
from hashbidder.domain.time_unit import TimeUnit

# Warn when the runway at the planned burn rate drops below this duration.
LOW_BALANCE_RUNWAY = timedelta(hours=72)

_ONE_DAY = timedelta(days=1)


class BalanceStatus(Enum):
    """Outcome of a balance check."""

    SUFFICIENT = "sufficient"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class BalanceCheck:
    """Result of checking the account balance against a plan.

    `runway` is `timedelta.max` when the plan has no creates.
    """

    required_sat: Sats
    available_sat: Sats
    burn_rate: SatsBurnRate
    runway: timedelta
    status: BalanceStatus


def _plan_burn_rate(plan: ReconciliationPlan) -> SatsBurnRate:
    """Sum the burn rate across all planned creates."""
    total = SatsBurnRate.zero()
    for create in plan.creates:
        speed_eh_per_day = create.config.speed_limit.to(HashUnit.EH, TimeUnit.DAY).value
        price_sat_per_eh_day = create.config.price.to(HashUnit.EH, TimeUnit.DAY).sats
        total += SatsBurnRate(
            amount=speed_eh_per_day * Decimal(price_sat_per_eh_day),
            period=_ONE_DAY,
        )
    return total


def check_balance(plan: ReconciliationPlan, available_sat: Sats) -> BalanceCheck:
    """Compare the account balance to the sats needed to fund plan creates.

    `INSUFFICIENT` if the balance cannot cover the creates' `amount_sat`.
    `LOW` if covered but the runway at the planned burn rate is under
    `LOW_BALANCE_RUNWAY`. `SUFFICIENT` otherwise.
    """
    required = sum((int(c.amount) for c in plan.creates), start=0)
    burn_rate = _plan_burn_rate(plan)
    runway = burn_rate.runway(available_sat)

    if available_sat < required:
        status = BalanceStatus.INSUFFICIENT
    elif runway < LOW_BALANCE_RUNWAY:
        status = BalanceStatus.LOW
    else:
        status = BalanceStatus.SUFFICIENT

    return BalanceCheck(
        required_sat=Sats(required),
        available_sat=available_sat,
        burn_rate=burn_rate,
        runway=runway,
        status=status,
    )
