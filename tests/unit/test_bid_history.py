"""Tests for the BidHistory domain type."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from hashbidder.domain.bid_history import BidHistory, BidHistoryEntry
from hashbidder.domain.hashrate import Hashrate, HashratePrice, HashUnit
from hashbidder.domain.sats import Sats
from hashbidder.domain.time_unit import TimeUnit

EH_DAY = Hashrate(Decimal(1), HashUnit.EH, TimeUnit.DAY)
T0 = datetime(2026, 4, 17, 8, 0, tzinfo=UTC)


def _entry(t: datetime, price_sat: int, speed: str) -> BidHistoryEntry:
    return BidHistoryEntry(
        timestamp=t,
        price=HashratePrice(sats=Sats(price_sat), per=EH_DAY),
        speed_limit_ph=Hashrate(Decimal(speed), HashUnit.PH, TimeUnit.SECOND),
    )


class TestBidHistoryNormalisation:
    """Tests for the newest-first sort invariant."""

    def test_out_of_order_input_is_sorted(self) -> None:
        """Entries given older-first are reordered to newest-first."""
        older = _entry(T0, 500_000, "5")
        newer = _entry(T0 + timedelta(minutes=1), 500_000, "5")
        history = BidHistory(entries=(older, newer))
        assert history.entries == (newer, older)

    def test_shuffled_input_is_sorted(self) -> None:
        """Arbitrarily-ordered input is normalised deterministically."""
        a = _entry(T0, 500_000, "5")
        b = _entry(T0 + timedelta(minutes=1), 500_000, "5")
        c = _entry(T0 + timedelta(minutes=2), 500_000, "5")
        history = BidHistory(entries=(b, a, c))
        assert history.entries == (c, b, a)


class TestLastPriceDecreaseAt:
    """Tests for BidHistory.last_price_decrease_at."""

    def test_empty_history(self) -> None:
        """No entries → None."""
        assert BidHistory(entries=()).last_price_decrease_at() is None

    def test_single_entry(self) -> None:
        """One entry has no predecessor → None."""
        history = BidHistory(entries=(_entry(T0, 500_000, "5"),))
        assert history.last_price_decrease_at() is None

    def test_monotone_increasing(self) -> None:
        """Price only rises → None."""
        history = BidHistory(
            entries=(
                _entry(T0, 500_000, "5"),
                _entry(T0 + timedelta(minutes=1), 510_000, "5"),
                _entry(T0 + timedelta(minutes=2), 520_000, "5"),
            )
        )
        assert history.last_price_decrease_at() is None

    def test_single_decrease(self) -> None:
        """One drop → its timestamp."""
        t1 = T0 + timedelta(minutes=1)
        history = BidHistory(
            entries=(_entry(T0, 500_000, "5"), _entry(t1, 400_000, "5"))
        )
        assert history.last_price_decrease_at() == t1

    def test_up_down_up_picks_middle_event(self) -> None:
        """A decrease sandwiched between increases is still the 'last' one."""
        t1 = T0 + timedelta(minutes=1)
        t2 = T0 + timedelta(minutes=2)  # ← the decrease
        t3 = T0 + timedelta(minutes=3)
        history = BidHistory(
            entries=(
                _entry(T0, 500_000, "5"),
                _entry(t1, 600_000, "5"),
                _entry(t2, 450_000, "5"),
                _entry(t3, 700_000, "5"),
            )
        )
        assert history.last_price_decrease_at() == t2

    def test_noop_entries_ignored(self) -> None:
        """Equal-valued entries are not decreases; the real drop wins."""
        t1 = T0 + timedelta(minutes=1)  # real decrease
        t2 = T0 + timedelta(minutes=2)  # no-op
        t3 = T0 + timedelta(minutes=3)  # no-op
        history = BidHistory(
            entries=(
                _entry(T0, 500_000, "5"),
                _entry(t1, 400_000, "5"),
                _entry(t2, 400_000, "5"),
                _entry(t3, 400_000, "5"),
            )
        )
        assert history.last_price_decrease_at() == t1

    def test_unaffected_by_speed_changes(self) -> None:
        """Only price transitions are considered."""
        t1 = T0 + timedelta(minutes=1)
        history = BidHistory(
            entries=(_entry(T0, 500_000, "10"), _entry(t1, 500_000, "5"))
        )
        assert history.last_price_decrease_at() is None


class TestLastSpeedDecreaseAt:
    """Tests for BidHistory.last_speed_decrease_at."""

    def test_empty_history(self) -> None:
        """No entries → None."""
        assert BidHistory(entries=()).last_speed_decrease_at() is None

    def test_single_entry(self) -> None:
        """One entry has no predecessor → None."""
        history = BidHistory(entries=(_entry(T0, 500_000, "5"),))
        assert history.last_speed_decrease_at() is None

    def test_monotone_increasing(self) -> None:
        """Speed only rises → None."""
        history = BidHistory(
            entries=(
                _entry(T0, 500_000, "5"),
                _entry(T0 + timedelta(minutes=1), 500_000, "6"),
                _entry(T0 + timedelta(minutes=2), 500_000, "7"),
            )
        )
        assert history.last_speed_decrease_at() is None

    def test_single_decrease(self) -> None:
        """One drop → its timestamp."""
        t1 = T0 + timedelta(minutes=1)
        history = BidHistory(
            entries=(_entry(T0, 500_000, "10"), _entry(t1, 500_000, "5"))
        )
        assert history.last_speed_decrease_at() == t1

    def test_up_down_up_picks_middle_event(self) -> None:
        """A decrease sandwiched between increases is still the 'last' one."""
        t1 = T0 + timedelta(minutes=1)
        t2 = T0 + timedelta(minutes=2)  # ← the decrease
        t3 = T0 + timedelta(minutes=3)
        history = BidHistory(
            entries=(
                _entry(T0, 500_000, "5"),
                _entry(t1, 500_000, "10"),
                _entry(t2, 500_000, "3"),
                _entry(t3, 500_000, "15"),
            )
        )
        assert history.last_speed_decrease_at() == t2

    def test_noop_entries_ignored(self) -> None:
        """Equal-valued entries are not decreases; the real drop wins."""
        t1 = T0 + timedelta(minutes=1)  # real decrease
        t2 = T0 + timedelta(minutes=2)  # no-op
        t3 = T0 + timedelta(minutes=3)  # no-op
        history = BidHistory(
            entries=(
                _entry(T0, 500_000, "10"),
                _entry(t1, 500_000, "5"),
                _entry(t2, 500_000, "5"),
                _entry(t3, 500_000, "5"),
            )
        )
        assert history.last_speed_decrease_at() == t1

    def test_unaffected_by_price_changes(self) -> None:
        """Only speed transitions are considered."""
        t1 = T0 + timedelta(minutes=1)
        history = BidHistory(
            entries=(_entry(T0, 500_000, "5"), _entry(t1, 400_000, "5"))
        )
        assert history.last_speed_decrease_at() is None


class TestBothFieldsTogether:
    """Tests that confirm the two methods are independent."""

    def test_distinct_decrease_timestamps(self) -> None:
        """Price and speed decreases are reported independently."""
        t1 = T0 + timedelta(minutes=1)  # price drops
        t2 = T0 + timedelta(minutes=2)  # nothing
        t3 = T0 + timedelta(minutes=3)  # speed drops
        history = BidHistory(
            entries=(
                _entry(T0, 500_000, "10"),
                _entry(t1, 400_000, "10"),
                _entry(t2, 400_000, "10"),
                _entry(t3, 400_000, "5"),
            )
        )
        assert history.last_price_decrease_at() == t1
        assert history.last_speed_decrease_at() == t3
