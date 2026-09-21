"""
tests/test_action.py — Tests unitaires de l'interprétation d'action
(env/action.py), y compris le short-selling (short_enabled=True).
"""

import pytest
from env.action import interpret_action, apply_cooldown


class TestLongOnlyUnchanged:
    """short_enabled=False (défaut): behavior is functionally equivalent to
    before (no-op in both cases), though the 'type' label changed in the
    flat-sell edge case (hold vs. old zero-amount sell). Buy/sell when
    positioned remain unchanged."""

    def test_sell_while_flat_is_hold_without_short(self):
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            short_enabled=False,
        )
        assert trade["type"] == "hold"

    def test_buy_unchanged(self):
        trade = interpret_action(
            raw_action=0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=1.0,
        )
        assert trade["type"] == "buy"
        assert trade["amount_asset"] > 0


class TestShortOpen:
    def test_sell_while_flat_opens_short_when_enabled(self):
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=1.0,
            short_enabled=True,
        )
        assert trade["type"] == "short"
        assert trade["amount_asset"] > 0
        assert trade["fee"] > 0
        # amount_usdt = produit NET (après frais), comme "sell"
        gross = trade["amount_asset"] * 50000.0
        assert trade["amount_usdt"] == pytest.approx(gross - trade["fee"], rel=1e-6)

    def test_short_capped_by_max_position_pct(self):
        trade = interpret_action(
            raw_action=-1.0,  # pleine puissance
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=0.25,
            short_enabled=True,
        )
        assert trade["proportion"] == pytest.approx(0.25)
        notional = trade["amount_asset"] * 50000.0
        assert notional == pytest.approx(10000.0 * 0.25, rel=1e-6)

    def test_sell_while_already_short_increases_short(self):
        # equity = 10000 + (-0.05 * 50000) = 7500
        # existing_short_notional = 0.05 * 50000 = 2500
        # available = 7500 - 2500 = 5000
        balance_usdt = 10000.0
        balance_asset = -0.05  # déjà short 0.05 BTC
        asset_price = 50000.0
        dead_zone = 0.05
        action = -0.8

        trade = interpret_action(
            raw_action=action,
            balance_usdt=balance_usdt,
            balance_asset=balance_asset,
            asset_price=asset_price,
            dead_zone=dead_zone,
            short_enabled=True,
        )
        assert trade["type"] == "short"

        proportion = (abs(action) - dead_zone) / (1.0 - dead_zone)
        equity = balance_usdt + balance_asset * asset_price
        existing_short_notional = abs(min(balance_asset, 0.0)) * asset_price
        available = max(0.0, equity - existing_short_notional)
        expected_notional = available * proportion

        notional = trade["amount_asset"] * asset_price
        # Sized off the reduced `available` equity, not the inflated
        # balance_usdt (this is exactly the case Fix 1 changes).
        assert notional == pytest.approx(expected_notional, rel=1e-6)
        assert notional < balance_usdt * proportion  # would have been inflated pre-fix


    def test_stacked_shorts_notional_bounded_by_equity(self):
        """
        Regression test for the unbounded-leverage bug: opening a short ADDS
        its net proceeds to balance_usdt (unlike a buy, which SPENDS it and
        therefore self-limits), so sizing off raw balance_usdt let repeated
        shorts compound the very balance used to size the next short. Fix 1
        sizes against equity minus existing short exposure instead, so total
        short notional should never exceed the portfolio's equity (1x cap).
        """
        asset_price = 50000.0
        balance_usdt = 10000.0
        balance_asset = 0.0

        # First short at max size.
        first = interpret_action(
            raw_action=-1.0,
            balance_usdt=balance_usdt,
            balance_asset=balance_asset,
            asset_price=asset_price,
            short_enabled=True,
        )
        assert first["type"] == "short"

        balance_usdt_after_1 = balance_usdt + first["amount_usdt"]
        balance_asset_after_1 = balance_asset - first["amount_asset"]

        # Second stacked short at max size, fed by the post-first-short state.
        second = interpret_action(
            raw_action=-1.0,
            balance_usdt=balance_usdt_after_1,
            balance_asset=balance_asset_after_1,
            asset_price=asset_price,
            short_enabled=True,
        )
        assert second["type"] == "short"

        balance_asset_after_2 = balance_asset_after_1 - second["amount_asset"]

        equity = balance_usdt + balance_asset * asset_price  # initial equity
        total_notional = abs(balance_asset_after_2) * asset_price

        # Total short exposure must not exceed the portfolio's (initial)
        # equity by more than a small floating-point tolerance — i.e. no
        # unbounded leverage from stacking shorts.
        assert total_notional <= equity + 1e-6


class TestCover:
    def test_buy_while_short_covers(self):
        trade = interpret_action(
            raw_action=0.8,
            balance_usdt=10000.0,
            balance_asset=-0.1,  # short 0.1 BTC
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=1.0,
            short_enabled=True,
        )
        assert trade["type"] == "cover"
        assert trade["amount_asset"] > 0
        assert trade["amount_asset"] <= 0.1 + 1e-9
        # amount_usdt = cout total INCLUANT les frais, comme "buy"
        cost = trade["amount_asset"] * 50000.0
        assert trade["amount_usdt"] == pytest.approx(cost + trade["fee"], rel=1e-6)

    def test_cover_proportional_to_short_size(self):
        trade = interpret_action(
            raw_action=1.0,  # pleine puissance -> proportion cappee a max_position_pct
            balance_usdt=10000.0,
            balance_asset=-0.2,
            asset_price=50000.0,
            dead_zone=0.05,
            fee_rate=0.001,
            max_position_pct=0.5,
            short_enabled=True,
        )
        assert trade["proportion"] == pytest.approx(0.5)
        assert trade["amount_asset"] == pytest.approx(0.2 * 0.5, rel=1e-6)


class TestCooldownStillWorksWithNewTypes:
    def test_cooldown_blocks_short(self):
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=0.0,
            asset_price=50000.0,
            short_enabled=True,
        )
        blocked = apply_cooldown(steps_since_trade=0, cooldown_steps=5, trade=trade)
        assert blocked["type"] == "hold"
