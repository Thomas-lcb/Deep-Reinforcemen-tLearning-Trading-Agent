"""
tests/test_action.py — Tests unitaires de l'interprétation d'action
(env/action.py), y compris le short-selling (short_enabled=True).
"""

import pytest
from env.action import interpret_action, apply_cooldown


class TestLongOnlyUnchanged:
    """short_enabled=False (défaut) doit reproduire le comportement actuel."""

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
        trade = interpret_action(
            raw_action=-0.8,
            balance_usdt=10000.0,
            balance_asset=-0.05,  # déjà short 0.05 BTC
            asset_price=50000.0,
            short_enabled=True,
        )
        assert trade["type"] == "short"


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
