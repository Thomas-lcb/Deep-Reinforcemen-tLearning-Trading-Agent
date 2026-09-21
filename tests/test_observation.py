"""
tests/test_observation.py — Tests unitaires de la construction
de l'observation (env/observation.py), y compris le PnL latent
d'une position courte.
"""

import numpy as np
import pytest

from env.observation import get_observation


def _base_kwargs(**overrides):
    kwargs = dict(
        market_data=np.zeros((10, 3), dtype=np.float32),
        balance_usdt=5000.0,
        balance_asset=0.0,
        asset_price=50000.0,
        entry_price=0.0,
        steps_since_trade=0,
        initial_capital=10000.0,
        lookback_window=10,
    )
    kwargs.update(overrides)
    return kwargs


class TestUnrealizedPnlPct:
    def test_long_position_positive_pnl(self):
        obs = get_observation(**_base_kwargs(
            balance_asset=0.1, entry_price=40000.0, asset_price=50000.0,
        ))
        # unrealized_pnl_pct est la 3e colonne du vecteur portefeuille
        pnl = obs[0, -2]
        assert pnl == pytest.approx((50000.0 - 40000.0) / 40000.0, rel=1e-5)

    def test_flat_position_zero_pnl(self):
        obs = get_observation(**_base_kwargs(balance_asset=0.0, entry_price=0.0))
        pnl = obs[0, -2]
        assert pnl == pytest.approx(0.0)

    def test_short_position_pnl_inverted(self):
        # Short ouvert a 50000, prix retombe a 45000 -> profit pour le short
        obs = get_observation(**_base_kwargs(
            balance_asset=-0.1, entry_price=50000.0, asset_price=45000.0,
        ))
        pnl = obs[0, -2]
        expected = (50000.0 - 45000.0) / 50000.0
        assert pnl == pytest.approx(expected, rel=1e-5)
        assert pnl > 0  # le prix a baisse, le short est gagnant

    def test_short_position_losing(self):
        # Short ouvert a 50000, prix monte a 55000 -> perte pour le short
        obs = get_observation(**_base_kwargs(
            balance_asset=-0.1, entry_price=50000.0, asset_price=55000.0,
        ))
        pnl = obs[0, -2]
        assert pnl < 0
