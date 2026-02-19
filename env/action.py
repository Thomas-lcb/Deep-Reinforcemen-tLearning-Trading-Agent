"""
env/action.py — Logique de l'Action Space avec dead zone.

L'agent produit une valeur continue dans [-1, 1].
- [-1, -dead_zone] → Vendre (proportionnel)
- [-dead_zone, +dead_zone] → Hold (rien faire)
- [+dead_zone, +1] → Acheter (proportionnel)

cf. agent.md §5.B
"""

import numpy as np
import gymnasium as gym


def build_action_space() -> gym.spaces.Box:
    """
    Create the continuous action space: Box([-1], [1]).

    Returns:
        gym.spaces.Box action space.
    """
    return gym.spaces.Box(
        low=np.array([-1.0], dtype=np.float32),
        high=np.array([1.0], dtype=np.float32),
        shape=(1,),
        dtype=np.float32,
    )


def interpret_action(
    raw_action: float,
    balance_usdt: float,
    balance_asset: float,
    asset_price: float,
    dead_zone: float = 0.05,
    fee_rate: float = 0.001,
    max_position_pct: float = 1.0,
) -> dict:
    """
    Interpret the agent's raw action into a concrete trade.

    Args:
        raw_action: Value in [-1, 1] from the agent.
        balance_usdt: Current USDT balance.
        balance_asset: Current asset quantity.
        asset_price: Current asset price.
        dead_zone: Actions within [-dead_zone, +dead_zone] are treated as Hold.
        fee_rate: Transaction fee rate (e.g. 0.001 = 0.1%).
        max_position_pct: Maximum proportion of capital per trade (e.g. 0.25 = 25%).

    Returns:
        Dict with keys:
        - 'type': 'buy' | 'sell' | 'hold'
        - 'amount_usdt': USDT amount to trade (for buy)
        - 'amount_asset': Asset amount to trade (for sell)
        - 'proportion': Effective proportion of capital used
        - 'fee': Estimated fee for this trade
    """
    # Clamp action
    action = float(np.clip(raw_action, -1.0, 1.0))

    # Dead zone → Hold
    if abs(action) <= dead_zone:
        return {
            "type": "hold",
            "amount_usdt": 0.0,
            "amount_asset": 0.0,
            "proportion": 0.0,
            "fee": 0.0,
        }

    if action > dead_zone:
        # BUY: scale proportion from 0 to 1 over [dead_zone, 1]
        proportion = (action - dead_zone) / (1.0 - dead_zone)
        # Cap at max_position_pct
        proportion = min(proportion, max_position_pct)
        amount_usdt = balance_usdt * proportion

        # Account for fees: we can only buy (amount / (1 + fee))
        effective_usdt = amount_usdt / (1.0 + fee_rate)
        amount_asset = effective_usdt / asset_price if asset_price > 0 else 0.0
        fee = amount_usdt - effective_usdt

        return {
            "type": "buy",
            "amount_usdt": amount_usdt,
            "amount_asset": amount_asset,
            "proportion": proportion,
            "fee": fee,
        }

    else:
        # SELL: scale proportion from 0 to 1 over [-1, -dead_zone]
        proportion = (abs(action) - dead_zone) / (1.0 - dead_zone)
        # Cap at max_position_pct
        proportion = min(proportion, max_position_pct)
        amount_asset = balance_asset * proportion

        # Revenue after fees
        gross_usdt = amount_asset * asset_price
        fee = gross_usdt * fee_rate
        net_usdt = gross_usdt - fee

        return {
            "type": "sell",
            "amount_usdt": net_usdt,
            "amount_asset": amount_asset,
            "proportion": proportion,
            "fee": fee,
        }


def apply_cooldown(
    steps_since_trade: int,
    cooldown_steps: int,
    trade: dict,
) -> dict:
    """
    Enforce a minimum cooldown between trades.
    If the cooldown hasn't expired, force the action to Hold.

    Args:
        steps_since_trade: Steps since the last executed trade.
        cooldown_steps: Minimum steps between trades (0 = disabled).
        trade: Trade dict from interpret_action().

    Returns:
        Original trade or Hold trade if cooldown active.
    """
    if cooldown_steps <= 0:
        return trade

    if trade["type"] != "hold" and steps_since_trade < cooldown_steps:
        return {
            "type": "hold",
            "amount_usdt": 0.0,
            "amount_asset": 0.0,
            "proportion": 0.0,
            "fee": 0.0,
        }

    return trade
