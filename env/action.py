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
    short_enabled: bool = False,
) -> dict:
    """
    Interpret the agent's raw action into a concrete trade.

    Args:
        raw_action: Value in [-1, 1] from the agent.
        balance_usdt: Current USDT balance.
        balance_asset: Current asset quantity. Negative = short position
            (only possible when short_enabled has been True in the past).
        asset_price: Current asset price.
        dead_zone: Actions within [-dead_zone, +dead_zone] are treated as Hold.
        fee_rate: Transaction fee rate (e.g. 0.001 = 0.1%).
        max_position_pct: Maximum proportion of capital per trade (e.g. 0.25 = 25%).
        short_enabled: If False (default), a negative action while flat or
            short does nothing (today's long-only behavior, unchanged bit
            for bit). If True, it opens/increases a short position.

    Returns:
        Dict with keys:
        - 'type': 'buy' | 'sell' | 'short' | 'cover' | 'hold'
        - 'amount_usdt': for 'buy'/'cover', total cash outlay INCLUDING fee.
          for 'sell'/'short', net proceeds AFTER fee.
        - 'amount_asset': asset quantity traded.
        - 'proportion': effective proportion of capital/position used.
        - 'fee': fee for this trade.
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
        proportion = (action - dead_zone) / (1.0 - dead_zone)
        proportion = min(proportion, max_position_pct)

        if balance_asset < 0:
            # COVER: buy back a fraction of the existing short.
            amount_asset = abs(balance_asset) * proportion
            cost = amount_asset * asset_price
            fee = cost * fee_rate
            amount_usdt = cost + fee

            return {
                "type": "cover",
                "amount_usdt": amount_usdt,
                "amount_asset": amount_asset,
                "proportion": proportion,
                "fee": fee,
            }

        # BUY: scale proportion from 0 to 1 over [dead_zone, 1]
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
        proportion = (abs(action) - dead_zone) / (1.0 - dead_zone)
        proportion = min(proportion, max_position_pct)

        if balance_asset <= 0:
            if not short_enabled:
                # Unchanged today's behavior: nothing to sell while flat.
                return {
                    "type": "hold",
                    "amount_usdt": 0.0,
                    "amount_asset": 0.0,
                    "proportion": 0.0,
                    "fee": 0.0,
                }

            # SHORT: open/increase a short position.
            notional = balance_usdt * proportion
            amount_asset = notional / asset_price if asset_price > 0 else 0.0
            gross_usdt = amount_asset * asset_price
            fee = gross_usdt * fee_rate
            amount_usdt = gross_usdt - fee

            return {
                "type": "short",
                "amount_usdt": amount_usdt,
                "amount_asset": amount_asset,
                "proportion": proportion,
                "fee": fee,
            }

        # SELL: scale proportion from 0 to 1 over [-1, -dead_zone]
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
