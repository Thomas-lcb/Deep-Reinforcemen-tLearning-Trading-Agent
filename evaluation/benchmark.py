"""
evaluation/benchmark.py — Stratégies de référence pour le backtest (Phase 4).

Deux baselines pour juger si un modèle entraîné vaut mieux qu'une stratégie
triviale :
- buy_and_hold_curve : achète une fois au début, tient jusqu'à la fin.
- random_baseline_curve : rejoue le VRAI CryptoTradingEnv (mêmes frais,
  même dead zone, mêmes règles) avec des actions aléatoires — comparaison
  à armes égales avec le modèle, pas une simulation séparée simplifiée.
"""

import numpy as np
import pandas as pd

from env.trading_env import CryptoTradingEnv


def buy_and_hold_curve(
    close_prices: np.ndarray,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
) -> np.ndarray:
    """
    Courbe d'equity d'une position achetée intégralement au premier prix
    (frais payés une fois à l'entrée) et jamais revendue.
    """
    close_prices = np.asarray(close_prices, dtype=float)
    if len(close_prices) == 0:
        return np.array([])
    effective_capital = initial_capital * (1.0 - fee_rate)
    units = effective_capital / close_prices[0]
    return units * close_prices


def random_baseline_curve(
    dfs: dict,
    config: dict,
    seed: int = 0,
    mode: str = "test",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Fait tourner un épisode complet du vrai CryptoTradingEnv avec des
    actions aléatoires (échantillonnées dans l'action space), en mode
    déterministe pour les données (pas de randomisation de domaine ni
    de plafond d'épisode en dehors de mode="train" — cf. trading_env.py).

    Returns:
        (equity_curve, trade_history) — equity_curve inclut la valeur au
        reset (capital initial) suivie de la valeur après chaque step.
    """
    env = CryptoTradingEnv(df=dfs, config=config, mode=mode)
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)

    equity = [info["portfolio_value"]]
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        equity.append(info["portfolio_value"])
        done = terminated or truncated

    return np.array(equity), env.get_trade_history()
