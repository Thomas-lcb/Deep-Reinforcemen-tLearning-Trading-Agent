"""
evaluation/metrics.py — Métriques de performance pour le backtest (Phase 4).

Fonctions pures : prennent une courbe d'equity (portfolio_value au fil du
temps) et/ou un trade_history (DataFrame avec une colonne pnl_pct, NaN pour
les trades d'ouverture, float pour les trades de clôture — cf.
env/trading_env.py::_log_trade), retournent un scalaire.

Les ratios annualisés (Sharpe, Sortino, Calmar) supposent un taux sans
risque nul (convention standard en crypto) et paramètrent periods_per_year
selon la granularité des données (525 600 pour des bougies 1 minute).
"""

import numpy as np
import pandas as pd


def compute_returns(equity_curve: np.ndarray) -> np.ndarray:
    """Rendements simples pas-à-pas à partir d'une courbe d'equity."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) < 2:
        return np.array([])
    return equity_curve[1:] / equity_curve[:-1] - 1.0


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 525_600,
    risk_free: float = 0.0,
) -> float:
    """
    Sharpe ratio annualisé. Retourne 0.0 (pas 1.0/inf) quand la volatilité
    est nulle ou qu'il n'y a pas assez de données — le ratio est indéfini
    dans ce cas, même convention de garde que env/reward.py's sharpe_bonus.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    std = np.std(excess)
    if std < 1e-8:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = 525_600,
    risk_free: float = 0.0,
) -> float:
    """
    Sortino ratio annualisé : comme Sharpe mais ne pénalise que la
    volatilité à la baisse (rendements < 0). Retourne 0.0 si aucun
    rendement négatif (pas de downside à mesurer) ou pas assez de données.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-8:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Perte maximale depuis un plus-haut, en fraction positive (0.25 = -25%).
    0.0 si la courbe ne baisse jamais sous son plus-haut courant.
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (running_peak - equity_curve) / running_peak
    return float(np.max(drawdowns))


def calmar_ratio(equity_curve: np.ndarray, periods_per_year: int = 525_600) -> float:
    """
    Rendement annualisé / drawdown maximal. Retourne 0.0 si le drawdown
    maximal est nul (ratio indéfini) ou pas assez de données.
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    n = len(equity_curve) - 1
    if n < 1 or equity_curve[0] <= 0:
        return 0.0
    mdd = max_drawdown(equity_curve)
    if mdd < 1e-8:
        return 0.0
    annualized_return = (equity_curve[-1] / equity_curve[0]) ** (periods_per_year / n) - 1.0
    return float(annualized_return / mdd)


def win_rate(trade_history: pd.DataFrame) -> float:
    """
    Fraction de trades de CLÔTURE (pnl_pct non-NaN) gagnants. NaN si aucun
    trade de clôture n'existe (le taux est indéfini, pas 0).
    """
    closed = trade_history["pnl_pct"].dropna()
    if len(closed) == 0:
        return float("nan")
    return float((closed > 0).sum() / len(closed))


def profit_factor(trade_history: pd.DataFrame) -> float:
    """
    Somme des gains / |somme des pertes|, sur les trades de clôture
    uniquement. inf si aucune perte, NaN si aucun trade de clôture.
    """
    closed = trade_history["pnl_pct"].dropna()
    if len(closed) == 0:
        return float("nan")
    gains = closed[closed > 0].sum()
    losses = closed[closed < 0].sum()
    if losses == 0:
        return float("inf")
    return float(gains / abs(losses))


def summarize(
    equity_curve: np.ndarray,
    trade_history: pd.DataFrame,
    periods_per_year: int = 525_600,
) -> dict:
    """Calcule toutes les métriques ci-dessus en un seul dict, prêt à afficher."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    returns = compute_returns(equity_curve)
    total_return_pct = (
        (equity_curve[-1] / equity_curve[0] - 1.0) * 100.0
        if len(equity_curve) >= 2 and equity_curve[0] > 0
        else float("nan")
    )
    return {
        "total_return_pct": total_return_pct,
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar": calmar_ratio(equity_curve, periods_per_year),
        "win_rate": win_rate(trade_history),
        "profit_factor": profit_factor(trade_history),
    }
