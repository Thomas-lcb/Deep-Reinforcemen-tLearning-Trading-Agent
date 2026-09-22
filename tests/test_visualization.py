"""
tests/test_visualization.py — Smoke tests for evaluation/visualization.py.

Plotly figures aren't meaningfully assertable pixel-by-pixel; these tests
check that the right number/kind of traces get built for the trade types
present (in particular short/cover/liquidation, added alongside the
short-selling feature — a previous gap flagged in the final branch review),
and that nothing raises.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from evaluation.visualization import plot_training_results, TRADE_MARKERS


@pytest.fixture
def portfolio_history():
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "value": [10000, 10100, 9900, 9500, 9800],
        "action": [0.5, -0.3, -1.0, 0.2, 0.0],
        "trade_type": ["buy", "sell", "liquidation", "hold", "hold"],
        "reward": [0.01, -0.01, -0.05, 0.0, 0.0],
    })


@pytest.fixture
def trade_history():
    return pd.DataFrame({
        "step": [1, 2, 3, 4],
        "type": ["buy", "sell", "short", "liquidation"],
        "portfolio_value": [10000, 10100, 9900, 9500],
        "pnl_pct": [float("nan"), 0.01, float("nan"), -0.3],
    })


class TestPlotTrainingResults:
    def test_empty_portfolio_history_returns_none(self):
        assert plot_training_results(pd.DataFrame()) is None

    def test_returns_figure_without_trade_history(self, portfolio_history):
        fig = plot_training_results(portfolio_history)
        assert isinstance(fig, go.Figure)

    def test_one_trace_per_trade_type_present(self, portfolio_history, trade_history):
        fig = plot_training_results(portfolio_history, trade_history)
        trace_names = {t.name for t in fig.data}
        # buy, sell, short, liquidation are present in the fixture; cover is not.
        for expected in ["Buy", "Sell", "Short", "Liquidation"]:
            assert expected in trace_names
        assert "Cover" not in trace_names

    def test_all_trade_marker_types_have_distinct_symbols_or_colors(self):
        # Every configured marker must be visually distinguishable from the others.
        seen = set()
        for trade_type, marker in TRADE_MARKERS.items():
            key = (marker["symbol"], marker["color"])
            assert key not in seen, f"{trade_type} marker collides with another type"
            seen.add(key)

    def test_benchmark_overlay_adds_named_traces(self, portfolio_history):
        benchmarks = {
            "Buy & Hold": [10000, 10050, 10020, 9990, 10010],
            "Aléatoire": [10000, 9980, 9950, 9900, 9920],
        }
        fig = plot_training_results(portfolio_history, benchmark_curves=benchmarks)
        trace_names = {t.name for t in fig.data}
        assert "Buy & Hold" in trace_names
        assert "Aléatoire" in trace_names
