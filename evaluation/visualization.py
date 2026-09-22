
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# One (symbol, color) per trade type. Liquidation gets a highly distinct
# marker (a red X) since it's the failure mode a backtest should make
# impossible to miss on the chart, not just another data point.
TRADE_MARKERS = {
    "buy": dict(symbol="triangle-up", size=12, color="green"),
    "sell": dict(symbol="triangle-down", size=12, color="red"),
    "short": dict(symbol="triangle-down", size=12, color="orange"),
    "cover": dict(symbol="triangle-up", size=12, color="cyan"),
    "liquidation": dict(symbol="x", size=16, color="red", line=dict(width=2, color="darkred")),
}

BENCHMARK_COLORS = ["gray", "purple", "yellow"]


def plot_training_results(
    portfolio_history: pd.DataFrame,
    trade_history: pd.DataFrame = None,
    benchmark_curves: dict = None,
):
    """
    Plot interactive training/backtest results using Plotly.

    Args:
        portfolio_history: DataFrame with columns [step, value, action, trade_type, reward]
                           (env.get_portfolio_history()).
        trade_history: DataFrame with columns [step, type, price, amount_usdt, ...]
                       (env.get_trade_history()) — 'type' may be any of
                       buy/sell/short/cover/liquidation.
        benchmark_curves: Optional {label: array-like} of equity curves to
                          overlay on the NAV subplot for comparison (e.g.
                          {"Buy & Hold": ..., "Aléatoire": ...}), same
                          length/step alignment as portfolio_history.
    """
    if portfolio_history.empty:
        print("No portfolio history to plot.")
        return None

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # 1. Portfolio Value (NAV)
    fig.add_trace(
        go.Scatter(
            x=portfolio_history['step'],
            y=portfolio_history['value'],
            name="Net Asset Value (NAV)",
            line=dict(color='blue', width=2)
        ),
        row=1, col=1, secondary_y=False
    )

    # 2. Optional benchmark overlays (Buy & Hold, random baseline, ...)
    if benchmark_curves:
        for i, (label, curve) in enumerate(benchmark_curves.items()):
            color = BENCHMARK_COLORS[i % len(BENCHMARK_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=portfolio_history['step'][: len(curve)],
                    y=curve,
                    name=label,
                    line=dict(color=color, width=1.5, dash='dot'),
                ),
                row=1, col=1, secondary_y=False
            )

    # 3. Trade markers on NAV — one trace per trade type present, so each
    # gets its own legend entry and color (buy/sell/short/cover/liquidation).
    if trade_history is not None and not trade_history.empty:
        for trade_type, marker in TRADE_MARKERS.items():
            subset = trade_history[trade_history['type'] == trade_type]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset['step'],
                    y=subset['portfolio_value'],
                    mode='markers',
                    name=trade_type.capitalize(),
                    marker=marker,
                ),
                row=1, col=1, secondary_y=False
            )

    # 4. Reward / Action
    fig.add_trace(
        go.Bar(
            x=portfolio_history['step'],
            y=portfolio_history['action'],
            name="Action (Position)",
            marker=dict(color='gray', opacity=0.3)
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="Training Session Analysis",
        xaxis_title="Step",
        yaxis_title="Portfolio Value ($)",
        height=800,
        showlegend=True,
        template="plotly_dark"
    )

    return fig

def render_visualization(env, filename="training_viz.html", benchmark_curves: dict = None):
    """
    Helper to render env history to HTML.
    """
    pf_df = env.get_portfolio_history()
    tr_df = env.get_trade_history()

    fig = plot_training_results(pf_df, tr_df, benchmark_curves=benchmark_curves)
    if fig:
        fig.write_html(filename)
        print(f"Visualization saved to {filename}")
