
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_training_results(portfolio_history: pd.DataFrame, trade_history: pd.DataFrame = None):
    """
    Plot interactive training/backtest results using Plotly.
    
    Args:
        portfolio_history: DataFrame with columns [step, value, balance_usdt, balance_asset, price]
                           (Assuming 'price' was logged or can be joined)
                           Actually Env logs: step, value, action, trade_type, reward.
                           It doesn't log price/balances by default in get_portfolio_history unless added.
                           Env's `portfolio_history` list contains: step, value, action, trade_type, reward.
                           Env's `trade_history` contains: step, price, type, ...
        trade_history: DataFrame with columns [step, type, price, amount_usdt, ...]
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

    # 1. Price Chart (simulated from trades or need external price data?)
    # Since we don't have full price history here (unless passed), 
    # we can only plot prices where trades happened or if portfolio_history includes it.
    # Updated Env's portfolio_history doesn't include price. 
    # But we can infer it or just plot Equity Curve vs Action.
    
    # Let's plot Portfolio Value (NAV)
    fig.add_trace(
        go.Scatter(
            x=portfolio_history['step'], 
            y=portfolio_history['value'], 
            name="Net Asset Value (NAV)",
            line=dict(color='blue', width=2)
        ),
        row=1, col=1, secondary_y=False
    )
    
    # 2. Buy/Sell Markers on NAV (or Price if available)
    if trade_history is not None and not trade_history.empty:
        buys = trade_history[trade_history['type'] == 'buy']
        sells = trade_history[trade_history['type'] == 'sell']
        
        # We need to map trade step to portfolio value at that step
        # Assuming steps match.
        
        # Markers for Buys
        fig.add_trace(
            go.Scatter(
                x=buys['step'],
                y=buys['portfolio_value'], # Plot on NAV curve
                mode='markers',
                name='Buy',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Markers for Sells
        fig.add_trace(
            go.Scatter(
                x=sells['step'],
                y=sells['portfolio_value'],
                mode='markers',
                name='Sell',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ),
            row=1, col=1, secondary_y=False
        )

    # 3. Reward / Action
    # Plot standard deviation of rewards? Or Action magnitude?
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

def render_visualization(env, filename="training_viz.html"):
    """
    Helper to render env history to HTML.
    """
    pf_df = env.get_portfolio_history()
    tr_df = env.get_trade_history()
    
    fig = plot_training_results(pf_df, tr_df)
    if fig:
        fig.write_html(filename)
        print(f"Visualization saved to {filename}")
