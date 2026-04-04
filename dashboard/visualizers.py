"""
Visualization functions for order book and price data.

This module provides Plotly-based plotting utilities for visualizing
market microstructure, execution trajectories, and trading performance.
"""

from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_order_book_depth(
    bids: List[Tuple[float, int]],
    asks: List[Tuple[float, int]],
    title: str = "Level 2 Order Book Depth"
) -> go.Figure:
    """
    Create order book depth visualization.
    
    Displays L2 order book with bids (green) on left side and asks (red) on right side.
    
    Args:
        bids: List of (price, quantity) tuples, sorted descending by price
        asks: List of (price, quantity) tuples, sorted ascending by price
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Extract bid prices and quantities
    bid_prices = [b[0] for b in bids]
    bid_quantities = [b[1] for b in bids]
    
    # Extract ask prices and quantities
    ask_prices = [a[0] for a in asks]
    ask_quantities = [a[1] for a in asks]
    
    # Add bid bars (green, facing left by using negative quantities for display)
    fig.add_trace(go.Bar(
        y=bid_prices,
        x=[-q for q in bid_quantities],  # Negative for left-facing bars
        orientation='h',
        name='Bids',
        marker=dict(color='rgba(0, 200, 100, 0.7)'),
        hovertemplate='Price: $%{y:.2f}<br>Quantity: %{x:.0f}<extra></extra>'
    ))
    
    # Add ask bars (red, facing right)
    fig.add_trace(go.Bar(
        y=ask_prices,
        x=ask_quantities,
        orientation='h',
        name='Asks',
        marker=dict(color='rgba(255, 50, 50, 0.7)'),
        hovertemplate='Price: $%{y:.2f}<br>Quantity: %{x:.0f}<extra></extra>'
    ))
    
    # Update layout for mirror image effect
    fig.update_layout(
        title=title,
        xaxis_title="Quantity (Bids ← → Asks)",
        yaxis_title="Price ($)",
        height=500,
        hovermode='closest',
        barmode='overlay',
        xaxis=dict(
            tickformat=',d',
            tickmode='linear',
        ),
        yaxis=dict(
            tickformat='$,.2f',
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=1.0,
            xanchor='center',
            yanchor='top'
        )
    )
    
    # Format x-axis as absolute values
    fig.update_xaxes(ticktext=[str(abs(int(x))) for x in fig.data[0].x], 
                     tickvals=fig.data[0].x)
    
    return fig


def plot_execution_trajectory(
    prices: List[float],
    agent_trades: List[Dict],
    title: str = "Execution Trajectory vs. Mid-Price Benchmark"
) -> go.Figure:
    """
    Plot agent execution trajectory against mid-price benchmark.
    
    Shows mid-price as continuous line with scatter points marking agent trades.
    
    Args:
        prices: List of mid-prices over time
        agent_trades: List of dicts with keys: {step, side, price, quantity}
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot mid-price line
    steps = list(range(len(prices)))
    fig.add_trace(go.Scatter(
        x=steps,
        y=prices,
        mode='lines',
        name='Mid-Price Benchmark',
        line=dict(color='rgba(100, 150, 255, 0.8)', width=2),
        hovertemplate='Step: %{x}<br>Mid-Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Separate trades by side
    buy_trades = [t for t in agent_trades if t.get('side') == 'BUY']
    sell_trades = [t for t in agent_trades if t.get('side') == 'SELL']
    
    # Plot buy trades (green scatter)
    if buy_trades:
        buy_steps = [t.get('step', 0) for t in buy_trades]
        buy_prices = [t.get('price', 0) for t in buy_trades]
        buy_sizes = [max(t.get('quantity', 1) / 10, 8) for t in buy_trades]  # Scale for visibility
        
        fig.add_trace(go.Scatter(
            x=buy_steps,
            y=buy_prices,
            mode='markers',
            name='Buy Executions',
            marker=dict(
                color='green',
                size=buy_sizes,
                opacity=0.7,
                line=dict(color='darkgreen', width=2)
            ),
            hovertemplate='Step: %{x}<br>Execution Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Plot sell trades (red scatter)
    if sell_trades:
        sell_steps = [t.get('step', 0) for t in sell_trades]
        sell_prices = [t.get('price', 0) for t in sell_trades]
        sell_sizes = [max(t.get('quantity', 1) / 10, 8) for t in sell_trades]
        
        fig.add_trace(go.Scatter(
            x=sell_steps,
            y=sell_prices,
            mode='markers',
            name='Sell Executions',
            marker=dict(
                color='red',
                size=sell_sizes,
                opacity=0.7,
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='Step: %{x}<br>Execution Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(
            tickformat='$,.2f',
        ),
        xaxis=dict(
            tickformat='d',
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig
    
    print(f"[VISUALIZATION] Agent inventory: {title}")
    print(f"  Snapshots: {len(timestamps)}")
    print(f"  Max inventory: {max_inventory:.0f} shares")


def plot_market_impact(
    trade_times: List[datetime],
    price_before: List[float],
    price_after: List[float],
    trade_quantities: List[float],
    title: str = "Market Impact"
) -> None:
    """
    Plot market impact caused by agent trades.
    
    Args:
        trade_times: Timestamps of agent trades
        price_before: Price before each trade
        price_after: Price after each trade
        trade_quantities: Quantities traded
        title: Chart title
    """
    # TODO: Implement market impact visualization
    # Show price movement attributable to agent's trades
    # Quantify market impact as function of trade size
    
    print(f"[VISUALIZATION] Market impact: {title}")
    print(f"  Trades analyzed: {len(trade_times)}")
