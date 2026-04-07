"""
Visualization functions for order book and price data.

Updated to consume strict Pydantic Trade objects and correctly
map LLM-AGENT executions against the TWAP mid-price trajectory.
"""

from typing import List, Tuple, Any
import plotly.graph_objects as go

def plot_order_book_depth(
    bids: List[Tuple[float, int]],
    asks: List[Tuple[float, int]],
    title: str = "Level 2 Order Book Depth"
) -> go.Figure:
    """
    Create order book depth visualization.
    Displays L2 order book with bids (green) on left side and asks (red) on right side.
    """
    fig = go.Figure()
    
    # Extract bid/ask prices and quantities
    bid_prices = [b[0] for b in bids]
    bid_quantities = [b[1] for b in bids]
    
    ask_prices = [a[0] for a in asks]
    ask_quantities = [a[1] for a in asks]
    
    # Add bid bars (green, facing left by using negative quantities for display)
    fig.add_trace(go.Bar(
        y=bid_prices,
        x=[-q for q in bid_quantities],  # Negative for left-facing bars
        orientation='h',
        name='Bids',
        marker=dict(color='rgba(0, 200, 100, 0.7)'),
        hovertemplate='Price: $%{y:.2f}<br>Quantity: %{customdata}<extra></extra>',
        customdata=bid_quantities # Pass absolute values for the hover tooltip
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
        xaxis=dict(tickformat=',d', tickmode='linear'),
        yaxis=dict(tickformat='$,.2f'),
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='top')
    )
    
    # Format x-axis as absolute values so bids don't show negative numbers on the axis
    if fig.data and len(fig.data[0].x) > 0:
        fig.update_xaxes(
            ticktext=[str(abs(int(x))) for x in fig.data[0].x], 
            tickvals=fig.data[0].x
        )
    
    return fig

def plot_execution_trajectory(
    prices: List[float],
    agent_trades: List[Any], # Accepts Pydantic Trade objects
    title: str = "Execution Trajectory vs. Mid-Price Benchmark"
) -> go.Figure:
    """
    Plot actual filled trades against the mid-price benchmark.
    Parses Pydantic Trade objects to map buyer/seller IDs to the LLM's actions.
    """
    fig = go.Figure()
    
    # Plot continuous mid-price benchmark line
    steps = list(range(len(prices)))
    fig.add_trace(go.Scatter(
        x=steps,
        y=prices,
        mode='lines',
        name='Mid-Price (TWAP Benchmark)',
        line=dict(color='rgba(100, 150, 255, 0.8)', width=2),
        hovertemplate='Step: %{x}<br>Mid-Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Separate trades by checking if the LLM was the buyer or seller
    buy_trades = [t for t in agent_trades if t.buyer_id == 'LLM-AGENT']
    sell_trades = [t for t in agent_trades if t.seller_id == 'LLM-AGENT']
    
    # Plot buy executions (green scatter)
    if buy_trades:
        buy_steps = [int(t.timestamp) for t in buy_trades] # We stored step in timestamp
        buy_prices = [t.price for t in buy_trades]
        buy_sizes = [max(t.quantity / 5, 8) for t in buy_trades]  # Dynamic bubble sizing
        
        fig.add_trace(go.Scatter(
            x=buy_steps,
            y=buy_prices,
            mode='markers',
            name='Filled Buys',
            marker=dict(
                color='green',
                size=buy_sizes,
                opacity=0.7,
                line=dict(color='darkgreen', width=2)
            ),
            hovertemplate='Step: %{x}<br>Fill Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Plot sell executions (red scatter)
    if sell_trades:
        sell_steps = [int(t.timestamp) for t in sell_trades]
        sell_prices = [t.price for t in sell_trades]
        sell_sizes = [max(t.quantity / 5, 8) for t in sell_trades]
        
        fig.add_trace(go.Scatter(
            x=sell_steps,
            y=sell_prices,
            mode='markers',
            name='Filled Sells',
            marker=dict(
                color='red',
                size=sell_sizes,
                opacity=0.7,
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='Step: %{x}<br>Fill Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Clean, institutional layout
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Execution Price ($)",
        height=500,
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(tickformat='$,.2f'),
        xaxis=dict(tickformat='d'),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig