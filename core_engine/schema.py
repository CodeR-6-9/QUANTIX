"""
Data schemas for the LOB environment using Pydantic.

This module defines the core data structures used throughout the simulation,
including agent state, actions, and rewards with type validation.
"""

from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field


class Order(BaseModel):
    """
    Represents a single order in the limit order book.
    
    Implements price-time priority semantics for matching:
    - Higher bids and lower asks are matched first
    - Among orders at the same price, earlier timestamps are matched first
    """
    
    order_id: str = Field(..., description="Unique order identifier")
    side: Literal["BUY", "SELL"] = Field(..., description="Order side: BUY or SELL")
    price: float = Field(..., description="Order limit price", gt=0)
    quantity: int = Field(..., description="Order quantity in shares", gt=0)
    timestamp: float = Field(..., description="Order submission timestamp (seconds since epoch)")
    agent_id: str = Field(..., description="ID of agent that placed this order")
    
    def __lt__(self, other: 'Order') -> bool:
        """
        Implement less-than comparison for priority queue operations.
        
        Price-Time Priority:
        - Primary: Price comparison (will be negated for bids in heap)
        - Secondary: Timestamp comparison (older = earlier = wins)
        
        Args:
            other: Order to compare against
            
        Returns:
            True if this order has lower priority than other order
        """
        # If prices differ, lower price has lower priority (except we negate for bids)
        if self.price != other.price:
            return self.price < other.price
        # If prices are equal, older timestamp (smaller value) has higher priority
        return self.timestamp < other.timestamp


class Trade(BaseModel):
    """
    Represents an executed trade between buyer and seller.
    
    Trades are immutable records of matched orders.
    """
    
    trade_id: str = Field(..., description="Unique trade identifier")
    buyer_id: str = Field(..., description="ID of buyer agent")
    seller_id: str = Field(..., description="ID of seller agent")
    price: float = Field(..., description="Execution price", gt=0)
    quantity: int = Field(..., description="Executed quantity in shares", gt=0)
    timestamp: float = Field(..., description="Execution timestamp (seconds since epoch)")


class OrderBookSnapshot(BaseModel):
    """Snapshot of the current order book state."""
    
    bids: List[Order] = Field(default_factory=list, description="Buy-side orders")
    asks: List[Order] = Field(default_factory=list, description="Sell-side orders")
    timestamp: datetime = Field(default_factory=datetime.now)
    mid_price: float = Field(..., description="Mid-point price")
    spread: float = Field(..., description="Bid-ask spread")


class MarketState(BaseModel):
    """Current state of the market."""
    
    symbol: str = Field(..., description="Trading symbol (e.g., 'AAPL')")
    current_price: float = Field(..., description="Current mid price")
    bid_price: float = Field(..., description="Best bid price")
    ask_price: float = Field(..., description="Best ask price")
    bid_quantity: float = Field(..., description="Quantity at best bid")
    ask_quantity: float = Field(..., description="Quantity at best ask")
    volume_24h: float = Field(default=0.0)
    volatility: float = Field(..., description="Current market volatility")


class AgentState(BaseModel):
    """
    Observation state provided to LLM agent at each environment step.
    
    Represents the order book state the agent needs to make trading decisions.
    """
    
    time_remaining: int = Field(..., description="Steps remaining until episode termination")
    inventory_remaining: int = Field(..., description="Shares remaining to execute")
    mid_price: float = Field(..., description="Current mid-price: (best_bid + best_ask) / 2")
    bids: List[Tuple[float, int]] = Field(
        ..., 
        description="Top 3 bid levels: [(price, quantity), ...]"
    )
    asks: List[Tuple[float, int]] = Field(
        ..., 
        description="Top 3 ask levels: [(price, quantity), ...]"
    )


class AgentAction(BaseModel):
    """
    Action taken by LLM agent for execution.
    
    Specifies how many shares to execute and in what style (aggressive/passive).
    """
    
    side: str = Field(..., description="'BUY' or 'SELL'")
    shares_to_execute: int = Field(..., description="Number of shares to execute (>=0)")
    execution_style: str = Field(
        ..., 
        description="'AGGRESSIVE' (crosses spread) or 'PASSIVE' (at best bid/ask)"
    )


class StepReward(BaseModel):
    """Reward structure for a single step in the LOB environment."""
    
    total_reward: float = Field(..., description="Slippage penalty (negative) or 0")
    done: bool = Field(default=False, description="Episode termination flag")
