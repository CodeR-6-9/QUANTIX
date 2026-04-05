"""
Market Maker agent for providing liquidity.

Market makers maintain tight bid-ask spreads and provide consistent
liquidity to the order book. They quote on both sides of the market
and adjust their prices as the mid-price moves.

Market Making Strategy:
- Post num_levels orders on each side (buy and sell)
- Prices are linearly spaced around the mid-price
- Every step, cancel all old quotes and post new ones
- Classic "quote-and-update" liquidity provision strategy
"""

from typing import List, Dict, Any, Literal
import uuid
import time

from ..schema import Order


class MarketMaker:
    """
    Market Maker background agent.
    
    Continuously posts buy and sell orders around the mid-price
    to provide liquidity, earning the spread. Uses a simple but
    realistic multi-level quoting strategy.
    
    Strategy Details:
    - Posts num_levels orders at different price levels
    - Bid prices: mid - spread_width, mid - 2*spread_width, mid - 3*spread_width, ...
    - Ask prices: mid + spread_width, mid + 2*spread_width, mid + 3*spread_width, ...
    - Each level has the same order_size quantity
    - Every step: cancel all active quotes, post new quotes at updated prices
    """
    
    def __init__(
        self,
        agent_id: str = "MM-001",
        num_levels: int = 3,
        spread_width: float = 0.5,
        order_size: int = 100
    ) -> None:
        """
        Initialize Market Maker agent.
        
        Args:
            agent_id: Unique identifier for this market maker agent
            num_levels: Number of price levels to quote on each side (buy/sell)
            spread_width: Price spacing between consecutive levels (e.g., 0.5 = $0.50)
            order_size: Number of shares to quote at each price level
        """
        self.agent_id: str = agent_id
        self.num_levels: int = num_levels
        self.spread_width: float = spread_width
        self.order_size: int = order_size
        
        # Track active order IDs so we can cancel them when price moves
        # List of order_id strings that are currently resting on the book
        self.active_order_ids: List[str] = []
        
        # Statistics tracking
        self.quotes_posted: int = 0
        self.quotes_canceled: int = 0
    
    def step(self, current_mid_price: float, current_time: float) -> Dict[str, Any]:
        """
        Execute one step of the market making algorithm.
        
        Strategy:
        1. Cancel all previous quotes (wipe the old price levels)
        2. Generate new quotes at the updated mid-price
           - num_levels BID orders: prices below mid
           - num_levels ASK orders: prices above mid
        3. Return both lists for the LOB to process
        
        Timing:
        - In high-frequency contexts, this cancels and re-quotes every millisecond
        - In this simulator, step() is called once per discrete time step
        
        Args:
            current_mid_price: Current market mid-price (e.g., 150.0)
            current_time: Current timestamp (seconds since epoch)
            
        Returns:
            Dictionary with two keys:
            - "cancels": List of order_id strings to cancel
            - "new_orders": List of Order objects to submit
        """
        # ===== CANCELLATION PHASE =====
        # Cancel all active quotes from previous step
        cancels = self.active_order_ids.copy()
        self.quotes_canceled += len(cancels)
        
        # Clear the list for new quotes
        self.active_order_ids = []
        
        # ===== QUOTE GENERATION PHASE =====
        new_orders: List[Order] = []
        
        # Generate BID orders (buy side): posted below mid-price
        for level in range(1, self.num_levels + 1):
            # Bid price decreases as we go deeper: mid - width, mid - 2*width, ...
            bid_price = current_mid_price - (level * self.spread_width)
            
            # Create a unique order ID for this bid
            order_id = f"{self.agent_id}_bid_{uuid.uuid4().hex[:8]}"
            
            # Create the Order object
            bid_order = Order(
                order_id=order_id,
                side="BUY",
                price=bid_price,
                quantity=self.order_size,
                timestamp=current_time,
                agent_id=self.agent_id
            )
            
            new_orders.append(bid_order)
            self.active_order_ids.append(order_id)
        
        # Generate ASK orders (sell side): posted above mid-price
        for level in range(1, self.num_levels + 1):
            # Ask price increases as we go deeper: mid + width, mid + 2*width, ...
            ask_price = current_mid_price + (level * self.spread_width)
            
            # Create a unique order ID for this ask
            order_id = f"{self.agent_id}_ask_{uuid.uuid4().hex[:8]}"
            
            # Create the Order object
            ask_order = Order(
                order_id=order_id,
                side="SELL",
                price=ask_price,
                quantity=self.order_size,
                timestamp=current_time,
                agent_id=self.agent_id
            )
            
            new_orders.append(ask_order)
            self.active_order_ids.append(order_id)
        
        self.quotes_posted += len(new_orders)
        
        return {
            "cancels": cancels,
            "new_orders": new_orders
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about this market maker's activity.
        
        Returns:
            Dictionary with activity metrics
        """
        return {
            "agent_id": self.agent_id,
            "quotes_posted": self.quotes_posted,
            "quotes_canceled": self.quotes_canceled,
            "active_orders": len(self.active_order_ids),
            "num_levels": self.num_levels,
            "spread_width": self.spread_width,
            "order_size": self.order_size
        }
