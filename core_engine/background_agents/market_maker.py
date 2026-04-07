"""
Market Maker agent for providing liquidity.

Optimized for high-frequency simulation:
- Uses O(1) integer-based Order IDs instead of slow UUID generation.
- Enforces strict 2-decimal price rounding to prevent floating-point fragmentation.
"""

from typing import List, Dict, Any
from ..schema import Order


class MarketMaker:
    """
    Market Maker background agent.
    Continuously posts buy and sell orders around the mid-price to provide liquidity.
    """
    
    def __init__(
        self,
        agent_id: str = "MM-SIM",
        num_levels: int = 3,
        spread_width: float = 0.05,
        order_size: int = 100
    ) -> None:
        self.agent_id = agent_id
        self.num_levels = num_levels
        self.spread_width = spread_width
        self.order_size = order_size
        
        self.active_order_ids: List[str] = []
        
        self.quotes_posted = 0
        self.quotes_canceled = 0

    def step(self, current_mid_price: float, current_time: float) -> Dict[str, Any]:
        """
        Execute one step of the market making algorithm.
        1. Cancel old quotes.
        2. Post new quotes at strict tick intervals.
        """
        # ===== CANCELLATION PHASE =====
        cancels = self.active_order_ids.copy()
        self.quotes_canceled += len(cancels)
        self.active_order_ids.clear()
        
        # ===== QUOTE GENERATION PHASE =====
        new_orders: List[Order] = []
        
        for level in range(1, self.num_levels + 1):
            # --- BIDS (Buy Side) ---
            # STRICT ROUNDING to prevent floating-point micro-levels in the order book
            bid_price = round(current_mid_price - (level * self.spread_width), 2)
            bid_id = f"{self.agent_id}_B_{self.quotes_posted}"
            
            new_orders.append(Order(
                order_id=bid_id,
                side="BUY",
                price=bid_price,
                quantity=self.order_size,
                timestamp=current_time,
                agent_id=self.agent_id
            ))
            self.active_order_ids.append(bid_id)
            self.quotes_posted += 1
            
            # --- ASKS (Sell Side) ---
            ask_price = round(current_mid_price + (level * self.spread_width), 2)
            ask_id = f"{self.agent_id}_A_{self.quotes_posted}"
            
            new_orders.append(Order(
                order_id=ask_id,
                side="SELL",
                price=ask_price,
                quantity=self.order_size,
                timestamp=current_time,
                agent_id=self.agent_id
            ))
            self.active_order_ids.append(ask_id)
            self.quotes_posted += 1
            
        return {
            "cancels": cancels,
            "new_orders": new_orders
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "quotes_posted": self.quotes_posted,
            "quotes_canceled": self.quotes_canceled,
            "active_orders": len(self.active_order_ids)
        }