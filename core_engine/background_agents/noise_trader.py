"""
Noise Trader agent for random market activity.

Noise traders generate random trading activity without directional bias,
simulating retail traders or uninformed market participant behavior.
They submit market orders (aggressive prices that execute immediately)
to create trading volume and push prices around non-systematically.

Market Dynamics Created:
- Random trading volume (not predictable)
- Pushes mid-price around without information
- Creates realistic market microstructure
- Simulates "noise" in financial markets (retail flow, etc)
"""

from typing import Dict, Any, List
import random
import uuid
import time

from ..schema import Order


class NoiseTrader:
    """
    Noise Trader background agent.
    
    Generates random trading orders with configurable intensity.
    Uses aggressive market-order prices to ensure execution.
    
    Trading Strategy:
    - With probability trade_probability, submit a market order
    - Otherwise, stay inactive
    - Market orders are priced aggressively:
      - BUY orders: price = mid + $5 (guaranteed fill, paying the ask)
      - SELL orders: price = mid - $5 (guaranteed fill, taking the bid)
    - Quantity is randomized slightly around base_order_size
    """
    
    def __init__(
        self,
        agent_id: str = "NT-001",
        trade_probability: float = 0.2,
        base_order_size: int = 50
    ) -> None:
        """
        Initialize Noise Trader agent.
        
        Args:
            agent_id: Unique identifier for this noise trader
            trade_probability: Probability (0-1) of trading in each step.
                              e.g., 0.2 = 20% chance to trade each step
            base_order_size: Base quantity for orders (will be randomized +/- 10-20)
        """
        self.agent_id: str = agent_id
        self.trade_probability: float = trade_probability
        self.base_order_size: int = base_order_size
        
        # Statistics tracking
        self.trades_submitted: int = 0
        self.total_volume: int = 0
        self.buy_count: int = 0
        self.sell_count: int = 0
    
    def step(self, current_mid_price: float, current_time: float) -> Dict[str, Any]:
        """
        Execute one step of the noise trading strategy.
        
        Decision Process:
        1. Generate random number in [0, 1)
        2. If random number > trade_probability: stay idle this step
        3. Otherwise:
           - Pick side randomly (50/50 BUY vs SELL)
           - Randomize order size around base_order_size
           - Price aggressively to guarantee execution (market order)
           - Create and submit Order object
        
        Aggressive Pricing:
        - BUY @ mid + $5: takes the ask side, guaranteed to fill
        - SELL @ mid - $5: takes the bid side, guaranteed to fill
        - This ensures the noise trader's orders execute immediately
        
        Args:
            current_mid_price: Current market mid-price (e.g., 150.0)
            current_time: Current timestamp (seconds since epoch)
            
        Returns:
            Dictionary with two keys:
            - "cancels": Empty list (noise traders don't cancel)
            - "new_orders": List with 0 or 1 Order object
        """
        # Decision: Should we trade this step?
        if random.random() > self.trade_probability:
            # No trade this step
            return {
                "cancels": [],
                "new_orders": []
            }
        
        # ===== TRADING DECISION =====
        # Pick a random side: BUY or SELL (50/50)
        side = random.choice(["BUY", "SELL"])
        
        # Randomize quantity around base_order_size
        # Example: if base_order_size=50, size could be 40-80
        quantity = self.base_order_size + random.randint(-10, 20)
        
        # Ensure quantity is positive (edge case if base_order_size is very small)
        quantity = max(1, quantity)
        
        # ===== AGGRESSIVE PRICING =====
        # Price aggressively to guarantee execution (market order semantics)
        if side == "BUY":
            # BUY at mid + $5: takes the ask, will execute immediately
            price = current_mid_price + 5.0
            self.buy_count += 1
        else:  # SELL
            # SELL at mid - $5: takes the bid, will execute immediately
            price = current_mid_price - 5.0
            self.sell_count += 1
        
        # ===== ORDER CREATION =====
        # Generate unique order ID
        order_id = f"{self.agent_id}_{uuid.uuid4().hex[:8]}"
        
        # Create Order object
        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=current_time,
            agent_id=self.agent_id
        )
        
        # Update statistics
        self.trades_submitted += 1
        self.total_volume += quantity
        
        return {
            "cancels": [],
            "new_orders": [order]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about this noise trader's activity.
        
        Returns:
            Dictionary with activity metrics
        """
        return {
            "agent_id": self.agent_id,
            "trades_submitted": self.trades_submitted,
            "total_volume": self.total_volume,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "trade_probability": self.trade_probability,
            "base_order_size": self.base_order_size
        }
