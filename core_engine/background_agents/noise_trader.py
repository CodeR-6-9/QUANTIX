"""
Noise Trader agent for random market activity.

Optimized for high-frequency simulation:
- Replaces slow UUIDs with O(1) counters.
- Enforces strict price rounding.
- Sweeps top-of-book without destroying deep market liquidity.
"""

from typing import Dict, Any, Literal
import random

from ..schema import Order


class NoiseTrader:
    """
    Noise Trader background agent.
    Generates random retail trading volume to push the mid-price around.
    """
    
    def __init__(
        self,
        agent_id: str = "NT-SIM",
        trade_probability: float = 0.2,
        base_order_size: int = 50
    ) -> None:
        self.agent_id = agent_id
        self.trade_probability = trade_probability
        self.base_order_size = base_order_size
        
        # Statistics & ID tracking
        self.trades_submitted = 0
        self.total_volume = 0
        self.buy_count = 0
        self.sell_count = 0

    def step(self, current_mid_price: float, current_time: float) -> Dict[str, Any]:
        """
        Execute one step of the noise trading strategy.
        Randomly submit aggressive market orders to create volume.
        """
        # 1. Random activation check
        if random.random() > self.trade_probability:
            return {"cancels": [], "new_orders": []}
            
        # 2. Trading Decision
        side: Literal["BUY", "SELL"] = random.choice(["BUY", "SELL"])
        
        # Randomize quantity (e.g., 50 base -> 40 to 70 shares)
        quantity = max(1, self.base_order_size + random.randint(-10, 20))
        
        # 3. Aggressive Pricing (Rounded to prevent float fragmentation)
        # We cross the spread by 0.50 to guarantee a fill against the Market Maker,
        # but not so far that we wipe out 10 levels of deep liquidity.
        if side == "BUY":
            price = round(current_mid_price + 0.50, 2)
            self.buy_count += 1
        else:
            price = round(current_mid_price - 0.50, 2)
            self.sell_count += 1
            
        # 4. Fast O(1) Order Generation
        self.trades_submitted += 1
        self.total_volume += quantity
        
        order_id = f"{self.agent_id}_{self.trades_submitted}"
        
        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=current_time,
            agent_id=self.agent_id
        )
        
        return {
            "cancels": [],
            "new_orders": [order]
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "trades_submitted": self.trades_submitted,
            "total_volume": self.total_volume,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "trade_probability": self.trade_probability
        }