"""
Adversarial High-Frequency Trading Bot.
Implements predatory market microstructure tactics (Spoofing & Penny Jumping)
to actively hunt and exploit the LLM agent's execution footprint.
"""

from typing import Dict, List, Any
from ..schema import Order

class ToxicTrader:
    def __init__(
        self, 
        agent_id: str = "TOXIC-HFT", 
        penny_jump_threshold: int = 300, 
        tick_size: float = 0.01
    ):
        self.agent_id = agent_id
        self.penny_jump_threshold = penny_jump_threshold
        self.tick_size = tick_size
        self.order_counter = 0
        self.step_counter = 0
        
        # State tracking to manage lifecycle of malicious orders
        self.active_spoofs: List[str] = []
        self.active_penny_jumps: List[str] = []

    def _generate_order_id(self) -> str:
        """Fast integer-based ID generation (bypasses slow UUIDs)."""
        self.order_counter += 1
        return f"{self.agent_id}_{self.order_counter}"

    def step(
        self, 
        current_micro_price: float, 
        current_time: float, 
        lob_state: Dict[str, List[tuple]]
    ) -> Dict[str, List[Any]]:
        """
        Evaluates the current limit order book and executes predatory tactics.
        Returns newly created malicious orders and IDs of fake orders to cancel.
        """
        self.step_counter += 1
        new_orders = []
        cancels = []

        # ==========================================
        # LIFECYCLE MANAGEMENT: Pull fake liquidity
        # ==========================================
        # Spoofing requires cancelling the massive fake orders on the very next step
        # before they accidentally get executed.
        if self.active_spoofs:
            cancels.extend(self.active_spoofs)
            self.active_spoofs.clear()

        # We also cancel old penny jumps to simulate HFT "flickering" quotes
        if self.active_penny_jumps:
            cancels.extend(self.active_penny_jumps)
            self.active_penny_jumps.clear()

        bids = lob_state.get("bids", [])
        asks = lob_state.get("asks", [])

        # ==========================================
        # TACTIC 1: PENNY JUMPING (Front-Running)
        # ==========================================
        # Scan for massive bids from the LLM. If we find one, place an order 
        # exactly $0.01 higher to steal queue priority and intercept the fill.
        for price, qty in bids:
            if qty >= self.penny_jump_threshold:
                jump_price = round(price + self.tick_size, 2)
                order_id = self._generate_order_id()
                new_orders.append(
                    Order.model_construct(
                        order_id=order_id,
                        side="BUY",
                        price=jump_price,
                        quantity=100,  # Standard lot size to steal priority
                        timestamp=current_time,
                        agent_id=self.agent_id
                    )
                )
                self.active_penny_jumps.append(order_id)
                break  # Just jump the absolute top massive order

        # Scan for massive asks to jump
        for price, qty in asks:
            if qty >= self.penny_jump_threshold:
                jump_price = round(price - self.tick_size, 2)
                order_id = self._generate_order_id()
                new_orders.append(
                    Order.model_construct(
                        order_id=order_id,
                        side="SELL",
                        price=jump_price,
                        quantity=100,
                        timestamp=current_time,
                        agent_id=self.agent_id
                    )
                )
                self.active_penny_jumps.append(order_id)
                break

        # ==========================================
        # TACTIC 2: DETERMINISTIC SPOOFING
        # ==========================================
        # Occasionally flash a massive 5,000-share order deep in the book.
        # This dramatically manipulates the Micro-Price equation to bait the LLM.
        
        if self.step_counter % 4 == 0:
            # Spoof the Bid: Creates a fake "Buy Wall" (Bullish Illusion)
            spoof_price = round(current_micro_price - 0.50, 2)
            spoof_id = self._generate_order_id()
            new_orders.append(
                Order.model_construct(
                    order_id=spoof_id,
                    side="BUY",
                    price=spoof_price,
                    quantity=5000, # Massive fake liquidity
                    timestamp=current_time,
                    agent_id=self.agent_id
                )
            )
            self.active_spoofs.append(spoof_id)
            
        elif self.step_counter % 4 == 2:
            # Spoof the Ask: Creates a fake "Sell Wall" (Bearish Illusion)
            spoof_price = round(current_micro_price + 0.50, 2)
            spoof_id = self._generate_order_id()
            new_orders.append(
                Order.model_construct(
                    order_id=spoof_id,
                    side="SELL",
                    price=spoof_price,
                    quantity=5000,
                    timestamp=current_time,
                    agent_id=self.agent_id
                )
            )
            self.active_spoofs.append(spoof_id)

        return {
            "new_orders": new_orders,
            "cancels": cancels
        }