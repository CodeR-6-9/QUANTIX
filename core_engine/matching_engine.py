"""
Institutional Limit Order Book matching engine.

Performance characteristics:
- add_order(): O(log N) for heap operations, O(Q) for matching
- cancel_order(): O(1) via lazy deletion
- get_best_bid_ask(): Amortized O(1) via top-level heap cleanup
"""

import time
import heapq
from typing import List, Tuple, Dict, Optional

from .schema import Order, Trade


class LimitOrderBook:
    """
    High-performance Limit Order Book (LOB) utilizing Python's heapq.
    Leverages the self-aware Price-Time Priority implemented in the Order schema.
    """
    
    def __init__(self, symbol: str = "LOB-SIM") -> None:
        self.symbol: str = symbol
        
        # Heaps now store actual Order objects directly.
        # Sorting logic (Max-heap for bids, Min-heap for asks) is handled natively by Order.__lt__
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        
        # Fast O(1) lookup dictionary for cancellations
        self.active_orders: Dict[str, Order] = {}
        
        self.trade_history: List[Trade] = []
        self.trade_counter: int = 0
        
    def add_order(self, order: Order) -> List[Trade]:
        """
        Match incoming orders against resting liquidity or post to book.
        """
        if order.quantity <= 0:
            return []
            
        self.active_orders[order.order_id] = order
        trades: List[Trade] = []
        remaining_quantity = order.quantity
        
        # ===== MATCHING PHASE =====
        if order.side == "BUY":
            while remaining_quantity > 0 and self.asks:
                resting_order = self.asks[0] # Peek at best ask
                
                # Lazy Deletion: Clean up canceled orders that bubbled to the top
                if resting_order.quantity == 0:
                    heapq.heappop(self.asks)
                    continue
                    
                # No match possible (incoming bid is lower than best ask)
                if order.price < resting_order.price:
                    break
                    
                # Execute Trade
                match_qty = min(remaining_quantity, resting_order.quantity)
                
                trade = Trade(
                    trade_id=f"T_{self.trade_counter}",
                    buyer_id=order.agent_id,
                    seller_id=resting_order.agent_id,
                    price=resting_order.price, # Execution always happens at resting price
                    quantity=match_qty,
                    timestamp=time.time()
                )
                trades.append(trade)
                self.trade_history.append(trade)
                self.trade_counter += 1
                
                remaining_quantity -= match_qty
                resting_order.quantity -= match_qty
                
                # Clean up resting order if fully filled
                if resting_order.quantity == 0:
                    heapq.heappop(self.asks)
                    if resting_order.order_id in self.active_orders:
                        del self.active_orders[resting_order.order_id]
                        
        else: # SELL
            while remaining_quantity > 0 and self.bids:
                resting_order = self.bids[0] # Peek at best bid
                
                if resting_order.quantity == 0:
                    heapq.heappop(self.bids)
                    continue
                    
                # No match possible (incoming ask is higher than best bid)
                if order.price > resting_order.price:
                    break
                    
                # Execute Trade
                match_qty = min(remaining_quantity, resting_order.quantity)
                
                trade = Trade(
                    trade_id=f"T_{self.trade_counter}",
                    buyer_id=resting_order.agent_id,
                    seller_id=order.agent_id,
                    price=resting_order.price,
                    quantity=match_qty,
                    timestamp=time.time()
                )
                trades.append(trade)
                self.trade_history.append(trade)
                self.trade_counter += 1
                
                remaining_quantity -= match_qty
                resting_order.quantity -= match_qty
                
                if resting_order.quantity == 0:
                    heapq.heappop(self.bids)
                    if resting_order.order_id in self.active_orders:
                        del self.active_orders[resting_order.order_id]
                        
        # ===== POSTING PHASE =====
        if remaining_quantity > 0:
            order.quantity = remaining_quantity
            if order.side == "BUY":
                heapq.heappush(self.bids, order)
            else:
                heapq.heappush(self.asks, order)
        else:
            # Order fully filled immediately, don't keep it in active dict
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
                
        return trades

    def cancel_order(self, order_id: str) -> bool:
        """
        O(1) Lazy Deletion. Sets quantity to 0. The matching loop or 
        get_best_bid_ask() will safely pop it when it reaches the top of the heap.
        """
        if order_id not in self.active_orders:
            return False
            
        order = self.active_orders[order_id]
        order.quantity = 0 
        del self.active_orders[order_id]
        return True

    def get_best_bid_ask(self) -> Tuple[Optional[float], int, Optional[float], int]:
        """
        Amortized O(1) peek at the top of the book.
        Maintains heap integrity by popping 'dead' zero-quantity orders from the top.
        """
        # Clean dead bids at the top
        while self.bids and self.bids[0].quantity == 0:
            heapq.heappop(self.bids)
            
        # Clean dead asks at the top
        while self.asks and self.asks[0].quantity == 0:
            heapq.heappop(self.asks)

        bid_price = self.bids[0].price if self.bids else None
        bid_qty = self.bids[0].quantity if self.bids else 0
        
        ask_price = self.asks[0].price if self.asks else None
        ask_qty = self.asks[0].quantity if self.asks else 0
        
        return bid_price, bid_qty, ask_price, ask_qty

    def get_l2_state(self) -> Dict[str, List[Tuple[float, int]]]:
        """
        Aggregate top 3 price levels for the Agent's observation.
        """
        # Ensure the tops of the heaps are clean before copying
        self.get_best_bid_ask()
        
        def _aggregate_levels(heap: List[Order], limit: int = 3) -> List[Tuple[float, int]]:
            levels = []
            price_qty_map = {}
            heap_copy = heap.copy()
            
            while heap_copy and len(levels) < limit:
                order = heapq.heappop(heap_copy)
                if order.quantity == 0:
                    continue
                    
                if order.price in price_qty_map:
                    price_qty_map[order.price] += order.quantity
                else:
                    price_qty_map[order.price] = order.quantity
                    if len(price_qty_map) <= limit:
                        levels.append(order.price)
                        
            return [(p, price_qty_map[p]) for p in levels]

        return {
            "bids": _aggregate_levels(self.bids),
            "asks": _aggregate_levels(self.asks)
        }