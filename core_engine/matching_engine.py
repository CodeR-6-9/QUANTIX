"""
Limit Order Book matching engine.

This module implements the core matching logic for the limit order book,
including order insertion, cancellation, and matching algorithms using
pure Python heapq. Performance characteristics:
- add_order(): O(log N) for heap operations, O(Q) for matching where Q is filled quantity
- cancel_order(): O(1) via lazy deletion
- get_l2_state(): O(log N) to iterate through heaps

Architecture:
- self.bids: Min-heap of (-price, timestamp, order_id) for max-price semantics
- self.asks: Min-heap of (price, timestamp, order_id) for min-price semantics
- self.active_orders: Dict mapping order_id -> Order for O(1) lookup
"""

from typing import List, Tuple, Dict, Optional
import heapq
import time

from .schema import Order, Trade


class LimitOrderBook:
    """
    High-performance Limit Order Book (LOB) implementation.
    
    Features:
    - Pro-rata matching at best available prices
    - Price-time priority semantics
    - O(1) order cancellation via lazy deletion
    - Efficient state queries with aggregated depth
    
    Matching Logic:
    - Incoming BUY orders match against ASK orders (lowest asks first)
    - Incoming SELL orders match against BID orders (highest bids first)
    - Matching at the best resting order price, not incoming price
    """
    
    def __init__(self, symbol: str = "AAPL") -> None:
        """
        Initialize an empty limit order book.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
        """
        self.symbol: str = symbol
        
        # Heap-based order book state
        # For BIDs: max-heap implemented as min-heap of negated prices
        # Heap entries: (neg_price, timestamp, order_id)
        self.bids: List[Tuple[float, float, str]] = []
        
        # For ASKs: min-heap of prices
        # Heap entries: (price, timestamp, order_id)
        self.asks: List[Tuple[float, float, str]] = []
        
        # Fast lookup and modification of orders
        # Maps order_id -> Order object for O(1) lookups
        self.active_orders: Dict[str, Order] = {}
        
        # Trade history and statistics
        self.trade_history: List[Trade] = []
        self.trade_counter: int = 0
        
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add a new order to the book and execute immediate matches.
        
        Matching Algorithm:
        1. Incoming BUY orders scan the ASK heap (lowest price first)
        2. Incoming SELL orders scan the BID heap (highest price first)
        3. Execute partial or full matches at resting order prices
        4. Use lazy deletion: mark quantity=0 for canceled orders in heap
        5. Post any remaining quantity to the appropriate heap
        
        Performance: O(log N) amortized for heap operations + O(Q) for matching
        where Q is the quantity matched.
        
        Args:
            order: Order object to add (must have quantity > 0)
            
        Returns:
            List of Trade objects executed during matching
        """
        # Validate order
        if order.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {order.quantity}")
        
        # Store order in active orders for fast lookup
        self.active_orders[order.order_id] = order
        
        # List to accumulate trades during matching
        trades: List[Trade] = []
        
        # Track remaining quantity to be filled
        remaining_quantity = order.quantity
        
        # ===== MATCHING PHASE =====
        if order.side == "BUY":
            # Incoming BUY: match against ASKs (lowest ask prices first)
            while remaining_quantity > 0 and self.asks:
                # Peek at best ask (min-heap, so lowest price is first)
                ask_price, ask_timestamp, ask_order_id = self.asks[0]
                
                # Get the resting ask order
                resting_order = self.active_orders.get(ask_order_id)
                
                # Skip this heap entry if it's been canceled (lazy deletion)
                if resting_order is None or resting_order.quantity == 0:
                    heapq.heappop(self.asks)
                    continue
                
                # Check if buy price crosses ask price
                if order.price < ask_price:
                    # No more possible matches
                    break
                
                # Execute trade at the resting order price (ASK price)
                match_quantity = min(remaining_quantity, resting_order.quantity)
                execution_price = resting_order.price  # Executed at ask price
                
                # Create trade record
                trade = Trade(
                    trade_id=f"T_{self.trade_counter}",
                    buyer_id=order.agent_id,
                    seller_id=resting_order.agent_id,
                    price=execution_price,
                    quantity=match_quantity,
                    timestamp=time.time()
                )
                trades.append(trade)
                self.trade_history.append(trade)
                self.trade_counter += 1
                
                # Update quantities
                remaining_quantity -= match_quantity
                resting_order.quantity -= match_quantity
                
                # If resting order is fully filled, remove from heap (lazy)
                if resting_order.quantity == 0:
                    heapq.heappop(self.asks)
        
        elif order.side == "SELL":
            # Incoming SELL: match against BIDs (highest bid prices first)
            while remaining_quantity > 0 and self.bids:
                # Peek at best bid (min-heap of negated prices, so most negative = highest price)
                neg_bid_price, bid_timestamp, bid_order_id = self.bids[0]
                bid_price = -neg_bid_price  # Convert back to positive
                
                # Get the resting bid order
                resting_order = self.active_orders.get(bid_order_id)
                
                # Skip this heap entry if it's been canceled (lazy deletion)
                if resting_order is None or resting_order.quantity == 0:
                    heapq.heappop(self.bids)
                    continue
                
                # Check if sell price crosses bid price
                if order.price > bid_price:
                    # No more possible matches
                    break
                
                # Execute trade at the resting order price (BID price)
                match_quantity = min(remaining_quantity, resting_order.quantity)
                execution_price = resting_order.price  # Executed at bid price
                
                # Create trade record
                trade = Trade(
                    trade_id=f"T_{self.trade_counter}",
                    buyer_id=resting_order.agent_id,
                    seller_id=order.agent_id,
                    price=execution_price,
                    quantity=match_quantity,
                    timestamp=time.time()
                )
                trades.append(trade)
                self.trade_history.append(trade)
                self.trade_counter += 1
                
                # Update quantities
                remaining_quantity -= match_quantity
                resting_order.quantity -= match_quantity
                
                # If resting order is fully filled, remove from heap (lazy)
                if resting_order.quantity == 0:
                    heapq.heappop(self.bids)
        
        # ===== POST-MATCHING PHASE =====
        # If there's remaining quantity, post to book
        if remaining_quantity > 0:
            # Update the order's quantity to reflect what remains
            order.quantity = remaining_quantity
            
            if order.side == "BUY":
                # Add to bid heap: use negated price for max-heap semantics
                heapq.heappush(
                    self.bids,
                    (-order.price, order.timestamp, order.order_id)
                )
            else:  # SELL
                # Add to ask heap: min-heap with regular price
                heapq.heappush(
                    self.asks,
                    (order.price, order.timestamp, order.order_id)
                )
        else:
            # Order was fully filled, remove from active orders
            del self.active_orders[order.order_id]
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order using lazy deletion (O(1) complexity).
        
        Implementation Notes:
        - Does NOT immediately remove from heaps (which would be O(N))
        - Instead, marks quantity=0 in active_orders
        - The matching loop eventually skips these "dead" entries
        - Periodic maintenance would be needed to clean heaps in production
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was found and canceled, False if not found
        """
        # Check if order exists
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        
        # Lazy deletion: mark quantity as 0
        # The matching loop will skip this order
        order.quantity = 0
        
        # Remove from active orders dictionary
        del self.active_orders[order_id]
        
        return True
    
    def get_l2_state(self) -> Dict[str, List[Tuple[float, int]]]:
        """
        Get Level 2 (L2) market data: top 3 price levels on each side.
        
        Returns aggregated quantity at each price level, ignoring canceled orders
        (those with quantity=0) via lazy deletion.
        
        Performance: O(log N) to iterate through heaps and skip dead entries.
        
        Returns:
            Dictionary with keys 'bids' and 'asks', each containing a list of
            (price, aggregated_quantity) tuples sorted from best to worst:
            - bids: highest price first
            - asks: lowest price first
        """
        bids_l2 = []
        asks_l2 = []
        
        # Create temporary copies of heaps to iterate without modifying
        bids_copy = self.bids.copy()
        asks_copy = self.asks.copy()
        
        # Collect bid levels
        price_qty_map: Dict[float, int] = {}
        while bids_copy and len(bids_l2) < 3:
            neg_price, timestamp, order_id = heapq.heappop(bids_copy)
            price = -neg_price
            
            # Skip canceled orders
            resting_order = self.active_orders.get(order_id)
            if resting_order is None or resting_order.quantity == 0:
                continue
            
            # Aggregate quantity at this price level
            if price in price_qty_map:
                price_qty_map[price] += resting_order.quantity
            else:
                price_qty_map[price] = resting_order.quantity
                if len(price_qty_map) <= 3:
                    bids_l2.append((price, price_qty_map[price]))
        
        # Update aggregated quantities in result
        bids_l2 = [(p, price_qty_map[p]) for p, _ in bids_l2]
        
        # Collect ask levels
        price_qty_map = {}
        while asks_copy and len(asks_l2) < 3:
            price, timestamp, order_id = heapq.heappop(asks_copy)
            
            # Skip canceled orders
            resting_order = self.active_orders.get(order_id)
            if resting_order is None or resting_order.quantity == 0:
                continue
            
            # Aggregate quantity at this price level
            if price in price_qty_map:
                price_qty_map[price] += resting_order.quantity
            else:
                price_qty_map[price] = resting_order.quantity
                if len(price_qty_map) <= 3:
                    asks_l2.append((price, price_qty_map[price]))
        
        # Update aggregated quantities in result
        asks_l2 = [(p, price_qty_map[p]) for p, _ in asks_l2]
        
        return {
            "bids": bids_l2,  # Highest price first (best bids)
            "asks": asks_l2   # Lowest price first (best asks)
        }
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]:
        """
        Get best bid and ask prices with quantities (skipping canceled orders).
        
        Returns:
            Tuple of (bid_price, bid_qty, ask_price, ask_qty)
            or (None, 0, None, 0) if book is empty on either side
        """
        # Find best bid
        bid_price, bid_qty = None, 0
        bids_copy = self.bids.copy()
        while bids_copy:
            neg_price, timestamp, order_id = heapq.heappop(bids_copy)
            resting_order = self.active_orders.get(order_id)
            if resting_order and resting_order.quantity > 0:
                bid_price = -neg_price
                bid_qty = resting_order.quantity
                break
        
        # Find best ask
        ask_price, ask_qty = None, 0
        asks_copy = self.asks.copy()
        while asks_copy:
            price, timestamp, order_id = heapq.heappop(asks_copy)
            resting_order = self.active_orders.get(order_id)
            if resting_order and resting_order.quantity > 0:
                ask_price = price
                ask_qty = resting_order.quantity
                break
        
        return bid_price, bid_qty, ask_price, ask_qty
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Retrieve an order by ID.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order object if found, None otherwise
        """
        return self.active_orders.get(order_id)
    
    def get_total_bids_quantity(self) -> int:
        """Get total quantity resting on bid side."""
        total = 0
        for order in self.active_orders.values():
            if order.side == "BUY" and order.quantity > 0:
                total += order.quantity
        return total
    
    def get_total_asks_quantity(self) -> int:
        """Get total quantity resting on ask side."""
        total = 0
        for order in self.active_orders.values():
            if order.side == "SELL" and order.quantity > 0:
                total += order.quantity
        return total
