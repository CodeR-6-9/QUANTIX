"""
Tests for the institutional Limit Order Book matching engine.

Verifies Price-Time priority matching, partial fills, 
lazy order deletion, and Level 2 depth aggregation.
"""

import pytest
import time

from core_engine.schema import Order, Trade
from core_engine.matching_engine import LimitOrderBook


@pytest.fixture
def empty_book() -> LimitOrderBook:
    """Fixture to provide a fresh order book for each test."""
    return LimitOrderBook(symbol="TEST_COIN")

# ==========================================
# 1. CORE INSERTION & QUOTING
# ==========================================

def test_engine_initializes(empty_book: LimitOrderBook) -> None:
    """Test that the matching engine boots up correctly."""
    assert empty_book.symbol == "TEST_COIN"
    assert len(empty_book.bids) == 0
    assert len(empty_book.asks) == 0
    assert len(empty_book.active_orders) == 0

def test_add_passive_orders(empty_book: LimitOrderBook) -> None:
    """Test posting liquidity that does not cross the spread."""
    # Post Bid
    bid = Order(order_id="B1", side="BUY", price=100.0, quantity=50, timestamp=1.0, agent_id="MM")
    trades_1 = empty_book.add_order(bid)
    
    # Post Ask
    ask = Order(order_id="A1", side="SELL", price=101.0, quantity=50, timestamp=2.0, agent_id="MM")
    trades_2 = empty_book.add_order(ask)
    
    # Assert no trades happened (passive)
    assert len(trades_1) == 0
    assert len(trades_2) == 0
    
    # Assert book state
    b_price, b_qty, a_price, a_qty = empty_book.get_best_bid_ask()
    assert b_price == 100.0
    assert b_qty == 50
    assert a_price == 101.0
    assert a_qty == 50

# ==========================================
# 2. MATCHING & SPREAD CROSSING
# ==========================================

def test_full_execution(empty_book: LimitOrderBook) -> None:
    """Test an aggressive order that completely fills against a resting order."""
    # Market Maker posts Ask
    empty_book.add_order(Order(order_id="A1", side="SELL", price=100.0, quantity=50, timestamp=1.0, agent_id="MM"))
    
    # Agent Buys exactly 50 shares
    agent_buy = Order(order_id="BUY1", side="BUY", price=100.0, quantity=50, timestamp=2.0, agent_id="AGENT")
    trades = empty_book.add_order(agent_buy)
    
    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert trades[0].quantity == 50
    assert trades[0].buyer_id == "AGENT"
    assert trades[0].seller_id == "MM"
    
    # Book should be empty now
    b_price, _, a_price, _ = empty_book.get_best_bid_ask()
    assert a_price is None

def test_partial_execution(empty_book: LimitOrderBook) -> None:
    """Test an aggressive order that only partially fills a resting order."""
    # Market Maker posts huge Ask
    empty_book.add_order(Order(order_id="A1", side="SELL", price=100.0, quantity=500, timestamp=1.0, agent_id="MM"))
    
    # Agent Buys 50 shares
    agent_buy = Order(order_id="BUY1", side="BUY", price=100.0, quantity=50, timestamp=2.0, agent_id="AGENT")
    trades = empty_book.add_order(agent_buy)
    
    assert len(trades) == 1
    assert trades[0].quantity == 50
    
    # MM should have 450 shares left at the best ask
    _, _, a_price, a_qty = empty_book.get_best_bid_ask()
    assert a_price == 100.0
    assert a_qty == 450

def test_market_sweep(empty_book: LimitOrderBook) -> None:
    """Test a large aggressive order that sweeps multiple price levels."""
    # Post 3 levels of Asks
    empty_book.add_order(Order(order_id="A1", side="SELL", price=101.0, quantity=10, timestamp=1.0, agent_id="MM"))
    empty_book.add_order(Order(order_id="A2", side="SELL", price=102.0, quantity=10, timestamp=2.0, agent_id="MM"))
    empty_book.add_order(Order(order_id="A3", side="SELL", price=103.0, quantity=10, timestamp=3.0, agent_id="MM"))
    
    # Agent buys 25 shares at $105.00 (Aggressive Market Sweep)
    sweep_buy = Order(order_id="SWEEP", side="BUY", price=105.0, quantity=25, timestamp=4.0, agent_id="AGENT")
    trades = empty_book.add_order(sweep_buy)
    
    # Should execute 2 full trades and 1 partial trade
    assert len(trades) == 3
    assert trades[0].price == 101.0 and trades[0].quantity == 10
    assert trades[1].price == 102.0 and trades[1].quantity == 10
    assert trades[2].price == 103.0 and trades[2].quantity == 5  # Partial fill on level 3
    
    # Best ask should now be 103.0 with 5 shares left
    _, _, a_price, a_qty = empty_book.get_best_bid_ask()
    assert a_price == 103.0
    assert a_qty == 5

# ==========================================
# 3. ADVANCED LOB MECHANICS
# ==========================================

def test_lazy_cancellation(empty_book: LimitOrderBook) -> None:
    """Test O(1) lazy deletion and amortized heap cleanup."""
    empty_book.add_order(Order(order_id="B1", side="BUY", price=100.0, quantity=50, timestamp=1.0, agent_id="MM"))
    empty_book.add_order(Order(order_id="B2", side="BUY", price=99.0, quantity=50, timestamp=2.0, agent_id="MM"))
    
    # Cancel the best bid
    assert empty_book.cancel_order("B1") is True
    
    # The get_best_bid_ask function should safely pop the 'dead' order
    # and reveal the next best bid ($99.0)
    b_price, b_qty, _, _ = empty_book.get_best_bid_ask()
    assert b_price == 99.0
    assert b_qty == 50

def test_l2_state_aggregation(empty_book: LimitOrderBook) -> None:
    """Test that L2 correctly aggregates quantities at the same price level."""
    # Post two bids at exactly $100.0
    empty_book.add_order(Order(order_id="B1", side="BUY", price=100.0, quantity=50, timestamp=1.0, agent_id="MM1"))
    empty_book.add_order(Order(order_id="B2", side="BUY", price=100.0, quantity=25, timestamp=2.0, agent_id="MM2"))
    # Post one bid lower
    empty_book.add_order(Order(order_id="B3", side="BUY", price=99.0, quantity=10, timestamp=3.0, agent_id="MM3"))
    
    l2 = empty_book.get_l2_state()
    
    # Top bid level should aggregate 50 + 25 = 75 shares
    assert l2["bids"][0] == (100.0, 75)
    assert l2["bids"][1] == (99.0, 10)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])