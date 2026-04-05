"""
Tests for the matching engine.

Test cases for order insertion, cancellation, and matching logic.
"""

import pytest
from datetime import datetime

from core_engine.schema import Order, OrderBookSnapshot
from core_engine.matching_engine import LimitOrderBook, OrderSide


def test_matching_engine_initializes() -> None:
    """Test that matching engine initializes correctly."""
    book = LimitOrderBook(symbol="AAPL")
    
    assert book.symbol == "AAPL"
    assert len(book.buy_orders) == 0
    assert len(book.sell_orders) == 0
    assert len(book.fill_history) == 0


def test_add_order_to_empty_book() -> None:
    """Test adding a single order to an empty book."""
    book = LimitOrderBook()
    
    order = Order(
        order_id="ord_001",
        side="BUY",
        price=100.0,
        quantity=10.0
    )
    
    fills, remaining = book.add_order(order)
    
    # In empty book, order should not fill
    assert len(fills) == 0
    assert remaining is not None
    assert remaining.order_id == "ord_001"


def test_order_matching() -> None:
    """Test that orders match correctly at best prices."""
    # TODO: Implement comprehensive matching tests
    assert True  # Dummy test to ensure pytest runs


def test_order_cancellation() -> None:
    """Test order cancellation logic."""
    # TODO: Implement cancellation tests
    assert True


def test_get_order_book_snapshot() -> None:
    """Test getting current order book snapshot."""
    # TODO: Implement snapshot tests
    assert True


def test_best_bid_ask() -> None:
    """Test retrieval of best bid and ask."""
    # TODO: Implement best price tests
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
