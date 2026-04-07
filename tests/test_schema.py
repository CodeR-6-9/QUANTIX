"""
Tests for data schema validation and LOB physics.

Verifies Pydantic strict typing, validation rejections, and the custom
Price-Time Priority sorting logic inside the Order objects.
"""

import pytest
from pydantic import ValidationError

from core_engine.schema import (
    Order,
    Trade,
    AgentState,
    AgentAction,
    StepReward
)

# ==========================================
# 1. ORDER & PRICE-TIME PRIORITY TESTS
# ==========================================

def test_order_model_creation() -> None:
    """Test creating a valid Order object."""
    order = Order(
        order_id="O_001",
        side="BUY",
        price=150.0,
        quantity=100,
        timestamp=1600000000.0,
        agent_id="TRADER_1"
    )
    
    assert order.order_id == "O_001"
    assert order.side == "BUY"
    assert order.price == 150.0
    assert order.quantity == 100
    assert order.agent_id == "TRADER_1"

def test_order_validation_failures() -> None:
    """Test that Pydantic blocks invalid order data (negative quantities/prices)."""
    with pytest.raises(ValidationError):
        # Negative quantity should fail (gt=0)
        Order(order_id="1", side="BUY", price=100.0, quantity=-10, timestamp=1.0, agent_id="A")
        
    with pytest.raises(ValidationError):
        # Invalid side should fail (Literal["BUY", "SELL"])
        Order(order_id="1", side="SHORT", price=100.0, quantity=10, timestamp=1.0, agent_id="A")

def test_price_time_priority_bids() -> None:
    """
    Test BUY order sorting (Bids).
    Highest price should win. If prices match, oldest timestamp wins.
    """
    bid_low = Order(order_id="1", side="BUY", price=100.0, quantity=10, timestamp=2.0, agent_id="A")
    bid_high = Order(order_id="2", side="BUY", price=101.0, quantity=10, timestamp=2.0, agent_id="A")
    bid_old = Order(order_id="3", side="BUY", price=100.0, quantity=10, timestamp=1.0, agent_id="A")
    
    # Highest price wins
    assert bid_high < bid_low
    
    # Same price, older timestamp wins
    assert bid_old < bid_low
    
    # Python sorting check (should place best bid at index 0)
    book = [bid_low, bid_old, bid_high]
    book.sort()
    assert book[0].order_id == "2" # $101 wins
    assert book[1].order_id == "3" # $100, older wins
    assert book[2].order_id == "1" # $100, newer loses

def test_price_time_priority_asks() -> None:
    """
    Test SELL order sorting (Asks).
    Lowest price should win. If prices match, oldest timestamp wins.
    """
    ask_high = Order(order_id="1", side="SELL", price=101.0, quantity=10, timestamp=2.0, agent_id="A")
    ask_low = Order(order_id="2", side="SELL", price=100.0, quantity=10, timestamp=2.0, agent_id="A")
    ask_old = Order(order_id="3", side="SELL", price=101.0, quantity=10, timestamp=1.0, agent_id="A")
    
    # Lowest price wins
    assert ask_low < ask_high
    
    # Same price, older timestamp wins
    assert ask_old < ask_high
    
    # Python sorting check
    book = [ask_high, ask_low, ask_old]
    book.sort()
    assert book[0].order_id == "2" # $100 wins
    assert book[1].order_id == "3" # $101, older wins
    assert book[2].order_id == "1" # $101, newer loses

# ==========================================
# 2. STATE & ACTION TESTS
# ==========================================

def test_agent_state_model() -> None:
    """Test creating AgentState strictly with the new L2 Tuples."""
    state = AgentState(
        time_remaining=15,
        inventory_remaining=500,
        mid_price=150.0,
        bids=[(149.95, 100), (149.90, 200)],
        asks=[(150.05, 100), (150.10, 200)]
    )
    
    assert state.time_remaining == 15
    assert state.inventory_remaining == 500
    assert len(state.bids) == 2
    assert state.bids[0] == (149.95, 100)

def test_agent_action_strict_typing() -> None:
    """Test that AgentAction rejects LLM hallucinations."""
    # Valid Action
    action = AgentAction(
        side="SELL",
        shares_to_execute=100,
        execution_style="PASSIVE"
    )
    assert action.side == "SELL"
    
    # Invalid Action (Hallucinated execution style)
    with pytest.raises(ValidationError):
        AgentAction(side="BUY", shares_to_execute=10, execution_style="LIMIT")

# ==========================================
# 3. TRADE & REWARD TESTS
# ==========================================

def test_step_reward_model() -> None:
    """Test StepReward structure."""
    reward = StepReward(
        total_reward=-1.50, # Slippage is negative
        done=True
    )
    assert reward.total_reward == -1.50
    assert reward.done is True

def test_trade_model() -> None:
    """Test immutable Trade record creation."""
    trade = Trade(
        trade_id="T_1",
        buyer_id="AGENT",
        seller_id="MM",
        price=150.05,
        quantity=50,
        timestamp=1600000000.0
    )
    assert trade.quantity == 50
    assert trade.price == 150.05

if __name__ == "__main__":
    pytest.main([__file__, "-v"])