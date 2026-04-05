"""
Tests for data schema validation.

Test cases for Pydantic models and schema correctness.
"""

import pytest
from datetime import datetime

from core_engine.schema import (
    Order,
    OrderBookSnapshot,
    MarketState,
    AgentState,
    AgentAction,
    StepReward
)


def test_order_model_creation() -> None:
    """Test creating an Order object."""
    order = Order(
        order_id="test_001",
        side="BUY",
        price=150.0,
        quantity=100.0
    )
    
    assert order.order_id == "test_001"
    assert order.side == "BUY"
    assert order.price == 150.0
    assert order.quantity == 100.0


def test_agent_state_model() -> None:
    """Test creating AgentState."""
    market_state = MarketState(
        symbol="AAPL",
        current_price=150.0,
        bid_price=149.99,
        ask_price=150.01,
        bid_quantity=1000.0,
        ask_quantity=1000.0,
        volatility=0.15
    )
    
    snapshot = OrderBookSnapshot(
        mid_price=150.0,
        spread=0.02
    )
    
    state = AgentState(
        step=0,
        market_state=market_state,
        order_book=snapshot
    )
    
    assert state.step == 0
    assert state.market_state.symbol == "AAPL"
    assert state.agent_portfolio["cash"] == 0.0


def test_agent_action_model() -> None:
    """Test creating AgentAction."""
    action = AgentAction(
        action_type="BUY",
        symbol="AAPL",
        quantity=100.0,
        price=149.95
    )
    
    assert action.action_type == "BUY"
    assert action.quantity == 100.0
    assert action.price == 149.95


def test_step_reward_model() -> None:
    """Test creating StepReward."""
    reward = StepReward(
        total_reward=0.05,
        execution_reward=0.1,
        market_impact_penalty=-0.05,
        done=False
    )
    
    assert reward.total_reward == 0.05
    assert reward.done is False


def test_schema_validation() -> None:
    """Test Pydantic schema validation."""
    # TODO: Add comprehensive schema validation tests
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
