"""
Tests for inference pipeline.

Test cases for end-to-end inference and environment interaction.
"""

import pytest
import os
from unittest.mock import Mock, patch

from core_engine.env import LOBEnv
from core_engine.schema import AgentState, AgentAction
from agentic_llm.logger import log_step, log_episode_summary


def test_environment_initializes() -> None:
    """Test that LOB environment initializes correctly."""
    env = LOBEnv()
    
    assert env.symbol == "AAPL"
    assert env.cash == 100000.0
    assert env.step_count == 0


def test_environment_reset() -> None:
    """Test environment reset functionality."""
    env = LOBEnv()
    state = env.reset()
    
    assert isinstance(state, AgentState)
    assert state.step == 0
    assert env.step_count == 0


def test_environment_step() -> None:
    """Test environment step with HOLD action."""
    env = LOBEnv(max_steps=10)
    state = env.reset()
    
    action = AgentAction(
        action_type="HOLD",
        symbol="AAPL"
    )
    
    next_state, reward, done = env.step(action)
    
    assert isinstance(next_state, AgentState)
    assert next_state.step == 1
    assert done is False  # Not done after 1 step with max_steps=10


def test_step_logging() -> None:
    """Test step logging output."""
    # Create mock objects
    action = AgentAction(
        action_type="BUY",
        symbol="AAPL",
        quantity=100.0,
        price=150.0
    )
    
    market_state = Mock()
    market_state.symbol = "AAPL"
    market_state.current_price = 150.0
    market_state.bid_price = 149.99
    market_state.ask_price = 150.01
    
    observation = Mock(spec=AgentState)
    observation.market_state = market_state
    
    reward = Mock(spec=object)
    reward.total_reward = 0.05
    
    # This should not raise an exception
    log_step(action, observation, reward, step_number=1)


def test_episode_summary_logging() -> None:
    """Test episode summary logging."""
    # This should not raise an exception
    log_episode_summary(
        episode_number=1,
        total_steps=100,
        total_reward=5.0,
        final_pnl=250.0,
        trades_executed=25,
        avg_slippage=0.002
    )


class TestInferenceFlow:
    """Test complete inference flow."""
    
    def test_inference_initialization(self) -> None:
        """Test inference module initialization."""
        # TODO: Test inference.py initialization
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
