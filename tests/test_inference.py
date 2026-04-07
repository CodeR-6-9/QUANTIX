"""
Tests for the OpenEnv inference pipeline.

Verifies the integration between LOBEnv task configurations,
step progression, episode termination, and regex-compliant logging.
"""

import pytest
from core_engine.env import LOBEnv
from core_engine.schema import AgentState, AgentAction, StepReward
from agentic_llm.logger import log_start, log_step, log_end

# ==========================================
# 1. ENVIRONMENT CONFIGURATION TESTS
# ==========================================

def test_environment_task_loading() -> None:
    """Test that LOBEnv loads the correct openenv.yaml parameters."""
    # Test Easy config
    env_easy = LOBEnv(task_level="easy")
    assert env_easy.target_shares == 500
    assert env_easy.max_steps == 100
    
    # Test Fallback (invalid task should default to medium/fallback)
    env_invalid = LOBEnv(task_level="super_hard_mode")
    assert getattr(env_invalid, 'target_shares', None) is not None

def test_environment_reset() -> None:
    """Test environment reset returns strict AgentState."""
    env = LOBEnv(task_level="easy")
    state = env.reset()
    
    assert isinstance(state, AgentState)
    assert state.time_remaining == 100
    assert state.inventory_remaining == 500
    assert env.step_count == 0

# ==========================================
# 2. STEP EXECUTION & TERMINATION TESTS
# ==========================================

def test_environment_step_execution() -> None:
    """Test environment step updates inventory and time correctly."""
    env = LOBEnv(task_level="easy")
    env.reset()
    
    action = AgentAction(
        side="BUY",
        shares_to_execute=50,
        execution_style="AGGRESSIVE"
    )
    
    next_state, reward, done, info = env.step(action)
    
    assert isinstance(next_state, AgentState)
    assert next_state.time_remaining == 99  # Time decreased
    assert next_state.inventory_remaining == 450  # Inventory decreased
    assert isinstance(reward, StepReward)
    assert done is False

def test_environment_termination_and_scoring() -> None:
    """Test episode completes when inventory hits 0 and calculates score."""
    env = LOBEnv(task_level="easy")
    env.reset()
    
    # Force complete execution in one step (Fat finger simulated)
    action = AgentAction(
        side="BUY",
        shares_to_execute=500, # Matches full target_shares
        execution_style="AGGRESSIVE"
    )
    
    next_state, reward, done, info = env.step(action)
    
    assert next_state.inventory_remaining == 0
    assert done is True
    assert "score" in info  # Grader should have triggered
    assert isinstance(info["score"], float)

# ==========================================
# 3. LOGGER SIGNATURE TESTS
# ==========================================

def test_logging_signatures() -> None:
    """
    Test that the logger functions do not throw TypeErrors when passed 
    the updated Pydantic models. We don't need to test stdout, just the signatures.
    """
    action = AgentAction(side="SELL", shares_to_execute=10, execution_style="PASSIVE")
    
    # These should execute without raising exceptions
    try:
        log_start(task="easy", env="Institutional_LOB")
        log_step(step=1, action=action, reward=-0.05, done=False, error=None)
        log_end(success=True, steps=1, score=0.95, rewards=[-0.05])
    except Exception as e:
        pytest.fail(f"Logger functions raised an unexpected exception: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])