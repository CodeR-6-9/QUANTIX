"""
Structured logging for trading activity.

This module provides logging utilities for tracking agent actions,
observations, and rewards in a standardized format required by OpenEnv validators.

Strict format: Judges' regex parser relies entirely on these formats.
"""

from typing import Dict, Any
import json

from core_engine.schema import AgentAction


def log_start(task_level: str) -> None:
    """
    Log episode start with task level.
    
    Prints exactly: [START] Episode initialized. Task Level: {task_level}.
    
    Args:
        task_level: Task difficulty level (easy/medium/hard)
    """
    print(f"[START] Episode initialized. Task Level: {task_level}.")


def log_step(
    step_num: int,
    action: AgentAction,
    reward: float,
    info: Dict[str, Any]
) -> None:
    """
    Log a single step with exact format for validation.
    
    Prints exactly: [STEP] {step_num} | Action: {action.model_dump_json()} | Reward: {reward:.4f} | Info: {info}
    
    Args:
        step_num: Step counter (1-based)
        action: AgentAction taken by LLM
        reward: Reward value from env.step()
        info: Info dictionary from env.step()
    """
    action_json = action.model_dump_json()
    print(f"[STEP] {step_num} | Action: {action_json} | Reward: {reward:.4f} | Info: {info}")


def log_end(score: float, shortfall: float) -> None:
    """
    Log episode completion with final metrics.
    
    Prints exactly: [END] Episode complete. Final Score: {score:.4f} | Implementation Shortfall: {shortfall:.2f}.
    
    Args:
        score: Final episode score
        shortfall: Implementation Shortfall value
    """
    print(f"[END] Episode complete. Final Score: {score:.4f} | Implementation Shortfall: {shortfall:.2f}.")
