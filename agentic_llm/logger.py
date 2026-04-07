"""
Structured logging for trading activity.

STRICT FORMAT: Must perfectly match OpenEnv regex parser.
DO NOT add spaces, pipes, or conversational text.
"""

import os
from typing import Optional, List
from core_engine.schema import AgentAction

def log_start(task: str, env: str = "Institutional_LOB_Execution") -> None:
    """
    Log episode start.
    Required format: [START] task=<task_name> env=<benchmark> model=<model_name>
    """
    model = os.getenv("MODEL_NAME", "gpt-4-turbo")
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(
    step: int, 
    action: AgentAction, 
    reward: float, 
    done: bool, 
    error: Optional[str] = None
) -> None:
    """
    Log a single step.
    Required format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    """
    # Compact JSON to remove spaces that break regex
    action_str = action.model_dump_json().replace(" ", "") 
    error_val = error if error else "null"
    done_val = str(done).lower()
    
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", 
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """
    Log episode completion.
    Required format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", 
        flush=True
    )