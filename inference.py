"""
Main inference loop for OpenEnv LOB Simulator.

This is the entry point executed by Hugging Face Docker container.
Runs 3 tasks (easy/medium/hard) with LLM agent + LOBEnv.
Outputs logs in exact format for validator regex parsing.
"""

import os
import sys
from typing import Optional, Dict, Any

from core_engine.env import LOBEnv
from core_engine.schema import AgentState, AgentAction
from agentic_llm.client import LLMTrader
from agentic_llm.logger import log_start, log_step, log_end


def main() -> None:
    """
    Main execution: Run 3 difficulty tasks with LLM agent.
    
    Environment Variables:
    - API_BASE_URL: OpenAI API endpoint (default: https://api.openai.com/v1)
    - MODEL_NAME: LLM model name (default: gpt-4-turbo)
    - HF_TOKEN: API key from Hugging Face (used as api_key)
    - OPENAI_API_KEY: Fallback if HF_TOKEN missing
    
    Raises:
        ValueError: If no API key provided
    
    Output Format (for validation):
    [START] Episode initialized. Task Level: {task_level}.
    [STEP] {step_num} | Action: {...} | Reward: {reward:.4f} | Info: {...}
    [END] Episode complete. Final Score: {score:.4f} | Implementation Shortfall: {shortfall:.2f}.
    """
    
    # Read configuration from environment
    api_base_url: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4-turbo")
    api_key: Optional[str] = os.getenv("HF_TOKEN")
    
    # Fallback to OPENAI_API_KEY if HF_TOKEN not set
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Validate API key
    if not api_key:
        raise ValueError(
            "No API key provided. Set HF_TOKEN or OPENAI_API_KEY environment variable."
        )
    
    # Iterate through task levels
    for task_level in ["easy", "medium", "hard"]:
        try:
            # Initialize environment and agent for this task
            env: LOBEnv = LOBEnv(task_level=task_level)
            trader: LLMTrader = LLMTrader(
                api_key=api_key,
                model_name=model_name,
                api_base_url=api_base_url
            )
            
            # Log episode start
            log_start(task_level)
            
            # Reset environment
            state: AgentState = env.reset()
            done: bool = False
            step_count: int = 0
            final_info: Dict[str, Any] = {}
            
            # Main simulation loop
            while not done:
                # Get action from LLM trader
                action: AgentAction = trader.decide_action(state)
                
                # Step environment
                state, reward, done, info = env.step(action)
                step_count += 1
                final_info = info  # Keep final info for end logging
                
                # Log step
                log_step(step_count, action, reward, info)
            
            # Extract final metrics from info dictionary
            score: float = final_info.get("score", 0.0)
            implementation_shortfall: float = final_info.get(
                "implementation_shortfall", 0.0
            )
            
            # Log episode end
            log_end(score, implementation_shortfall)
            
        except Exception as e:
            print(f"[ERROR] Task {task_level} failed: {e}")
            raise


if __name__ == "__main__":
    main()
