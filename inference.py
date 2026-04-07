"""
Main inference loop for OpenEnv LOB Simulator.
PURE HUGGING FACE ROUTER VERSION - Audited for Regex Compliance.
"""

import os
import sys
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

from core_engine.env import LOBEnv
from core_engine.schema import AgentState, AgentAction
from agentic_llm.client import LLMTrader
from agentic_llm.logger import log_start, log_step, log_end

def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    
    model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # We update the environment variable so the logger.py picks up the correct model name
    os.environ["MODEL_NAME"] = model_name
    
    if not api_key:
        print("[DEBUG] CRITICAL: No API key provided (Set GROQ_API_KEY in .env)", flush=True)
        return

    # 2. ITERATE THROUGH TASKS
    for task_level in ["easy", "medium", "hard"]:
        step_count: int = 0
        rewards: List[float] = []
        score: float = 0.0
        success: bool = False
        
        try:
            # Initialize environment and trader
            env = LOBEnv(task_level=task_level)
            trader = LLMTrader(
                api_key=api_key,
                model_name=model_name,
                api_base_url=api_base_url
            )
            
            # [START] tag - Logs the task and the hardcoded HF model
            log_start(task=task_level, env="Institutional_LOB_Execution")
            
            state = env.reset()
            done = False
            final_info = {}
            
            while not done:
                error_msg = None
                try:
                    # Get decision from the LLM
                    action = trader.decide_action(state)
                except Exception as e:
                    # STRICT CLEANING: Remove anything that could break a regex line
                    error_msg = str(e).replace("\n", " ").replace("\r", " ").replace("|", "-")[:50]
                    action = AgentAction(side="SELL", shares_to_execute=0, execution_style="PASSIVE")
                
                # Progress the physics
                state, step_reward_obj, done, info = env.step(action)
                
                # REWARD EXTRACTION
                if hasattr(step_reward_obj, 'total_reward'):
                    reward_val = float(step_reward_obj.total_reward)
                elif hasattr(step_reward_obj, 'reward'):
                    reward_val = float(step_reward_obj.reward)
                else:
                    try: reward_val = float(step_reward_obj)
                    except: reward_val = 0.0
                
                step_count += 1
                rewards.append(reward_val)
                final_info = info 
                
                # [STEP] tag
                log_step(
                    step=step_count, 
                    action=action, 
                    reward=reward_val, 
                    done=done, 
                    error=error_msg
                )
            
            # Final Metrics
            score = float(final_info.get("score", 0.0))
            success = bool(score > 0.0)
            
            # [END] tag
            log_end(success=success, steps=step_count, score=score, rewards=rewards)
            
        except Exception as e:
            # Emergency log to ensure we never miss an [END] tag
            log_end(success=False, steps=step_count, score=0.0, rewards=rewards)

if __name__ == "__main__":
    main()