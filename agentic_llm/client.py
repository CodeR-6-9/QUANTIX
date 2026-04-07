# filepath: agentic_llm/client.py
"""
Hugging Face Router Client Wrapper for LLM-based Trading Agent.

This module provides the LLMTrader class optimized for the HF Inference API,
maintaining OpenAI compatibility while providing robust error handling.
"""

import json
from typing import Optional, Dict, Any
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from core_engine.schema import AgentState, AgentAction
from agentic_llm.prompts import SYSTEM_PROMPT, format_state_for_llm


class LLMTrader:
    """
    LLMTrader: Orchestrates the trading loop using the Hugging Face Router.
    
    Decision Pipeline:
    State (LOB) -> Context Formatting -> HF Model Inference -> JSON Parsing -> Action
    """
    
    def __init__(
        self,
        api_key: str,
        api_base_url: str = "https://router.huggingface.co/v1",
        model_name: str = "llama-3.3-70b-versatile"
    ) -> None:
        """
        Initialize the trader pointing to the Hugging Face Router.
        
        Args:
            api_key: Your HF_TOKEN
            api_base_url: HF Router endpoint
            model_name: The model repo ID (e.g., Qwen/Qwen2.5-72B-Instruct)
        """
        if not api_key or not api_key.strip():
            raise ValueError("HF_TOKEN (api_key) cannot be empty")
        
        self.model_name = model_name
        # The OpenAI client is compatible with the HF Router
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )
        
        # Monitoring Stats
        self.call_count = 0
        self.total_tokens = 0
        self.error_count = 0
    
    def decide_action(self, state: AgentState) -> AgentAction:
        """
        Primary decision loop. State -> LLM -> Action.
        """
        try:
            # PHASE 1: Format market state for the model
            market_context = format_state_for_llm(state)
            
            # PHASE 2: Inference via HF Router
            # Note: response_format={"type": "json_object"} is excluded here for 
            # maximum compatibility across different HF models. Manual parsing 
            # below handles code blocks reliably.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user", 
                        "content": f"Current market state:\n{market_context}\n\nRespond with JSON only."
                    }
                ],
                temperature=0.1,  # Low temp for deterministic execution logic
                max_tokens=250,
                timeout=120.0      # Increased timeout for large HF models
            )
            
            # Update Usage Stats
            self.call_count += 1
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            
            # PHASE 3: Parse and Validate
            response_text = response.choices[0].message.content.strip()
            return self._parse_json_response(response_text, state)
            
        except (APIError, APIConnectionError, RateLimitError) as e:
            self.error_count += 1
            print(f"[LLMTrader] HF Router API Error: {type(e).__name__}: {str(e)[:100]}")
            return self._safe_fallback_action(state)
            
        except Exception as e:
            self.error_count += 1
            print(f"[LLMTrader] Unexpected Error: {type(e).__name__}: {str(e)[:100]}")
            return self._safe_fallback_action(state)
    
    def _parse_json_response(self, response_text: str, state: AgentState) -> AgentAction:
        """
        Robust JSON extractor that handles markdown code blocks.
        """
        cleaned = response_text.strip()
        
        # Remove common LLM markdown wrapping
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        # Load raw data
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[LLMTrader] JSON Decode Failed: {e}")
            return self._safe_fallback_action(state)
        
        # Side validation
        side = str(data.get("side", "SELL")).upper()
        if side not in ["BUY", "SELL"]:
            side = "SELL"
        
        # Shares validation (Clamped to current inventory)
        shares = int(data.get("shares_to_execute", 0))
        shares = max(0, min(shares, state.inventory_remaining))
        
        # Style validation
        style = str(data.get("execution_style", "PASSIVE")).upper()
        if style not in ["AGGRESSIVE", "PASSIVE"]:
            style = "PASSIVE"
        
        return AgentAction(
            side=side,
            shares_to_execute=shares,
            execution_style=style
        )
    
    def _safe_fallback_action(self, state: AgentState) -> AgentAction:
        """
        Safe 'Hold' action used when the LLM is unreachable or incoherent.
        """
        return AgentAction(
            side="SELL",
            shares_to_execute=0,
            execution_style="PASSIVE"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics for the dashboard monitoring session."""
        return {
            "model": self.model_name,
            "api_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "errors": self.error_count,
            "avg_tokens": (self.total_tokens / self.call_count if self.call_count > 0 else 0)
        }