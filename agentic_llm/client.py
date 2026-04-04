"""
OpenAI Client Wrapper for LLM-based Trading Agent.

Provides LLMTrader class that:
1. Formats market state into readable context
2. Calls OpenAI API with trading task system prompt
3. Parses JSON response into AgentAction
4. Handles errors gracefully with fallback behavior
"""

import json
import re
from typing import Optional, Dict, Any

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from core_engine.schema import AgentState, AgentAction
from agentic_llm.prompts import SYSTEM_PROMPT, format_state_for_llm


class LLMTrader:
    """
    LLMTrader: OpenAI-powered execution algorithm for block liquidation.
    
    Orchestrates the complete trading loop:
    1. Observes market state (L2 order book, time, inventory)
    2. Formats state into readable context via format_state_for_llm()
    3. Calls OpenAI API with SYSTEM_PROMPT + current market state
    4. Parses JSON response to extract trading decision
    5. Returns AgentAction (side, shares, style) for LOBEnv to execute
    6. Handles LLM errors gracefully (returns safe fallback action)
    
    Performance is evaluated by grader.calculate_score() on Implementation Shortfall.
    """
    
    def __init__(
        self,
        api_key: str,
        api_base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4-turbo"
    ) -> None:
        """
        Initialize LLMTrader with OpenAI credentials.
        
        Args:
            api_key: OpenAI API key (required)
            api_base_url: API base URL (default: OpenAI's official endpoint)
            model_name: Model to use (default: gpt-4-turbo for better reasoning)
            
        Raises:
            ValueError: If api_key is empty or credentials are invalid
        
        Complexity: O(1) - minimal initialization
        """
        if not api_key or not api_key.strip():
            raise ValueError("api_key cannot be empty")
        
        self.api_key: str = api_key
        self.api_base_url: str = api_base_url
        self.model_name: str = model_name
        
        # Initialize OpenAI client with specified credentials
        self.client: OpenAI = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )
        
        # Statistics tracking
        self.call_count: int = 0
        self.total_tokens: int = 0
        self.error_count: int = 0
    
    def decide_action(self, state: AgentState) -> AgentAction:
        """
        LLM Decision Pipeline: State → LLM → Action
        
        End-to-end orchestration:
        1. Call format_state_for_llm(state) to get readable market context
        2. Send SYSTEM_PROMPT + market context to OpenAI
        3. Request JSON response (structured output)
        4. Parse response JSON → extract side, shares, style
        5. Validate & construct AgentAction
        6. On any error, fallback to safe default (SELL 0 PASSIVE)
        
        Args:
            state: AgentState from LOBEnv (time, inventory, L2 book)
            
        Returns:
            AgentAction with side/shares_to_execute/execution_style
            Always returns valid AgentAction (never raises exception)
        
        Complexity: O(1) - API call time dominates, not algorithmic
        """
        try:
            # PHASE 1: Format state into readable context for LLM
            market_context = format_state_for_llm(state)
            
            # PHASE 2: Call OpenAI API with structured output
            # Request: JSON response for strict parsing
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Current market state:\n{market_context}\n\nRespond with JSON only."
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=200,
                timeout=10.0
            )
            
            # Track API usage
            self.call_count += 1
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            
            # PHASE 3: Extract and parse JSON response
            response_text = response.choices[0].message.content.strip()
            action = self._parse_json_response(response_text, state)
            
            return action
            
        except (APIError, APIConnectionError, RateLimitError) as e:
            # OpenAI API errors - log and fallback
            self.error_count += 1
            print(f"[LLMTrader] API Error (attempt {self.error_count}): {type(e).__name__}: {str(e)[:100]}")
            return self._safe_fallback_action(state)
            
        except ValueError as e:
            # JSON parsing error
            self.error_count += 1
            print(f"[LLMTrader] Parse Error: {e}")
            return self._safe_fallback_action(state)
            
        except Exception as e:
            # Catch-all for unexpected errors
            self.error_count += 1
            print(f"[LLMTrader] Unexpected Error: {type(e).__name__}: {str(e)[:100]}")
            return self._safe_fallback_action(state)
    
    def _parse_json_response(self, response_text: str, state: AgentState) -> AgentAction:
        """
        Parse LLM's JSON response into AgentAction.
        
        Handles:
        - Valid JSON with all required fields
        - JSON wrapped in markdown code blocks (```json ... ```)
        - Missing fields (uses defaults)
        - Invalid types (coerces to correct type)
        
        Args:
            response_text: Raw LLM response (hopefully JSON)
            state: Current state (for context/validation)
            
        Returns:
            Validated AgentAction with side/shares/style
            
        Raises:
            ValueError: If JSON parsing fails completely
            
        Complexity: O(1) - fixed parsing overhead
        """
        # Strip markdown code blocks if present
        # LLMs sometimes wrap JSON in ```json ... ```
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]  # Remove ```json
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]  # Remove ```
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Parse JSON
        data = json.loads(cleaned)
        
        # Extract fields with defaults and type coercion
        side = str(data.get("side", "SELL")).upper()
        if side not in ["BUY", "SELL"]:
            side = "SELL"
        
        shares = int(data.get("shares_to_execute", 0))
        shares = max(0, min(shares, state.inventory_remaining))  # Clamp to valid range
        
        style = str(data.get("execution_style", "PASSIVE")).upper()
        if style not in ["AGGRESSIVE", "PASSIVE"]:
            style = "PASSIVE"
        
        # Construct AgentAction
        action = AgentAction(
            side=side,
            shares_to_execute=shares,
            execution_style=style
        )
        
        return action
    
    def _safe_fallback_action(self, state: AgentState) -> AgentAction:
        """
        Return safe default action when LLM fails.
        
        Fallback strategy: Do nothing (shares_to_execute=0)
        This prevents crashes while allowing graceful error recovery.
        
        Args:
            state: Current state (unused in fallback)
            
        Returns:
            Safe AgentAction: SELL 0 shares at PASSIVE style (effectively HOLD)
            
        Complexity: O(1)
        """
        return AgentAction(
            side="SELL",
            shares_to_execute=0,
            execution_style="PASSIVE"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get LLMTrader statistics for monitoring.
        
        Returns:
            Dict with API calls, total tokens, error count
            
        Complexity: O(1)
        """
        return {
            "model": self.model_name,
            "api_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "errors": self.error_count,
            "avg_tokens_per_call": (
                self.total_tokens / self.call_count if self.call_count > 0 else 0
            )
        }
