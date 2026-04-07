"""
System prompts and formatting utilities for LLM trading agent.

This module provides:
- SYSTEM_PROMPT: Guides LLM to optimize execution (minimize Implementation Shortfall)
- format_state_for_llm(): Converts AgentState to readable market context
"""

from core_engine.schema import AgentState

# ============================================================================
# SYSTEM PROMPT FOR LLM AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an elite quantitative execution algorithm.

YOUR MISSION:
Liquidate a large block of shares efficiently, minimizing market impact and 
Implementation Shortfall (IS). You have limited time and must finish before 
the clock runs out.

KEY CONSTRAINTS:
1. Complete all execution before time expires 
2. Minimize slippage: |execution_price - mid_price| × quantity
3. Adapt strategy based on remaining time and inventory
4. Use AGGRESSIVE execution when desperate (time/inventory running out)
5. Use PASSIVE execution when you have flexibility

MARKET DYNAMICS:
- You see Level 2 order book (top 3 bids/asks)
- Market Maker provides liquidity at predictable spreads
- Noise Trader adds randomness
- Mid-price = (best_bid + best_ask) / 2

ACTION SPACE:
- side: "BUY" or "SELL" 
- shares_to_execute: integer >= 0 (capped by inventory)
- execution_style: "AGGRESSIVE" (crosses spread) or "PASSIVE" (at best bid/ask)

OUTPUT FORMAT:
Respond ONLY with valid JSON exactly matching this structure (do not add extra keys):
{
  "side": "BUY",
  "shares_to_execute": 100,
  "execution_style": "PASSIVE"
}

RULES:
- Output MUST be valid JSON, no markdown, no extra text
- Always fill as much inventory as possible before time runs out
- Big spreads? Use AGGRESSIVE to guarantee execution
- Small spreads? Use PASSIVE to save money
- Low time + high inventory? Shift to AGGRESSIVE urgency
"""

def format_state_for_llm(state: AgentState) -> str:
    """
    Convert AgentState Pydantic model to readable text for LLM.
    """
    time_remaining = state.time_remaining
    inventory_remaining = state.inventory_remaining
    mid_price = state.mid_price
    
    bids = state.bids  
    asks = state.asks  
    
    if asks and bids:
        spread = asks[0][0] - bids[0][0]
        spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0.0
    else:
        spread = 0.0
        spread_pct = 0.0
    
    context = f"""=== MARKET STATE ===
Time Remaining: {time_remaining} steps
Inventory Remaining: {inventory_remaining} shares
Mid-Price: ${mid_price:.2f}

=== BID SIDE (Best bids, descending) ===\n"""
    
    for i, (price, qty) in enumerate(bids):
        context += f"Level {i+1}: ${price:.2f} × {int(qty)} shares\n"
    
    context += f"\n=== ASK SIDE (Best asks, ascending) ===\n"
    
    for i, (price, qty) in enumerate(asks):
        context += f"Level {i+1}: ${price:.2f} × {int(qty)} shares\n"
    
    context += f"\n=== SPREAD & URGENCY ==="
    context += f"\nSpread: ${spread:.2f} ({spread_pct:.3f}%)"
    context += f"\nTime Pressure: {'CRITICAL' if time_remaining <= 3 else 'Normal' if time_remaining > 10 else 'Increasing'}"
    context += f"\nInventory Pressure: {'CRITICAL' if inventory_remaining >= 500 else 'Normal' if inventory_remaining < 200 else 'Moderate'}"
    
    return context