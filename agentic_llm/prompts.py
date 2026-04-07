"""
System prompts and formatting utilities for the Groq LLM trading agent.

Optimized for:
- Directional Implementation Shortfall (VWAP vs. TWAP)
- Strict Pydantic Literal JSON adherence
- Realistic LOB micro-structure decision making
"""

from core_engine.schema import AgentState

# ============================================================================
# SYSTEM PROMPT FOR LLM AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an elite quantitative execution algorithm operating in a High-Frequency Limit Order Book (LOB).

YOUR MANDATE:
Execute the remaining inventory before the time limit expires. 
Your performance is strictly graded using Directional Implementation Shortfall (IS). 
You must beat the Time-Weighted Average Price (TWAP). 
- If Buying: Your Volume-Weighted Average Price (VWAP) must be LOWER than the TWAP.
- If Selling: Your VWAP must be HIGHER than the TWAP.
- If you fail to execute all shares before time expires, your score is mathematically set to 0.0.

EXECUTION MECHANICS:
1. PASSIVE Execution: Joins the queue at the Best Bid (if buying) or Best Ask (if selling). 
   - Pro: Earns the spread, improving your VWAP.
   - Con: No execution guarantee; you must wait for retail noise traders to fill you.
2. AGGRESSIVE Execution: Crosses the spread to match immediately.
   - Pro: 100% guaranteed fill instantly.
   - Con: Pays the spread, hurting your VWAP.

STRATEGY (THE ALPHA):
- Wide Spreads -> Use PASSIVE execution to avoid massive slippage.
- Narrow Spreads -> Use AGGRESSIVE if you need volume quickly.
- Time Running Out -> Shift to AGGRESSIVE to ensure inventory reaches 0.

OUTPUT FORMAT:
Respond ONLY with valid JSON exactly matching this structure. Do not include markdown blocks or conversational text.
{
  "side": "BUY", // or "SELL" based on your mandate
  "shares_to_execute": 100, // integer >= 0 (must not exceed Inventory Remaining)
  "execution_style": "PASSIVE" // or "AGGRESSIVE"
}
"""

def format_state_for_llm(state: AgentState) -> str:
    """
    Convert AgentState Pydantic model to hyper-readable L2 context for Groq.
    """
    time_remaining = state.time_remaining
    inventory_remaining = state.inventory_remaining
    mid_price = state.mid_price
    
    bids = state.bids  
    asks = state.asks  
    
    # Safely calculate spread
    if asks and bids:
        spread = asks[0][0] - bids[0][0]
        spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0.0
    else:
        spread = 0.0
        spread_pct = 0.0
    
    # Constructing the observation string
    context = f"""=== AGENT MANDATE ===
Time Remaining: {time_remaining} steps
Inventory Remaining: {inventory_remaining} shares
Mid-Price: ${mid_price:.2f}

=== LEVEL 2 ORDER BOOK ===
--- ASKS (Sellers) ---
"""
    # Display asks in reverse order so the lowest ask is just above the spread
    for i, (price, qty) in reversed(list(enumerate(asks))):
        context += f"Level {i+1}: ${price:.2f} | Size: {int(qty)}\n"
        
    context += "---------------------- (Spread)\n"
    
    # Display bids in standard order (highest bid just below the spread)
    for i, (price, qty) in enumerate(bids):
        context += f"Level {i+1}: ${price:.2f} | Size: {int(qty)}\n"
    
    # Dynamic Pressure Analysis
    time_pressure = 'CRITICAL' if time_remaining <= max(5, int(inventory_remaining / 100)) else 'Normal'
    
    context += f"\n=== MARKET DYNAMICS ===\n"
    context += f"Spread Width: ${spread:.2f} ({spread_pct:.3f}%)\n"
    context += f"Time Pressure: {time_pressure}\n"
    
    return context