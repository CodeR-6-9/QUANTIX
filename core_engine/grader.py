"""
Performance grading and evaluation metrics for execution quality.

Implements a Directional Implementation Shortfall (IS) scoring mechanism.
Compares the Agent's Volume-Weighted Average Price (VWAP) against the 
Time-Weighted Average Price (TWAP) benchmark.
"""

from typing import List
from .schema import Trade


def calculate_twap(initial_mid_price: float, final_mid_price: float) -> float:
    """
    Calculate Time-Weighted Average Price benchmark.
    Represents a passive, zero-intelligence execution benchmark.
    """
    return (initial_mid_price + final_mid_price) / 2.0


def calculate_score(
    agent_trades: List[Trade],
    total_target_shares: int,
    benchmark_price: float
) -> float:
    """
    Calculate execution quality score based on Directional Implementation Shortfall.
    
    Grading Rules:
    1. Incomplete execution = 0.0 (Severe penalty for failing the mandate)
    2. Beating the benchmark (Negative IS) = 1.0
    3. Missing the benchmark scales down linearly. A 5% slippage results in 0.0.
    """
    if total_target_shares <= 0 or benchmark_price <= 0 or not agent_trades:
        return 0.0

    # 1. Verification Phase: Did the agent finish the job?
    total_executed_shares = sum(trade.quantity for trade in agent_trades)
    if total_executed_shares < total_target_shares:
        return 0.0

    # 2. Directional Phase: Is the agent buying or selling?
    agent_bought = sum(t.quantity for t in agent_trades if t.buyer_id == "LLM-AGENT")
    agent_sold = sum(t.quantity for t in agent_trades if t.seller_id == "LLM-AGENT")
    is_buy_task = agent_bought >= agent_sold

    # 3. Execution Phase: Calculate Agent's VWAP
    total_volume = agent_bought + agent_sold
    total_dollars = sum(t.price * t.quantity for t in agent_trades)
    agent_vwap = total_dollars / total_volume if total_volume > 0 else 0.0

    # 4. Shortfall Phase: Directional calculation
    if is_buy_task:
        # BUYING: Shortfall is positive (bad) if Agent paid MORE than TWAP
        shortfall_per_share = agent_vwap - benchmark_price
    else:
        # SELLING: Shortfall is positive (bad) if Agent received LESS than TWAP
        shortfall_per_share = benchmark_price - agent_vwap

    # 5. Scoring Phase
    if shortfall_per_share <= 0:
        # The agent beat or perfectly matched the benchmark!
        return 1.0

    # Convert shortfall to a percentage of the benchmark
    shortfall_pct = shortfall_per_share / benchmark_price

    # Institutional Scaling: 
    # Multiply by 20 means a 5% deviation from benchmark drops the score to 0.0.
    # This enforces strict execution standards.
    score = max(0.0, 1.0 - (shortfall_pct * 20))
    
    return round(score, 4)