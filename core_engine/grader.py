"""
Performance grading and evaluation metrics for execution quality.

Implements an Institutional Multi-Factor Scoring mechanism.
Evaluates the Agent's VWAP against a blended benchmark of Arrival Price
and Continuous TWAP, while applying strict penalties for Timing Risk.
"""

from typing import List
from .schema import Trade


def calculate_continuous_twap(price_history: List[float]) -> float:
    """
    Calculate the true Continuous Time-Weighted Average Price.
    Represents the integral of the market price over the execution window.
    """
    if not price_history:
        return 0.0
    return sum(price_history) / len(price_history)


def calculate_score(
    agent_trades: List[Trade],
    total_target_shares: int,
    arrival_price: float,
    true_twap: float,
    max_steps: int,
    steps_taken: int
) -> float:
    """
    Calculate execution quality score using a Multi-Factor Implementation Shortfall.
    
    Grading Rules:
    1. Incomplete execution = 0.001 (Severe penalty for failing the mandate)
    2. Blended Benchmark = 50% Arrival Price + 50% Continuous TWAP
    3. Timing Penalty = Up to 10% score reduction for holding risk too long.
    """
    # OpenEnv Strict Bounds Lock
    if total_target_shares <= 0 or arrival_price <= 0 or not agent_trades:
        return 0.001

    # 1. Verification Phase: Did the agent finish the job?
    total_executed_shares = sum(trade.quantity for trade in agent_trades)
    if total_executed_shares < total_target_shares:
        return 0.001

    # 2. Directional Phase: Is the agent buying or selling?
    agent_bought = sum(t.quantity for t in agent_trades if t.buyer_id == "LLM-AGENT")
    agent_sold = sum(t.quantity for t in agent_trades if t.seller_id == "LLM-AGENT")
    is_buy_task = agent_bought >= agent_sold

    # 3. Execution Phase: Calculate Agent's VWAP
    total_volume = agent_bought + agent_sold
    total_dollars = sum(t.price * t.quantity for t in agent_trades)
    agent_vwap = total_dollars / total_volume if total_volume > 0 else 0.0

    # 4. Multi-Factor Benchmark Phase
    blended_benchmark = (arrival_price * 0.5) + (true_twap * 0.5)

    # 5. Shortfall Phase: Directional calculation
    if is_buy_task:
        # BUYING: Shortfall is positive (bad) if Agent paid MORE than benchmark
        shortfall_per_share = agent_vwap - blended_benchmark
    else:
        # SELLING: Shortfall is positive (bad) if Agent received LESS than benchmark
        shortfall_per_share = blended_benchmark - agent_vwap

    # 6. Scoring Phase
    if shortfall_per_share <= 0:
        base_score = 1.0 # The agent beat the benchmark!
    else:
        # Convert shortfall to a percentage of the benchmark
        shortfall_pct = shortfall_per_share / blended_benchmark
        # 5% deviation from benchmark drops the score to 0.0.
        base_score = max(0.0, 1.0 - (shortfall_pct * 20))

    # 7. Timing Risk Penalty
    # The longer it takes to execute, the more variance risk the portfolio took.
    # Max penalty is 10% (0.1) if the agent uses all available max_steps.
    time_penalty_factor = 1.0 - (0.1 * (steps_taken / max(1, max_steps)))
    
    final_score = base_score * time_penalty_factor
    
    # STRICT OPENENV VALIDATOR LOCK (Never return 0.0 or 1.0)
    return max(0.001, min(0.999, round(final_score, 4)))