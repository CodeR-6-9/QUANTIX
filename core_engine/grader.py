"""
Performance grading and evaluation metrics for execution quality.

This module implements Implementation Shortfall (IS) scoring, which measures
execution quality relative to a TWAP (Time-Weighted Average Price) benchmark.
The scoring function maps execution shortfall to a 0.0-1.0 performance score.

Key Concepts:
  - Benchmark Price: Simple average of start and end mid-prices (naive TWAP)
  - Benchmark Revenue: target_shares * benchmark_price (perfect execution)
  - Agent Revenue: sum(trade.price * trade.quantity) for executed trades
  - Implementation Shortfall: Benchmark Revenue - Agent Revenue
  - Score Mapping: shortfall -> score with penalties for incomplete execution
"""

from typing import List
from .schema import Trade


def calculate_twap(initial_mid_price: float, final_mid_price: float) -> float:
    """
    Calculate Time-Weighted Average Price benchmark.
    
    For simplicity, TWAP is the arithmetic mean of initial and final mid-prices.
    This represents a passive benchmark where the agent does nothing.
    
    Args:
        initial_mid_price: Starting mid-price (best_bid + best_ask) / 2
        final_mid_price: Ending mid-price after simulation
        
    Returns:
        TWAP benchmark price (float)
    
    Complexity: O(1)
    """
    return (initial_mid_price + final_mid_price) / 2.0


def calculate_score(
    agent_trades: List[Trade],
    total_target_shares: int,
    benchmark_price: float
) -> float:
    """
    Calculate execution quality score based on Implementation Shortfall.
    
    SCORE MAPPING:
      - Score = 1.0 if shortfall <= 0 (better than benchmark)
      - Score = 0.9 - (shortfall / benchmark_revenue) (linearly penalize)
      - Score = 0.0 if agent failed to execute all target_shares (harsh penalty)
    
    Implementation Shortfall (IS):
      - IS = |AgentCost - BenchmarkCost|
      - For BUY: AgentCost = sum(price * qty), BenchmarkCost = total_shares * benchmark_price
      - IS> 0 means agent paid more (or received less for SELL)
    
    Args:
        agent_trades: List of Trade objects executed by LLM agent
        total_target_shares: Total shares agent needed to execute
        benchmark_price: TWAP benchmark price (from calculate_twap)
        
    Returns:
        Score in range [0.0, 1.0] where 1.0 is perfect execution
    
    Raises:
        ValueError: If total_target_shares <= 0 or benchmark_price <= 0
    
    Complexity: O(N) where N = len(agent_trades)
    """
    if total_target_shares <= 0 or benchmark_price <= 0:
        raise ValueError("total_target_shares and benchmark_price must be positive")
    
    # PHASE 1: Calculate actual execution
    # =====================================
    total_executed_shares = sum(trade.quantity for trade in agent_trades)
    total_agent_revenue = sum(trade.price * trade.quantity for trade in agent_trades)
    
    # PHASE 2: Check for incomplete execution (SEVERE PENALTY)
    # ========================================================
    if total_executed_shares < total_target_shares:
        # Agent failed to execute all shares by end of episode
        # Return 0.0 as maximum penalty
        return 0.0
    
    # PHASE 3: Calculate Implementation Shortfall
    # ============================================
    # Benchmark: what agent would earn/pay with perfect execution at TWAP
    benchmark_revenue = total_target_shares * benchmark_price
    
    # Shortfall: how much agent lost relative to benchmark (always positive or zero)
    # For execution with average price paid, if agent paid more than benchmark, IS > 0
    shortfall = abs(total_agent_revenue - benchmark_revenue)
    
    # PHASE 4: Map shortfall to score [0.0, 1.0]
    # ============================================
    if shortfall <= 0.01:  # Near-zero shortfall (within 1 cent)
        # Agent achieved benchmark-matching or better execution
        score = 1.0
    else:
        # Linearly penalize based on IS as percentage of benchmark revenue
        # - 1% IS = 0.99 score
        # - 5% IS = 0.95 score
        # - 10% IS = 0.90 score
        # - 100% IS or more = 0.0 score
        is_percentage = shortfall / benchmark_revenue if benchmark_revenue > 0 else 1.0
        score = max(0.0, 1.0 - is_percentage)
    
    return score
