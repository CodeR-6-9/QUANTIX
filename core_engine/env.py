"""
Limit Order Book environment implementing OpenAI Gym interface.

This module provides the main LOBEnv simulation environment where:
  1. The agent (LLM) observes market state (L2 order book, mid-price, inventory)
  2. The agent submits trading action (shares_to_execute, execution_style)
  3. Background agents (MarketMaker, NoiseTrader) provide realistic liquidity
  4. The environment executes the agent's order against the LOB
  5. The environment returns next state, reward (slippage), and done flag

Termination condition: Agent completes all target_shares or max_steps reached.
"""

import time
from typing import Tuple, Dict, Any

from .schema import AgentState, AgentAction, StepReward, Trade
from .matching_engine import LimitOrderBook
from .background_agents.market_maker import MarketMaker
from .background_agents.noise_trader import NoiseTrader
from .grader import calculate_twap, calculate_score


class LOBEnv:
    """
    Limit Order Book Environment for Agent Training and Evaluation.
    
    OpenEnv-compliant interface where agents trade execution blocks by observing
    L2 market data and submitting limit/market orders. Background agents provide
    realistic liquidity. Episode ends when all target shares are executed or
    max_steps is reached.
    
    State Space:
      - time_remaining: steps left in episode [0, max_steps]
      - inventory_remaining: shares left to execute [0, target_shares]
      - mid_price: current (best_bid + best_ask) / 2
      - L2 data: top 3 bid/ask levels with quantities
    
    Action Space:
      - shares_to_execute: int in [0, inventory_remaining]
      - execution_style: "AGGRESSIVE" (market order) or "PASSIVE" (limit order)
    
    Reward:
      - Negative slippage: per-execution (intermediate)
      - Final score [0, 1]: based on Implementation Shortfall (terminal)
    """
    
    def __init__(
        self,
        task_level: str = "medium",
        symbol: str = "AAPL",
        initial_mid_price: float = 150.0
    ) -> None:
        """
        Initialize LOBEnv with specified difficulty level.
        
        Args:
            task_level: "easy" (50 shares, 15 steps) | "medium" (1000 shares, 20 steps) | "hard" (5000 shares, 25 steps)
            symbol: Trading instrument (e.g., "AAPL")
            initial_mid_price: Starting mid-price for simulation
            
        Complexity: O(1) initialization
        """
        self.symbol: str = symbol
        self.initial_mid_price: float = initial_mid_price
        self.current_time: float = 0.0
        
        # Task configuration based on difficulty
        task_config: Dict[str, Tuple[int, int]] = {
            "easy": (50, 15),           # (target_shares, max_steps)
            "medium": (1000, 20),
            "hard": (5000, 25)
        }
        if task_level not in task_config:
            raise ValueError(f"Invalid task_level: {task_level}. Choose from {list(task_config.keys())}")
        
        self.target_shares, self.max_steps = task_config[task_level]
        
        # Order book and background agents
        self.lob: LimitOrderBook = LimitOrderBook(symbol)
        self.market_maker: MarketMaker = MarketMaker(
            agent_id="MM-SIM",
            num_levels=3,
            spread_width=0.5,
            order_size=100
        )
        self.noise_trader: NoiseTrader = NoiseTrader(
            agent_id="NT-SIM",
            trade_probability=0.3,
            base_order_size=50
        )
        
        # Agent execution state
        self.inventory_remaining: int = self.target_shares  # Shares left to execute
        self.step_count: int = 0                            # Current step in episode
        self.agent_trades: list[Trade] = []                # Trades executed by LLM agent
        
        # Statistics and tracking
        self.episode_start_price: float = initial_mid_price
        self.episode_end_price: float = initial_mid_price
    
    def reset(self) -> AgentState:
        """
        Reset the environment for a new episode.
        
        Procedure:
          1. Clear order book
          2. Reset counters (step_count=0, inventory=target_shares, agent_trades=[])
          3. Pre-populate order book with background agent liquidity
          4. Return initial observation
        
        Returns:
            Initial AgentState observation for episode start
        
        Complexity: O(num_liquidity_steps) for pre-population
        """
        # Clear and reinitialize state
        self.lob = LimitOrderBook(self.symbol)
        self.market_maker = MarketMaker("MM-SIM", num_levels=3, spread_width=0.5, order_size=100)
        self.noise_trader = NoiseTrader("NT-SIM", trade_probability=0.3, base_order_size=50)
        
        self.step_count = 0
        self.current_time = 0.0
        self.inventory_remaining = self.target_shares
        self.agent_trades = []
        self.episode_start_price = self.initial_mid_price
        self.episode_end_price = self.initial_mid_price
        
        # PRE-POPULATION: Call market maker a few times to build initial liquidity
        # so the agent doesn't start with an empty book
        for _ in range(3):
            mm_result = self.market_maker.step(self.initial_mid_price, self.current_time)
            for order in mm_result["new_orders"]:
                self.lob.add_order(order)
            self.current_time += 0.1
        
        return self.state()
    
    def step(
        self,
        action: AgentAction
    ) -> Tuple[AgentState, StepReward, bool, Dict[str, Any]]:
        """
        Execute one simulation step: background agents -> LLM agent -> matching -> reward.
        
        Execution Flow:
          PHASE 1 (Background): Market maker and noise trader post/cancel orders
          PHASE 2 (LLM Action): Convert AgentAction to Order, submit to LOB
          PHASE 3 (Updates): Update inventory, advance time
          PHASE 4 (Reward): Calculate slippage penalty; check done condition
        
        Args:
            action: AgentAction with shares_to_execute (int) and execution_style ("AGGRESSIVE" or "PASSIVE")
            
        Returns:
            next_state: AgentState observation after step
            reward: StepReward with slippage penalty and done flag
            done: True if episode terminated (all shares executed or time limit)
            info: Dict with execution details, scores, and debugging info
        
        Complexity: O(log N) for matching, O(1) for reward calculation
        """
        self.step_count += 1
        self.current_time = float(self.step_count)  # Discrete time in steps
        
        # ========== PHASE 1: BACKGROUND AGENTS ==========
        # Get current market state before background orders
        bid_price, bid_qty, ask_price, ask_qty = self.lob.get_best_bid_ask()
        
        # Handle case where book is empty
        if bid_price is None:
            bid_price = self.initial_mid_price - 1.0
        if ask_price is None:
            ask_price = self.initial_mid_price + 1.0
        
        current_mid_price = (bid_price + ask_price) / 2.0
        
        # Market maker updates quotes
        mm_result = self.market_maker.step(current_mid_price, self.current_time)
        for order_id in mm_result["cancels"]:
            self.lob.cancel_order(order_id)
        for order in mm_result["new_orders"]:
            self.lob.add_order(order)
        
        # Noise trader places random orders
        nt_result = self.noise_trader.step(current_mid_price, self.current_time)
        for order_id in nt_result["cancels"]:
            self.lob.cancel_order(order_id)
        for order in nt_result["new_orders"]:
            self.lob.add_order(order)
        
        # Re-compute mid-price after background activity
        bid_price, bid_qty, ask_price, ask_qty = self.lob.get_best_bid_ask()
        if bid_price is None:
            bid_price = self.initial_mid_price - 1.0
        if ask_price is None:
            ask_price = self.initial_mid_price + 1.0
        current_mid_price = (bid_price + ask_price) / 2.0
        
        # ========== PHASE 2: LLM AGENT ACTION ==========
        step_trades: list[Trade] = []  # Trades executed in this step
        step_reward: float = 0.0       # Slippage penalty for this step
        
        if action.shares_to_execute > 0:
            # Determine execution price based on style
            if action.execution_style == "AGGRESSIVE":
                # Market order: cross spread to guarantee immediate fill
                # BUY @ (best_ask + 5.0), SELL @ (best_bid - 5.0)
                if action.side == "BUY":
                    execution_price = ask_price + 5.0
                else:  # SELL
                    execution_price = bid_price - 5.0
            else:  # "PASSIVE"
                # Limit order: place at best price without aggressive move
                if action.side == "BUY":
                    execution_price = ask_price
                else:  # SELL
                    execution_price = bid_price
            
            # Create Order object for agent
            from uuid import uuid4
            agent_order_id = f"AGENT_{uuid4().hex[:8]}"
            from .schema import Order
            agent_order = Order(
                order_id=agent_order_id,
                side=action.side,
                price=execution_price,
                quantity=action.shares_to_execute,
                timestamp=self.current_time,
                agent_id="LLM-AGENT"
            )
            
            # Submit to LOB and capture trades
            trades_executed = self.lob.add_order(agent_order)
            step_trades.extend(trades_executed)
            self.agent_trades.extend(trades_executed)
            
            # PHASE 3: UPDATE INVENTORY
            # Only count trades where LLM was the buyer/seller
            executed_shares = sum(
                t.quantity for t in trades_executed
                if (action.side == "BUY" and t.buyer_id == "LLM-AGENT") or
                   (action.side == "SELL" and t.seller_id == "LLM-AGENT")
            )
            self.inventory_remaining -= executed_shares
            self.inventory_remaining = max(0, self.inventory_remaining)
            
            # PHASE 4: CALCULATE REWARD (slippage penalty)
            # Slippage = how much worse than mid-price did agent execute?
            benchmark_price = current_mid_price  # Step-local benchmark
            for trade in step_trades:
                if action.side == "BUY":
                    # Paid more than mid is bad
                    slippage = trade.price - benchmark_price
                else:  # SELL
                    # Received less than mid is bad
                    slippage = benchmark_price - trade.price
                step_reward -= slippage * trade.quantity  # Negative = penalty
        
        # ========== TERMINATION LOGIC ==========
        done: bool = (
            self.inventory_remaining <= 0 or
            self.step_count >= self.max_steps
        )
        
        # Construct info dict
        info: Dict[str, Any] = {
            "step": self.step_count,
            "inventory_remaining": self.inventory_remaining,
            "shares_executed_this_step": sum(t.quantity for t in step_trades),
            "step_reward": step_reward,
        }
        
        # If episode is done, calculate final score
        if done:
            twap_benchmark = calculate_twap(
                self.episode_start_price,
                self.episode_end_price
            )
            final_score = calculate_score(
                self.agent_trades,
                self.target_shares,
                twap_benchmark
            )
            info["final_score"] = final_score
            info["twap_benchmark"] = twap_benchmark
            info["total_agent_trades"] = len(self.agent_trades)
            info["total_executed_shares"] = sum(t.quantity for t in self.agent_trades)
        
        next_state = self.state()
        reward = StepReward(total_reward=step_reward, done=done)
        
        return next_state, reward, done, info
    
    def state(self) -> AgentState:
        """
        Construct current agent state observation.
        
        Returns AgentState with:
          - time_remaining: steps until episode termination
          - inventory_remaining: shares left to execute
          - mid_price: (best_bid + best_ask) / 2
          - L2 data: top 3 levels of bids/asks with quantities
        
        Returns:
            AgentState observation for agent input to policy
        
        Complexity: O(1) if L2 is pre-computed, else O(log N)
        """
        # Fetch current market state
        bid_price, bid_qty, ask_price, ask_qty = self.lob.get_best_bid_ask()
        
        # Handle case where book is empty on one side
        if bid_price is None:
            bid_price = self.initial_mid_price - 1.0
        if ask_price is None:
            ask_price = self.initial_mid_price + 1.0
        
        current_mid_price = (bid_price + ask_price) / 2.0
        
        # Fetch L2 snapshot (top 3 levels)
        l2_state = self.lob.get_l2_state()
        
        # Calculate time remaining
        time_remaining = self.max_steps - self.step_count
        
        # Construct state object
        state = AgentState(
            time_remaining=time_remaining,
            inventory_remaining=self.inventory_remaining,
            mid_price=current_mid_price,
            bids=l2_state["bids"],      # Top 3 bid levels
            asks=l2_state["asks"]        # Top 3 ask levels
        )
        
        return state
