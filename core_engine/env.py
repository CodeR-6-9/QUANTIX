"""
Limit Order Book environment implementing OpenAI Gym interface.

This module provides the main LOBEnv simulation environment where:
  1. The agent (LLM) observes market state (L2 order book, micro-price, inventory)
  2. The agent submits trading action (shares_to_execute, execution_style)
  3. Background agents (MarketMaker, NoiseTrader) provide realistic liquidity
  4. The environment executes the agent's order against the LOB
  5. The environment returns next state, reward (slippage), and done flag

Termination condition: Agent completes all target_shares or max_steps reached.
"""

import time
import math
from typing import Tuple, Dict, Any

from .schema import AgentState, AgentAction, StepReward, Trade
from .matching_engine import LimitOrderBook
from .background_agents.market_maker import MarketMaker
from .background_agents.noise_trader import NoiseTrader
from .grader import calculate_continuous_twap, calculate_score


class LOBEnv:
    """
    Limit Order Book Environment for Agent Training and Evaluation.
    
    OpenEnv-compliant interface where agents trade execution blocks by observing
    L2 market data and submitting limit/market orders. Background agents provide
    realistic liquidity. Episode ends when all target shares are executed or
    max_steps is reached.
    """
    
    def __init__(
        self,
        task_level: str = "medium",
        symbol: str = "AAPL",
        initial_mid_price: float = 150.0
    ) -> None:
        """
        Initialize LOBEnv with specified difficulty level.
        """
        self.symbol: str = symbol
        self.initial_mid_price: float = initial_mid_price
        self.current_time: float = 0.0
        
        # Task configuration based on difficulty
        task_config: Dict[str, Tuple[int, int]] = {
            "easy": (50, 100),
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
        self.inventory_remaining: int = self.target_shares
        self.step_count: int = 0
        self.agent_trades: list[Trade] = []
        
        # Statistics and tracking
        self.episode_start_price: float = initial_mid_price
        self.episode_end_price: float = initial_mid_price
        self.price_history: list[float] = []  # Tracks integral of market price

    def _calculate_micro_price(self, bid_price: float, bid_qty: int, ask_price: float, ask_qty: int) -> float:
        """
        Calculate Volume-Weighted Micro-Price based on Order Book Imbalance (OBI).
        This provides a highly predictive true price compared to the naive mid-price.
        """
        total_volume = bid_qty + ask_qty
        if total_volume > 0:
            imbalance = bid_qty / total_volume
            return (bid_price * (1.0 - imbalance)) + (ask_price * imbalance)
        else:
            # Fallback to naive mid if book is completely empty
            return (bid_price + ask_price) / 2.0

    def reset(self) -> AgentState:
        """Reset the environment for a new episode."""
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
        self.price_history = []
        
        # PRE-POPULATION: Call market maker a few times to build initial liquidity
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
        self.step_count += 1
        self.current_time = float(self.step_count)
        
        # ========== PHASE 1: BACKGROUND AGENTS ==========
        bid_price, bid_qty, ask_price, ask_qty = self.lob.get_best_bid_ask()
        
        if bid_price is None: bid_price, bid_qty = self.initial_mid_price - 1.0, 0
        if ask_price is None: ask_price, ask_qty = self.initial_mid_price + 1.0, 0
        
        # Determine momentum via Micro-Price
        current_micro_price = self._calculate_micro_price(bid_price, bid_qty, ask_price, ask_qty)
        
        # Market maker and noise trader update quotes around the Micro-Price
        mm_result = self.market_maker.step(current_micro_price, self.current_time)
        for order_id in mm_result["cancels"]: self.lob.cancel_order(order_id)
        for order in mm_result["new_orders"]: self.lob.add_order(order)
        
        nt_result = self.noise_trader.step(current_micro_price, self.current_time)
        for order_id in nt_result["cancels"]: self.lob.cancel_order(order_id)
        for order in nt_result["new_orders"]: self.lob.add_order(order)
        
        # Re-compute Micro-Price after background activity updates the LOB
        bid_price, bid_qty, ask_price, ask_qty = self.lob.get_best_bid_ask()
        if bid_price is None: bid_price, bid_qty = self.initial_mid_price - 1.0, 0
        if ask_price is None: ask_price, ask_qty = self.initial_mid_price + 1.0, 0
        
        current_micro_price = self._calculate_micro_price(bid_price, bid_qty, ask_price, ask_qty)
        self.price_history.append(current_micro_price)
        
        # ========== PHASE 2: LLM AGENT ACTION ==========
        step_trades: list[Trade] = []
        step_reward: float = 0.0
        
        # FAT FINGER PROTECTION: Bound the requested shares by remaining inventory
        actual_shares = min(action.shares_to_execute, self.inventory_remaining)
        
        if actual_shares > 0:
            # Safely calculate the spread (min 1 cent to avoid division/zero errors)
            spread = max(0.01, ask_price - bid_price)
            
            # Determine execution price based on style
            if action.execution_style == "AGGRESSIVE":
                # Almgren-Chriss Non-Linear Market Impact Model
                if action.side == "BUY":
                    impact = spread * math.sqrt(actual_shares / max(1.0, float(ask_qty)))
                    execution_price = round(ask_price + impact, 2)
                else:  # SELL
                    impact = spread * math.sqrt(actual_shares / max(1.0, float(bid_qty)))
                    execution_price = round(bid_price - impact, 2)
            else:  # "PASSIVE"
                # Join the queue at best price
                if action.side == "BUY":
                    execution_price = round(bid_price, 2)
                else:  # SELL
                    execution_price = round(ask_price, 2)
            
            # Create Order object for agent
            from uuid import uuid4
            agent_order_id = f"AGENT_{uuid4().hex[:5]}"
            from .schema import Order
            agent_order = Order(
                order_id=agent_order_id,
                side=action.side,
                price=execution_price,
                quantity=actual_shares,
                timestamp=self.current_time,
                agent_id="LLM-AGENT"
            )
            
            # Submit to LOB and capture trades
            trades_executed = self.lob.add_order(agent_order)
            step_trades.extend(trades_executed)
            self.agent_trades.extend(trades_executed)
            
            # PHASE 3: UPDATE INVENTORY
            executed_shares = sum(
                t.quantity for t in trades_executed
                if (action.side == "BUY" and t.buyer_id == "LLM-AGENT") or
                   (action.side == "SELL" and t.seller_id == "LLM-AGENT")
            )
            self.inventory_remaining -= executed_shares
            self.inventory_remaining = max(0, self.inventory_remaining)
            
            # PHASE 4: CALCULATE REWARD (Slippage vs. Micro-Price Benchmark)
            benchmark_price = current_micro_price 
            for trade in step_trades:
                if action.side == "BUY":
                    slippage = trade.price - benchmark_price
                else:  # SELL
                    slippage = benchmark_price - trade.price
                step_reward -= slippage * trade.quantity 
        
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
            self.episode_end_price = current_micro_price
            continuous_twap = calculate_continuous_twap(self.price_history)
            
            final_score = calculate_score(
                agent_trades=self.agent_trades,
                total_target_shares=self.target_shares,
                arrival_price=self.episode_start_price,
                continuous_twap=continuous_twap,
                steps_taken=self.step_count,
                max_steps=self.max_steps
            )
            
            info["final_score"] = final_score
            info["twap_benchmark"] = continuous_twap
            info["arrival_price"] = self.episode_start_price
            info["total_agent_trades"] = len(self.agent_trades)
            info["total_executed_shares"] = sum(t.quantity for t in self.agent_trades)
        
        next_state = self.state()
        reward = StepReward(total_reward=step_reward, done=done)
        
        return next_state, reward, done, info
    
    def state(self) -> AgentState:
        """Construct current agent state observation."""
        bid_price, bid_qty, ask_price, ask_qty = self.lob.get_best_bid_ask()
        
        if bid_price is None: bid_price, bid_qty = self.initial_mid_price - 1.0, 0
        if ask_price is None: ask_price, ask_qty = self.initial_mid_price + 1.0, 0
        
        current_micro_price = self._calculate_micro_price(bid_price, bid_qty, ask_price, ask_qty)
        
        l2_state = self.lob.get_l2_state()
        time_remaining = self.max_steps - self.step_count
        
        state = AgentState(
            time_remaining=time_remaining,
            inventory_remaining=self.inventory_remaining,
            mid_price=current_micro_price, # State now feeds true Micro-Price!
            bids=l2_state["bids"],
            asks=l2_state["asks"]
        )
        
        return state