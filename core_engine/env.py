"""
Limit Order Book environment implementing OpenAI Gym interface.

This module provides the main LOBEnv simulation environment where:
  1. The agent (LLM) observes market state (L2 order book, mid-price, inventory)
  2. The agent submits trading action (shares_to_execute, execution_style)
  3. Background agents (MarketMaker, NoiseTrader, ToxicTrader) provide realistic liquidity
  4. The environment executes the agent's order against the LOB
  5. The environment returns next state, reward (slippage), and done flag
"""

import time
from typing import Tuple, Dict, Any

from .schema import AgentState, AgentAction, StepReward, Trade, Order
from .matching_engine import LimitOrderBook
from .background_agents.market_maker import MarketMaker
from .background_agents.noise_trader import NoiseTrader
from .background_agents.toxic_trader import ToxicTrader
from .grader import calculate_score


class LOBEnv:
    """
    Limit Order Book Environment for Agent Training and Evaluation.
    """
    
    def __init__(
        self,
        task_level: str = "medium",
        symbol: str = "AAPL",
        initial_mid_price: float = 150.0
    ) -> None:
        self.symbol: str = symbol
        self.initial_mid_price: float = initial_mid_price
        self.current_time: float = 0.0
        self.task_level = task_level
        self.order_counter = 0  
        
        # Task configuration based on difficulty
        task_config: Dict[str, Tuple[int, int]] = {
            "easy": (50, 100),          
            "medium": (1000, 20),
            "hard": (5000, 25)
        }
        if task_level not in task_config:
            raise ValueError(f"Invalid task_level: {task_level}. Choose from {list(task_config.keys())}")
        
        self.target_shares, self.max_steps = task_config[task_level]
        
        # Dynamic Compliance Mandates to force NLP reasoning
        mandate_config = {
            "easy": "MANDATE: Standard execution. Focus on minimizing slippage against the arrival price.",
            "medium": "MANDATE: Information leakage risk. You are strictly FORBIDDEN from using more than 5 AGGRESSIVE orders this episode.",
            "hard": "MANDATE: Market is fragile. You are strictly FORBIDDEN from using AGGRESSIVE orders if the spread is wider than $0.50."
        }
        self.mandate = mandate_config[task_level]
        self.aggressive_order_count = 0
        
        # Order book and background agents
        self.lob: LimitOrderBook = LimitOrderBook(symbol)
        self.market_maker: MarketMaker = MarketMaker(
            agent_id="MM-SIM", num_levels=3, spread_width=0.5, order_size=100
        )
        self.noise_trader: NoiseTrader = NoiseTrader(
            agent_id="NT-SIM", trade_probability=0.3, base_order_size=50
        )
        
        # ==========================================
        # STEP 2: Inject the Predator only on Hard Mode
        # ==========================================
        if task_level == "hard":
            self.toxic_trader = ToxicTrader(agent_id="TOXIC-HFT")
        else:
            self.toxic_trader = None
        
        # Agent execution state
        self.inventory_remaining: int = self.target_shares  
        self.step_count: int = 0                            
        self.agent_trades: list[Trade] = []                
        
        # Advanced Quant Statistics
        self.episode_start_price: float = initial_mid_price
        self.cumulative_mid_price: float = initial_mid_price
        self.twap_ticks: int = 1
    
    def reset(self) -> AgentState:
        """Reset the environment for a new episode."""
        self.lob = LimitOrderBook(self.symbol)
        self.market_maker = MarketMaker("MM-SIM", num_levels=3, spread_width=0.5, order_size=100)
        self.noise_trader = NoiseTrader("NT-SIM", trade_probability=0.3, base_order_size=50)
        
        if self.task_level == "hard":
            self.toxic_trader = ToxicTrader(agent_id="TOXIC-HFT")
            
        self.step_count = 0
        self.current_time = 0.0
        self.inventory_remaining = self.target_shares
        self.agent_trades = []
        self.aggressive_order_count = 0
        
        self.episode_start_price = self.initial_mid_price
        self.cumulative_mid_price = self.initial_mid_price
        self.twap_ticks = 1
        self.order_counter = 0
        
        # PRE-POPULATION
        for _ in range(3):
            mm_result = self.market_maker.step(self.initial_mid_price, self.current_time)
            for order in mm_result["new_orders"]:
                self.lob.add_order(order)
            self.current_time += 0.1
        
        return self.state()

    def _get_micro_price(self) -> Tuple[float, float, float, int, int]:
        """Helper to calculate the Volume-Weighted Micro-Price."""
        bid_price, bid_qty, ask_price, ask_qty = self.lob.get_best_bid_ask()
        
        if bid_price is None: bid_price = self.initial_mid_price - 1.0
        if ask_price is None: ask_price = self.initial_mid_price + 1.0
        
        total_liquidity = bid_qty + ask_qty
        if total_liquidity > 0:
            imbalance_ratio = bid_qty / total_liquidity
            micro_price = (bid_price * (1.0 - imbalance_ratio)) + (ask_price * imbalance_ratio)
        else:
            micro_price = (bid_price + ask_price) / 2.0
            
        return round(micro_price, 4), bid_price, ask_price, bid_qty, ask_qty
    
    def step(self, action: AgentAction) -> Tuple[AgentState, StepReward, bool, Dict[str, Any]]:
        self.step_count += 1
        self.current_time = float(self.step_count)
        
        # ========== PHASE 1: BACKGROUND AGENTS ==========
        current_micro_price, bid_price, ask_price, bid_qty, ask_qty = self._get_micro_price()
        
        mm_result = self.market_maker.step(current_micro_price, self.current_time)
        for order_id in mm_result["cancels"]:
            self.lob.cancel_order(order_id)
        for order in mm_result["new_orders"]:
            self.lob.add_order(order)
        
        nt_result = self.noise_trader.step(current_micro_price, self.current_time)
        for order_id in nt_result["cancels"]:
            self.lob.cancel_order(order_id)
        for order in nt_result["new_orders"]:
            self.lob.add_order(order)
            
        # ==========================================
        # STEP 3: TOXIC HFT PREDATOR (Hard Mode)
        # ==========================================
        if self.toxic_trader is not None:
            current_lob_state = self.lob.get_l2_state()
            toxic_actions = self.toxic_trader.step(
                current_micro_price=current_micro_price, 
                current_time=self.current_time,
                lob_state=current_lob_state
            )
            
            # Lazy deletion for fast cancellation
            for order_id in toxic_actions.get("cancels", []):
                if order_id in self.lob.active_orders:
                    self.lob.active_orders[order_id].quantity = 0 
                    
            for bad_order in toxic_actions.get("new_orders", []):
                self.lob.add_order(bad_order)
        
        # Recalculate state before LLM acts because HFT bot manipulated it
        current_micro_price, bid_price, ask_price, bid_qty, ask_qty = self._get_micro_price()
        current_spread = round(ask_price - bid_price, 2)
        
        # Accumulate true mathematical TWAP
        self.cumulative_mid_price += current_micro_price
        self.twap_ticks += 1
        
        # ========== PHASE 2: LLM AGENT ACTION ==========
        step_trades: list[Trade] = []  
        step_reward: float = 0.0       
        compliance_penalty: float = 0.0
        
        actual_shares = min(action.shares_to_execute, self.inventory_remaining)
        
        if actual_shares > 0:
            if action.execution_style == "AGGRESSIVE":
                self.aggressive_order_count += 1
                
                # Rule Violations (Dense Penalty)
                if "FORBIDDEN from using more than 5 AGGRESSIVE" in self.mandate and self.aggressive_order_count > 5:
                    compliance_penalty -= 500.0  
                if "FORBIDDEN from using AGGRESSIVE orders if the spread is wider" in self.mandate and current_spread > 0.50:
                    compliance_penalty -= 500.0  
                
                # GOD-LEVEL MATH: Almgren-Chriss Square Root Impact Model
                available_liquidity = max(1, ask_qty if action.side == "BUY" else bid_qty)
                impact_coefficient = 0.75 
                volume_ratio = actual_shares / available_liquidity
                slippage_penalty = max(0.01, current_spread) * impact_coefficient * (volume_ratio ** 0.5)
                
                if action.side == "BUY":
                    execution_price = round(ask_price + slippage_penalty, 2)
                else:  
                    execution_price = round(bid_price - slippage_penalty, 2)
            else:  
                # PASSIVE placement
                if action.side == "BUY":
                    execution_price = round(bid_price, 2)
                else:  
                    execution_price = round(ask_price, 2)
            
            # Fast Pydantic Instantiation
            self.order_counter += 1
            agent_order_id = f"A_{self.order_counter}"
            
            agent_order = Order.model_construct(
                order_id=agent_order_id,
                side=action.side,
                price=execution_price,
                quantity=actual_shares, 
                timestamp=self.current_time,
                agent_id="LLM-AGENT"
            )
            
            trades_executed = self.lob.add_order(agent_order)
            step_trades.extend(trades_executed)
            self.agent_trades.extend(trades_executed)
            
            executed_shares = sum(
                t.quantity for t in trades_executed
                if (action.side == "BUY" and t.buyer_id == "LLM-AGENT") or
                   (action.side == "SELL" and t.seller_id == "LLM-AGENT")
            )
            self.inventory_remaining -= executed_shares
            self.inventory_remaining = max(0, self.inventory_remaining)
            
            # Intermediate Slippage tracking against local micro-price
            for trade in step_trades:
                if action.side == "BUY":
                    slippage = trade.price - current_micro_price
                else:  
                    slippage = current_micro_price - trade.price
                step_reward -= slippage * trade.quantity  
                
        # Add compliance penalty to step reward
        step_reward += compliance_penalty
        
        # ========== TERMINATION LOGIC ==========
        done: bool = (
            self.inventory_remaining <= 0 or
            self.step_count >= self.max_steps
        )
        
        # Perfectly safe intermediate score for OpenEnv
        info: Dict[str, Any] = {
            "step": self.step_count,
            "inventory_remaining": self.inventory_remaining,
            "shares_executed_this_step": sum(t.quantity for t in step_trades),
            "step_reward": step_reward,
            "score": 0.500,  
            "task_scores": {"execution": 0.500} 
        }
        
        if done:
            # True Continuous Integral TWAP
            true_twap = round(self.cumulative_mid_price / max(1, self.twap_ticks), 4)
            
            raw_score = calculate_score(
                agent_trades=self.agent_trades,
                total_target_shares=self.target_shares,
                arrival_price=self.episode_start_price, 
                true_twap=true_twap,                    
                max_steps=self.max_steps,
                steps_taken=self.step_count
            )
            
            # Absolute mathematical lock at the deepest level
            safe_score = max(0.001, min(0.999, float(raw_score)))
            
            info["final_score"] = safe_score
            info["twap_benchmark"] = true_twap
            info["total_agent_trades"] = len(self.agent_trades)
            info["total_executed_shares"] = sum(t.quantity for t in self.agent_trades)
            
            # OPENENV STRICT REQUIREMENTS
            info["score"] = safe_score
            info["task_scores"] = {
                "execution": safe_score,
                "default": safe_score
            }
        
        next_state = self.state()
        reward = StepReward(total_reward=step_reward, done=done)
        
        return next_state, reward, done, info
    
    def state(self) -> AgentState:
        """Construct current agent state observation with Semantic hybrid data."""
        current_micro_price, bid_price, ask_price, bid_qty, ask_qty = self._get_micro_price()
        l2_state = self.lob.get_l2_state()
        time_remaining = self.max_steps - self.step_count
        
        # Generate Semantic "Market Shape" for the LLM
        total_bid_qty = sum(qty for price, qty in l2_state.get("bids", []))
        total_ask_qty = sum(qty for price, qty in l2_state.get("asks", []))
        
        if total_bid_qty > total_ask_qty * 3:
            shape_str = "HEAVY_BID_SKEW: Massive buy wall detected. Upward momentum likely."
        elif total_ask_qty > total_bid_qty * 3:
            shape_str = "HEAVY_ASK_SKEW: Massive sell wall detected. Downward pressure likely."
        elif total_bid_qty == 0 and total_ask_qty == 0:
            shape_str = "LIQUIDITY_VACUUM: Order book is dangerously thin."
        else:
            shape_str = "BALANCED: Liquidity is evenly distributed across both sides."
        
        state = AgentState.model_construct(
            time_remaining=time_remaining,
            inventory_remaining=self.inventory_remaining,
            mid_price=current_micro_price,
            bids=l2_state.get("bids", []),      
            asks=l2_state.get("asks", []),
            market_shape=shape_str,
            compliance_mandate=self.mandate
        )
        
        return state