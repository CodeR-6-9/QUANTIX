"""
Streamlit dashboard for OpenEnv LOB Simulator.

Professional institutional trading desk interface for visualizing and
monitoring LLM agent performance in limit order book execution tasks.
"""

import streamlit as st
from typing import Dict, Any, Optional, List
import os

from core_engine.env import LOBEnv
from core_engine.schema import AgentState, AgentAction
from agentic_llm.client import LLMTrader
from dashboard.visualizers import plot_order_book_depth, plot_execution_trajectory


def main() -> None:
    """
    Main Streamlit dashboard application.
    
    Features:
    - Professional institutional interface
    - Real-time simulation updates
    - Order book and execution trajectory visualization
    - Final metrics display
    """
    
    # Page configuration
    st.set_page_config(
        page_title="OpenEnv: Institutional Trade Execution",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("📊 OpenEnv: Institutional Trade Execution")
    st.markdown(
        """
        **Minimize Implementation Shortfall through Intelligent Execution**
        
        Watch an LLM agent optimize block trade execution in a realistic limit order book 
        environment with market microstructure dynamics.
        """
    )
    
    # ========================================================================
    # SIDEBAR: Configuration and Controls
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration & Controls")
        
        st.subheader("About This Task")
        st.markdown(
            """
            **Implementation Shortfall**: The difference between the execution price 
            and a passive (TWAP) benchmark.
            
            **Price-Time Priority**: Market orders match against standing 
            limit orders by price first, then time.
            
            **Objective**: Execute a large block of shares to minimize IS 
            while respecting market microstructure constraints.
            """
        )
        
        st.divider()
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for LLM-based decision making"
        )
        
        # Task level selector
        task_level = st.selectbox(
            "Task Difficulty Level",
            options=["easy", "medium", "hard"],
            format_func=lambda x: x.capitalize(),
            help="Varies inventory size, volatility, and time pressure"
        )
        
        # Model selection
        model_name = st.selectbox(
            "LLM Model",
            options=["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Choose your OpenAI model"
        )
        
        st.divider()
        
        # Run Simulation button
        run_button = st.button(
            "🚀 Run Simulation",
            use_container_width=True,
            type="primary"
        )
    
    # ========================================================================
    # MAIN CONTENT: Simulation and Results
    # ========================================================================
    
    if run_button:
        # Validate API key
        if not api_key:
            st.warning(
                "⚠️ Please provide an OpenAI API Key to run the simulation.",
                icon="🔑"
            )
            st.stop()
        
        # Show loading spinner
        with st.spinner("🔄 Initializing environment and agent..."):
            try:
                # Initialize environment and agent
                env = LOBEnv(task_level=task_level)
                trader = LLMTrader(
                    api_key=api_key,
                    model_name=model_name,
                    api_base_url="https://api.openai.com/v1"
                )
                
                # Reset environment
                state: AgentState = env.reset()
                
            except Exception as e:
                st.error(f"❌ Initialization Error: {str(e)}", icon="⚠️")
                st.stop()
        
        # Progress tracking
        done = False
        step_count = 0
        final_info: Dict[str, Any] = {}
        
        # Storage for visualization data
        price_history: List[float] = []
        trades_history: List[Dict] = []
        
        # Create placeholders for real-time updates
        status_container = st.container(border=True)
        metrics_container = st.container(border=True)
        progress_bar = st.progress(0, text="Initializing...")
        
        # Main simulation loop
        with st.spinner("⏳ Running simulation..."):
            while not done:
                # Get action from LLM trader
                try:
                    action: AgentAction = trader.decide_action(state)
                except Exception as e:
                    st.error(f"❌ LLM Error at step {step_count}: {str(e)}", icon="⚠️")
                    st.stop()
                
                # Step environment
                state, reward, done, info = env.step(action)
                step_count += 1
                final_info = info
                
                # Update history for visualizations
                price_history.append(state.mid_price)
                trades_history.append({
                    "step": step_count,
                    "side": action.side,
                    "price": state.mid_price,
                    "quantity": action.shares_to_execute
                })
                
                # Update real-time metrics every ~10 steps or at end
                if step_count % 10 == 0 or done:
                    with status_container:
                        st.markdown("### 📍 Current Status")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "⏱️ Time Remaining",
                                f"{state.time_remaining} steps",
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                "📦 Inventory Remaining",
                                f"{state.inventory_remaining} shares",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "💹 Current Mid-Price",
                                f"${state.mid_price:.2f}",
                                delta=None
                            )
                        
                        with col4:
                            st.metric(
                                "Step",
                                f"{step_count}",
                                delta=None
                            )
                    
                    # Update progress
                    max_steps = getattr(env, "max_steps", 60)
                    progress = min(step_count / max_steps, 1.0)
                    progress_bar.progress(progress, text=f"Step {step_count}...")
        
        # Simulation complete - show results
        st.success("✅ Simulation Complete!", icon="✅")
        
        # ====================================================================
        # FINAL RESULTS SECTION
        # ====================================================================
        
        st.divider()
        st.heading("📈 Final Results")
        
        # Extract final metrics
        final_score = final_info.get("score", 0.0)
        final_is = final_info.get("implementation_shortfall", 0.0)
        twap_benchmark = final_info.get("twap_benchmark", state.mid_price)  # Fallback
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Final Score",
                f"{final_score:.4f}",
                delta=None,
                help="1.0 = perfect execution, 0.0 = complete failure"
            )
        
        with col2:
            st.metric(
                "💰 Implementation Shortfall",
                f"${final_is:.2f}",
                delta=None,
                help="Difference from TWAP benchmark"
            )
        
        with col3:
            st.metric(
                "📌 TWAP Benchmark",
                f"${twap_benchmark:.2f}",
                delta=None,
                help="Time-weighted average price used for comparison"
            )
        
        st.divider()
        
        # Display visualizations
        st.heading("📊 Visualizations")
        
        col1, col2 = st.columns(2)
        
        # Order Book Depth (final state)
        with col1:
            st.subheader("Final Order Book Depth (L2)")
            
            if state.bids and state.asks:
                ob_fig = plot_order_book_depth(
                    bids=state.bids,
                    asks=state.asks,
                    title="Final L2 Order Book State"
                )
                st.plotly_chart(ob_fig, use_container_width=True)
            else:
                st.info("No order book data available")
        
        # Execution Trajectory
        with col2:
            st.subheader("Execution Trajectory vs. Mid-Price")
            
            if price_history and trades_history:
                traj_fig = plot_execution_trajectory(
                    prices=price_history,
                    agent_trades=trades_history,
                    title="Agent Execution vs. Mid-Price Benchmark"
                )
                st.plotly_chart(traj_fig, use_container_width=True)
            else:
                st.info("No execution data available")
        
        # Summary statistics
        st.divider()
        st.heading("📋 Execution Summary")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            total_executed = sum(t.get("quantity", 0) for t in trades_history)
            st.metric(
                "Total Shares Executed",
                f"{int(total_executed)}",
                help="Total shares traded during episode"
            )
        
        with summary_col2:
            num_trades = len(trades_history)
            st.metric(
                "Number of Trades",
                f"{num_trades}",
                help="Total trading actions taken"
            )
        
        with summary_col3:
            avg_price = sum(price_history) / len(price_history) if price_history else 0
            st.metric(
                "Avg Mid-Price During Episode",
                f"${avg_price:.2f}",
                help="Time-weighted average mid-price"
            )
        
        with summary_col4:
            buys = sum(1 for t in trades_history if t.get("side") == "BUY")
            sells = sum(1 for t in trades_history if t.get("side") == "SELL")
            st.metric(
                "Buy/Sell Ratio",
                f"{buys}/{sells}",
                help="Number of buy vs. sell actions"
            )
    
    else:
        # Initial welcome state
        st.info(
            "👈 Use the sidebar to configure your simulation and click "
            "**🚀 Run Simulation** to begin.",
            icon="ℹ️"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                ### 🎯 How It Works
                
                1. **Configure**: Select task difficulty and LLM model
                2. **Authenticate**: Provide your OpenAI API key
                3. **Execute**: Watch the agent optimize execution in real-time
                4. **Analyze**: Review final score and execution trajectory
                """
            )
        
        with col2:
            st.markdown(
                """
                ### 📚 Key Metrics
                
                - **Implementation Shortfall (IS)**
                - **Score**: (1.0 - IS/Benchmark)
                - **TWAP**: Time-Weighted Average Price
                - **Execution Price**: Actual fills at LOB
                """
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"\n❌ Dashboard Error: {str(e)}", icon="⚠️")
        st.stop()
