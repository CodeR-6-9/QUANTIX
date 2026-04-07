"""
Streamlit dashboard for OpenEnv LOB Simulator.

Synchronized with the Groq/HF Inference Router and the O(1) Matching Engine.
Visualizes Directional Implementation Shortfall and exact Order Book physics.
"""

import streamlit as st
from typing import Dict, Any, List

from core_engine.env import LOBEnv
from core_engine.schema import AgentState, AgentAction
from agentic_llm.client import LLMTrader
from dashboard.visualizers import plot_order_book_depth, plot_execution_trajectory

def main() -> None:
    # Page configuration
    st.set_page_config(
        page_title="QUANTIX LOB Execution",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 QUANTIX: Institutional Trade Execution")
    st.markdown(
        """
        **Minimize Directional Implementation Shortfall through Intelligent Execution**
        Watch your LLM agent optimize block trade execution in a High-Frequency Limit Order Book.
        """
    )
    
    # ========================================================================
    # SIDEBAR: Configuration and Controls
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown(
            """
            **Directional IS**: The agent is graded on beating the TWAP benchmark. 
            **Physics**: Strict Price-Time priority with Passive/Aggressive queue mechanics.
            """
        )
        st.divider()
        
        # Updated to reflect our new architecture
        api_key = st.text_input(
            "API Key (Groq / HF Token)",
            type="password",
            help="Required for inference."
        )
        
        api_base_url = st.text_input(
            "API Base URL",
            value="https://api.groq.com/openai/v1",
            help="Defaults to Groq. Change to https://router.huggingface.co/v1 for HF."
        )
        
        task_level = st.selectbox(
            "Task Difficulty Level",
            options=["easy", "medium", "hard"],
            format_func=lambda x: x.capitalize()
        )
        
        model_name = st.selectbox(
            "LLM Model",
            options=["llama-3.3-70b-versatile", "Qwen/Qwen2.5-72B-Instruct"],
            index=0
        )
        
        st.divider()
        run_button = st.button("🚀 Run Simulation", use_container_width=True, type="primary")
    
    # ========================================================================
    # MAIN CONTENT: Simulation and Results
    # ========================================================================
    if run_button:
        if not api_key:
            st.warning("⚠️ Please provide an API Key to run the simulation.", icon="🔑")
            st.stop()
        
        with st.spinner("🔄 Booting O(1) Matching Engine & Connecting to Router..."):
            try:
                env = LOBEnv(task_level=task_level)
                trader = LLMTrader(
                    api_key=api_key,
                    model_name=model_name,
                    api_base_url=api_base_url
                )
                state: AgentState = env.reset()
            except Exception as e:
                st.error(f"❌ Initialization Error: {str(e)}", icon="⚠️")
                st.stop()
        
        # State tracking
        done = False
        step_count = 0
        final_info: Dict[str, Any] = {}
        price_history: List[float] = [state.mid_price]
        
        # UI Containers
        status_container = st.container(border=True)
        progress_bar = st.progress(0, text="Executing Phase...")
        
        # Simulation Loop
        with st.spinner("⏳ Simulating High-Frequency Microstructure..."):
            while not done:
                try:
                    action: AgentAction = trader.decide_action(state)
                except Exception as e:
                    st.error(f"❌ LLM Error at step {step_count}: {str(e)}", icon="⚠️")
                    st.stop()
                
                state, reward, done, info = env.step(action)
                step_count += 1
                final_info = info
                price_history.append(state.mid_price)
                
                # Real-time UI updates
                if step_count % 5 == 0 or done:
                    with status_container:
                        st.markdown("### 📍 Live Execution Status")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("⏱️ Time Left", f"{state.time_remaining} ticks")
                        col2.metric("📦 Inventory", f"{state.inventory_remaining} shrs")
                        col3.metric("💹 Mid-Price", f"${state.mid_price:.2f}")
                        col4.metric("📊 Action", f"{action.side} {action.execution_style}")
                    
                    max_steps = getattr(env, "max_steps", 60)
                    progress = min(step_count / max_steps, 1.0)
                    progress_bar.progress(progress, text=f"Step {step_count} completed...")
        
        st.success("✅ Execution Complete!", icon="✅")
        
        # ====================================================================
        # FINAL RESULTS SECTION
        # ====================================================================
        st.divider()
        st.header("📈 Institutional Execution Report")
        
        # Extract correct final metrics from updated env
        final_score = final_info.get("final_score", 0.0)
        twap_benchmark = final_info.get("twap_benchmark", env.initial_mid_price)
        total_executed = final_info.get("total_executed_shares", 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Directional IS Score", f"{final_score:.4f}", help="1.0 = Perfect. 0.0 = Failed.")
        col2.metric("📦 Total Executed", f"{total_executed} / {env.target_shares}", help="Must finish all shares to score.")
        col3.metric("📌 TWAP Benchmark", f"${twap_benchmark:.2f}")
        
        st.divider()
        st.header("📊 Microstructure Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Final Order Book Depth (L2)")
            if state.bids and state.asks:
                ob_fig = plot_order_book_depth(
                    bids=state.bids,
                    asks=state.asks,
                    title="Terminal LOB State"
                )
                st.plotly_chart(ob_fig, use_container_width=True)
            else:
                st.info("No order book data available")
                
        with col2:
            st.subheader("Execution Trajectory")
            if price_history and env.agent_trades:
                # We now pass the EXACT trade objects from the physics engine
                traj_fig = plot_execution_trajectory(
                    prices=price_history,
                    agent_trades=env.agent_trades, 
                    title="Actual Fills vs. TWAP"
                )
                st.plotly_chart(traj_fig, use_container_width=True)
            else:
                st.info("No trades were executed.")

    else:
        st.info("👈 Configure your routing limits and click **🚀 Run Simulation** to begin.", icon="ℹ️")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"\n❌ UI Thread Error: {str(e)}", icon="⚠️")
        st.stop()