📈 QUANTIX: Institutional Limit Order Book Simulator

A production-grade Limit Order Book (LOB) simulator testing the ability of LLM Agents to execute massive block trades while minimizing market impact through High-Frequency microstructure strategies.

Built for the OpenEnv Hackathon, this project evaluates Generative AI not on chat capabilities, but on its ability to navigate adversarial financial physics in real-time.

🧠 The Core Problem: The Execution Dilemma

When an institutional trader needs to liquidate a $10M position, they cannot simply hit "Sell." They face a brutal triad of risks:

Timing Risk: Wait too long, and the market price moves against you.
Market Impact: Execute too aggressively, and you wipe out liquidity, suffering massive slippage.
Information Leakage: Show your hand, and high-frequency predators will front-run your orders.

The optimal strategy must balance passive quoting and aggressive sweeping. We measure this mathematically using Directional Implementation Shortfall (IS):

$$
\text{IS Score} = \max\left(0.0, 1.0 - \left( \frac{|\text{Agent VWAP} - \text{TWAP Benchmark}|}{\text{TWAP Benchmark}} \times 20 \right)\right)
$$

Agents must dynamically adapt to order book depth, time decay, and retail noise to beat the passive benchmark.

⚡ Architecture & Performance

1. The Physics: Amortized $O(1)$ Order Matching

The core engine is a hyper-optimized market microstructure environment built entirely in Python.

Self-Aware Pydantic Schemas: Native Price-Time Priority. Bids sort highest-to-lowest, Asks sort lowest-to-highest.
$O(\log N)$ Greedy Matching: Binary heap data structures enable sub-millisecond market sweeps and partial fills.
$O(1)$ Lazy Deletion: Order cancellations use tombstone markers, completely eliminating array-reconstruction memory leaks.

2. The Ecosystem: Realistic Market Dynamics

🟢 Market Maker: Provides continuous two-sided liquidity using a quote-and-update strategy with strict anti-fragmentation tick rounding.
🟡 Noise Trader: Injects realistic retail chop, crossing the spread to guarantee volume and push the mid-price randomly.
🤖 LLM Agent (Groq / HF / OpenAI): Observes the L2 order book, calculates pressure, and routes JSON-strict PASSIVE or AGGRESSIVE orders.

📂 Repository Structure

Plaintext

QUANTIX/
├── core_engine/              # The Order Book Physics Engine
│   ├── env.py                # LOBEnv: Step/Reset execution loop
│   ├── schema.py             # Pydantic strictly-typed market models
│   ├── matching_engine.py    # O(log N) Heaps & O(1) Lazy Deletion
│   ├── grader.py             # Directional IS & VWAP scoring
│   └── background_agents/    # Market Makers & Noise Traders
├── agentic_llm/              # The AI Brain
│   ├── client.py             # HF Router / Groq / OpenAI LLM wrapper
│   ├── prompts.py            # Microstructure strategy & L2 state formatter
│   └── logger.py             # Regex-compliant OpenEnv logging
├── dashboard/                # The UI
│   ├── app.py                # Streamlit institutional control panel
│   └── visualizers.py        # Plotly L2 depth & trajectory charts
└── tests/                    # 100% Coverage Test Suite
├── test_schema.py        # Mathematical proofs for Price-Time priority
├── test_matching.py      # Integration tests for partial fills/sweeps
└── test_inference.py     # End-to-end OpenEnv pipeline verification

🚀 Setup & Quickstart

Prerequisites

Python 3.10+
Groq, OpenAI, or Hugging Face API Key

1. Installation

Bash

git clone https://github.com/CodeR-6-9/QUANTIX.git
cd QUANTIX
pip install -r requirements.txt

2. Environment Variables

Create a .env file in the root directory:

Code snippet

GROQ_API_KEY="gsk_..."

# OPENAI_API_KEY="sk-..." (If using OpenAI instead)

MODEL_NAME="llama-3.3-70b-versatile"
API_BASE_URL="https://api.groq.com/openai/v1"

3. Run the Test Suite (Verification)

Ensure the physics engine and Pydantic schemas are mathematically sound:

Bash

pytest tests/ -v

4. Run the CLI Pipeline (OpenEnv Validation)

Execute the inference pipeline to generate regex-compliant [START], [STEP], and [END] tags for the competition autograder.

Bash

python inference.py

5. Launch the Institutional Dashboard

Boot up the Streamlit interface to visualize the LLM's trades, the L2 Order Book depth, and the VWAP vs. TWAP trajectory in real-time.

Bash

streamlit run dashboard/app.py

📊 Task Difficulties

| Level     | Goal        | Time Limit | Pressure | Challenge                                                  |
| --------- | ----------- | ---------- | -------- | ---------------------------------------------------------- |
| Easy 🟢   | 500 Shares  | 100 Ticks  | Low      | Execute cleanly with minimal market impact                 |
| Medium 🟡 | 1000 Shares | 20 Ticks   | High     | Balance severe time decay against spread crossing costs    |
| Hard 🔴   | 5000 Shares | 25 Ticks   | Extreme  | Survive adversarial conditions and massive inventory panic |

🤝 Contributing

To build your own agent, implement the decide_action(state: AgentState) → AgentAction interface. The LOBEnv handles the routing, matching, and grading. Pull requests for new background agents (e.g., Mean-Reversion or Momentum traders) are welcome.

🤝 The Team

Built with 💻 and ☕ by The Lost Tokens
Hridesh • Apoorva

Developed for the Meta PyTorch OpenEnv Hackathon
