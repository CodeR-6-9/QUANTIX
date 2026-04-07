# 📈 QUANTIX — Institutional Limit Order Book Simulator

A production-grade **Limit Order Book (LOB) simulator** designed to evaluate how LLM agents execute large block trades while minimizing market impact using high-frequency market microstructure strategies.

Built for the **OpenEnv Hackathon**, QUANTIX shifts evaluation from conversational ability to **real-time decision-making under adversarial financial conditions**.

---

## 🧠 Core Problem — The Execution Dilemma

Institutional trade execution is constrained by three competing risks:

* **Timing Risk** — Delayed execution exposes trades to adverse price movement
* **Market Impact** — Aggressive execution consumes liquidity and increases slippage
* **Information Leakage** — Revealing intent invites front-running by high-frequency traders

The optimal strategy must dynamically balance **passive liquidity provision** and **aggressive execution**.

### 📊 Evaluation Metric — Directional Implementation Shortfall (IS)

$$
\text{IS Score} = \max\left(0.0,; 1.0 - \left(\frac{|\text{Agent VWAP} - \text{TWAP Benchmark}|}{\text{TWAP Benchmark}} \times 20\right)\right)
$$

Agents are rewarded for outperforming a TWAP benchmark while adapting to:

* Order book depth
* Time decay
* Stochastic retail order flow

---

## ⚡ Architecture & Performance

### 1. Matching Engine — Market Microstructure Core

* **Amortized $O(1)$ operations** with optimized data handling
* **$O(\log N)$ matching** using binary heaps
* **$O(1)$ lazy deletion** via tombstone markers (no memory rebuilds)

#### Key Features

* Price-time priority enforced via **Pydantic schemas**
* Efficient partial fills and sweeping
* Sub-millisecond execution performance

---

### 2. Market Ecosystem — Realistic Simulation

* 🟢 **Market Maker**

  * Provides continuous two-sided liquidity
  * Uses spread-aware quoting with tick rounding

* 🟡 **Noise Trader**

  * Simulates retail activity
  * Randomly crosses spreads to generate volume

* 🤖 **LLM Agent (Groq / OpenAI / Hugging Face)**

  * Observes L2 order book state
  * Outputs structured **PASSIVE / AGGRESSIVE** actions

---

## 📂 Repository Structure

```
QUANTIX/
├── core_engine/              # Order book engine
│   ├── env.py                # Environment loop (step/reset)
│   ├── schema.py             # Typed market models (Pydantic)
│   ├── matching_engine.py    # Matching logic (heap-based)
│   ├── grader.py             # IS + VWAP scoring
│   └── background_agents/    # Market simulation agents
│
├── agentic_llm/              # LLM integration
│   ├── client.py             # Model routing (Groq/OpenAI/HF)
│   ├── prompts.py            # Strategy + state formatting
│   └── logger.py             # OpenEnv-compliant logging
│
├── dashboard/                # Visualization layer
│   ├── app.py                # Streamlit dashboard
│   └── visualizers.py        # Plotly charts
│
└── tests/                    # Test suite (100% coverage)
    ├── test_schema.py
    ├── test_matching.py
    └── test_inference.py
```

---

## 🚀 Setup & Quickstart

### Prerequisites

* Python **3.10+**
* API key for one of:

  * Groq
  * OpenAI
  * Hugging Face

---

### 1. Installation

```bash
git clone https://github.com/CodeR-6-9/QUANTIX.git
cd QUANTIX
pip install -r requirements.txt
```

---

### 2. Environment Configuration

Create a `.env` file:

```env
GROQ_API_KEY="gsk_..."

# Optional
# OPENAI_API_KEY="sk-..."

MODEL_NAME="llama-3.3-70b-versatile"
API_BASE_URL="https://api.groq.com/openai/v1"
```

---

### 3. Run Tests (Validation)

```bash
pytest tests/ -v
```

---

### 4. Run Inference Pipeline

Generates OpenEnv-compatible logs:

```bash
python inference.py
```

---

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Task Difficulty Levels

| Level     | Target Size | Time Limit | Market Pressure | Description                      |
| --------- | ----------- | ---------- | --------------- | -------------------------------- |
| 🟢 Easy   | 500 shares  | 100 ticks  | Low             | Minimal impact execution         |
| 🟡 Medium | 1000 shares | 20 ticks   | High            | Tradeoff: time vs cost           |
| 🔴 Hard   | 5000 shares | 25 ticks   | Extreme         | Adversarial liquidity conditions |

---

## 🤝 Contributing

To implement a custom agent:

```python
def decide_action(state: AgentState) -> AgentAction:
    ...
```

* The environment handles execution, routing, and scoring
* Contributions welcome:

  * New trading strategies (momentum, mean-reversion, etc.)
  * Improved execution policies
  * Additional background agents

---

## 👥 Team

**The Lost Tokens**

* Hridesh
* Apoorva

Built with 💻 and ☕ for the **Meta PyTorch OpenEnv Hackathon**
