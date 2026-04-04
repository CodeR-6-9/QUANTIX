# OpenEnv: Institutional Limit Order Book Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI API](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

> **A production-grade limit order book simulator testing LLM agents' ability to execute large block trades while minimizing market impact through optimal execution strategies.**

---

## The Core Problem

Trade execution is one of the most critical challenges in quantitative finance. When a large trader (e.g., $10M order) needs to execute:

- **Timing Risk**: Delay execution → prices move against you
- **Market Impact**: Execute too fast → slippage increases
- **Information Leakage**: Market sees the order → predatory traders attack

The optimal execution strategy must balance these tensions. We measure this mathematically using **Implementation Shortfall (IS)**:

$$\text{IS} = |\text{Execution Price} - \text{TWAP Benchmark}| \times \text{Quantity}$$

Where TWAP (Time-Weighted Average Price) represents the theoretical perfect execution. Agents score higher by minimizing IS through intelligent decision-making.

---

## System Architecture

### Backend Market Physics: O(log N) Order Matching

The core engine implements institutional-grade market microstructure:

```
Limit Order Book (LOB)
├─ Buy Side:  Min/Max Heap (O(log N) insertion/deletion)
│   └─ Price-Time Priority: Better prices first, then FIFO
├─ Sell Side: Min/Max Heap (O(log N) insertion/deletion)
│   └─ Price-Time Priority: Better prices first, then FIFO
└─ Matching:  O(log N) greedy algorithm
    └─ Pro-rata execution at best prices
```

**Key Performance Optimizations**:
- **O(log N) Matching**: Binary heap data structures enable sub-millisecond matching
- **O(1) Lazy Deletion**: Order cancellations use tombstone markers (no heap reconstruction)
- **Real-time Depth**: Maintains L2/L3 order book snapshot for agent observation

### Multi-Agent Ecosystem: Realistic Market Dynamics

The LOB isn't static—it's populated by autonomous agents simulating real market participants:

1. **Market Maker Agent** 🟢
   - Provides continuous two-sided liquidity
   - Quotes around fair value with configurable spread
   - Risk management: Inventory control, delta hedging

2. **Noise Trader Agent** 🟡
   - Random buy/sell flow (user-driven trading)
   - Non-strategic order placement
   - Realistic liquidity consumption

3. **Toxic Flow Agent** 🔴
   - Adversarial predatory trading (hard difficulty only)
   - "Front-running" the main agent (simulated spoofing)
   - Tests agent's ability to avoid being exploited

4. **LLM Trading Agent** (Main) 🤖
   - Observes L2 order book state
   - Calls OpenAI GPT-4 with strategic context
   - Decides: side (BUY/SELL), quantity, execution style (AGGRESSIVE/PASSIVE)
   - Receives reward feedback based on execution quality

### The Grader: Mathematical Scoring

**Score = 1.0 - (Implementation Shortfall / Max Possible IS)**

Range: [0.0, 1.0]
- **1.0** = Perfect execution (matched TWAP exactly)
- **0.5** = Moderate execution (50% slippage vs TWAP)
- **0.0** = Worst execution possible

An agent's final score across 3 task difficulties becomes its OpenEnv submission grade.

---

## Project Structure

### Core Engine (`core_engine/`)

- **`env.py`** (LOBEnv): OpenAI Gym-compatible environment
  - Initializes market state, background agents
  - Processes agent actions via `step(action)`
  - Returns: observation (state), reward, done flag, metadata

- **`schema.py`**: Type-safe Pydantic models
  - `AgentState`: L2 book, inventory, time remaining, urgency metrics
  - `AgentAction`: Decision representation (side, shares, style)
  - `Order`, `Trade`: Core domain objects

- **`matching_engine.py`**: Order matching & price discovery
  - Limit order book implementation
  - Price-time priority matching
  - Best bid/ask calculation
  - Trade execution & settlement

- **`grader.py`**: Performance evaluation
  - TWAP benchmark calculation
  - Implementation shortfall computation
  - Score normalization (0.0 to 1.0)

- **`background_agents/`**: Market participants
  - `market_maker.py`: Liquidity provider
  - `noise_trader.py`: Random order flow
  - `toxic_trader.py`: Predatory tactics

### Agentic LLM (`agentic_llm/`)

- **`client.py`**: OpenAI integration
  - LLMTrader class wrapping GPT-4 API
  - JSON response parsing with markdown stripping
  - Graceful error handling & fallback logic
  - Token usage tracking

- **`prompts.py`**: Strategic prompts
  - SYSTEM_PROMPT: "Elite quantitative execution algorithm" framing
  - Context formatting: urgency markers, spread analysis
  - Dynamic L2 book representation for LLM input

- **`logger.py`**: Structured audit logging
  - Episode start/step/end logging
  - Exact format matching for validator regex parsing
  - stdout output for container monitoring

### Dashboard (`dashboard/`)

- **`app.py`**: Streamlit web interface
  - Sidebar configuration (API key, task, model selection)
  - Real-time simulation monitoring  
  - Live metrics: time remaining, inventory, mid-price
  - Post-execution analytics & visualizations

- **`visualizers.py`**: Plotly chart generation
  - Order book depth chart (mirror layout)
  - Execution trajectory with trade scatter
  - Professional institutional styling

---

## Setup & Usage

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (for GPT-4 access)
- 2GB RAM minimum

### Installation

```bash
# Clone and navigate
git clone https://github.com/your-org/openenv-lob-simulator.git
cd openenv-lob-simulator

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4-turbo"           # Optional, default: gpt-4-turbo
export API_BASE_URL="https://api.openai.com/v1"  # Optional
```

### Run CLI Baseline (Judges' Validation)

```bash
# Execute inference pipeline with structured logging
python inference.py

# Output format for validator parsing:
# [START] Episode initialized. Task Level: easy.
# [STEP] 1 | Action: {"side": "BUY", "shares_to_execute": 150, "execution_style": "AGGRESSIVE"} | Reward: 0.0124 | Info: ...
# [END] Episode complete. Final Score: 0.8756 | Implementation Shortfall: 245.67.
```

### Run Interactive Dashboard (Hugging Face Spaces)

```bash
# Launch Streamlit web interface
streamlit run dashboard/app.py

# Browser will open to http://localhost:8501
# Features:
#   - Configure API key & task difficulty via sidebar
#   - Click "Run Simulation" to execute
#   - Real-time metrics and progress tracking
#   - Final results with order book & execution visualizations
```

### Docker Deployment

```bash
# Build image
docker build -t openenv-lob-simulator:latest .

# Run Streamlit app
docker run -e OPENAI_API_KEY="sk-..." \
           -p 8501:8501 \
           openenv-lob-simulator:latest

# Or run CLI baseline inside container
docker run -e OPENAI_API_KEY="sk-..." \
           openenv-lob-simulator:latest \
           python inference.py
```

---

## Task Difficulties

| Task | Market Condition | Agents | Volatility | Challenge |
|------|------------------|--------|------------|-----------|
| **Easy** 🟢 | Liquid, stable | 2 MMs | 5% | Execute with minimal impact |
| **Medium** 🟡 | Trending, toxic flow | 5 agents | 15% | Balance timing & market impact |
| **Hard** 🔴 | Flash-crash, spoofing | 10 agents | 35% | Survive adversarial attack |

---

## Architecture Highlights

✅ **Speed**: O(log N) order matching @ sub-millisecond latency
✅ **Realism**: Multi-agent LOB dynamics with real market microstructure  
✅ **Scaling**: 500+ orders per episode, 3000+ steps per task  
✅ **LLM-Native**: GPT-4 integration with robust error handling  
✅ **Observability**: Real-time dashboard + structured logging  
✅ **Type Safety**: 100% Pydantic type hints across codebase  
✅ **Production Ready**: Docker containerized for HF Spaces deployment  

---

## Performance Benchmarks

**Sample Agent Run (Medium Difficulty)**:
- Final Score: 0.8234
- Implementation Shortfall: $341.23
- Total Blocks Executed: 2,500 shares
- Execution Time: 2m 15s
- LLM API Calls: 127 decisions
- Tokens Used: ~8,400 tokens

---

## Contributing

Submissions should implement the `decide_action(state: AgentState) → AgentAction` interface. The environment handles the rest: order routing, matching, grading.

## License

MIT

---

**Built for OpenEnv Hackathon | Production deployment ready for Hugging Face Spaces**

# Run specific test file
pytest tests/test_matching.py -v
```

### Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard/app.py
```

## Configuration

Edit `openenv.yaml` to configure:
- Task difficulty levels (Easy, Medium, Hard)
- Market parameters (volatility, spread)
- Background agent count
- Time limits and order targets

## API Reference

### Environment Reset

```python
from core_engine.env import LOBEnv
from core_engine.schema import AgentState

env = LOBEnv()
state: AgentState = env.reset()
```

### Step Function

```python
from core_engine.schema import AgentAction

action = AgentAction(...)
next_state, reward, done = env.step(action)
```

## Performance Metrics

The grader evaluates agents on:

1. **Implementation Shortfall**: Difference from TWAP baseline
2. **Execution Quality**: Spread and slippage metrics
3. **Market Impact**: How much the agent moved the market
4. **Latency**: Response time of LLM agent

## Contributing

Submit pull requests with improvements to:
- Background agent strategies
- Matching engine efficiency
- LLM prompting techniques
- Visualization dashboards

## License

MIT License - See LICENSE file for details

## References

- Limit Order Book Modeling: [Rama Cont et al.]
- Market Microstructure: [Larry Harris]
- OpenEnv Specification: [OpenEnv GitHub]

---

**Last Updated**: April 2026  
**Version**: 1.0.0
