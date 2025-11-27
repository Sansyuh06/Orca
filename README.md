# Orca: Agentic Stock Analysis & Forecasting

**Google 5-Day AI Agents Intensive – Capstone Project**

## 🌊 Overview

Orca is an intelligent, autonomous stock analysis agent that revolutionizes how retail investors understand the market. Unlike traditional static dashboards that merely display data, Orca actively *investigates* stocks for you, combining quantitative finance with cutting-edge multi-LLM reasoning to deliver institutional-grade insights in plain English.

### The Problem We Solve

Retail investors face an overwhelming paradox: they have access to the same raw data as Wall Street professionals—price charts, technical indicators, SEC filings, news feeds—but lack the time, expertise, and tools to synthesize this information into actionable insights. Existing solutions fall into two extremes:

- **Too Simple**: Basic charting apps that show price movements but offer no context or analysis
- **Too Complex**: Professional terminals (Bloomberg, Reuters) with steep learning curves and prohibitive costs

**Orca bridges this gap**, acting as your personal AI quantitative analyst that doesn't just show you data—it *explains* it.

## ✨ Key Features

### 🤖 Multi-Agent Orchestration
Orca implements a sophisticated "Think-Act-Observe" loop inspired by the Google Agent Development Kit (ADK):
- **Think**: Analyzes market data and determines which tools to use
- **Act**: Executes calculations, fetches data, and runs forecasts
- **Observe**: Synthesizes results and learns from outcomes

### 🧠 Multi-Model Consensus ("Panel of Experts")
Instead of relying on a single AI model, Orca orchestrates multiple specialized LLMs working in parallel:
- **Mistral 7B**: General reasoning and balanced analysis
- **Phi-3 Mini**: Efficient, concise technical insights
- **Llama 3.1 8B**: Deep, structured fundamental analysis
- **Gemma 7B**: Sentiment-aware market psychology
- **Qwen2 7B**: Balanced logical reasoning

Each model analyzes the stock independently, providing diverse perspectives that reduce bias and hallucinations.

### ⚖️ Judge-Evaluated Outputs
A separate "Judge" model (DeepSeek-R1) evaluates all expert responses using strict criteria:
- **Accuracy** (0-2 points): Factual correctness
- **Depth of Analysis** (0-2 points): Insight quality beyond surface-level observations
- **Relevance** (0-2 points): Direct addressing of the user's question
- **Clarity & Structure** (0-2 points): Communication effectiveness
- **Actionability** (0-2 points): Practical value for decision-making

Only the highest-scoring response is presented to the user, ensuring quality and reducing information overload.

### 🛠️ Advanced Custom Tools

#### Technical Indicators
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Detects trend changes and momentum
- **Bollinger Bands**: Measures volatility and price extremes
- **Moving Averages**: MA20 and MA50 for trend identification

#### ML Price Forecasting
- Linear regression model trained on historical price data
- 30-day forward price predictions with confidence intervals
- Trend direction analysis (UP/DOWN) with expected percentage change

#### Quantum-Inspired Risk Assessment
A novel proprietary metric that amplifies volatility signals to detect hidden risks:
```
Quantum Risk = Volatility × 1.2
Trade Probability = max(0, 100 - 0.7 × Quantum Risk)
```
This provides early warning signals that traditional metrics might miss.

### 🔍 Glass Box Observability
Complete transparency into the agent's decision-making process:
- Real-time execution trace showing every tool call
- Timestamped logs of each phase (Mission → Scan → Think → Act → Observe)
- Performance metrics (execution time, model response times)
- Full audit trail for reproducibility and trust

### 💾 Session Memory & Context Awareness
Persistent memory system that enables natural conversations:
- **Short-term Memory**: Maintains conversation context for follow-up questions
- **Long-term Memory**: Stores user preferences (risk tolerance, favorite sectors, investment horizon)
- **Contextual Responses**: Adapts analysis based on previous interactions

## 🏗️ Architecture

Orca follows the **Model + Tools + Orchestration** pattern from the Google ADK:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│         (Interactive Dashboard + Chat Canvas)            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Orchestrator                            │
│         (ADK Agent - Think-Act-Observe Loop)             │
└─────┬──────────────┬──────────────┬─────────────────────┘
      │              │              │
┌─────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐
│   Tools   │  │  Models  │  │   Memory   │
│           │  │          │  │            │
│ • Data    │  │ • Panel  │  │ • Session  │
│   Fetch   │  │   of     │  │   Context  │
│ • Tech    │  │   Experts│  │ • User     │
│   Calc    │  │ • Judge  │  │   Prefs    │
│ • ML      │  │   Model  │  │            │
│   Forecast│  │          │  │            │
│ • Quantum │  │          │  │            │
│   Risk    │  │          │  │            │
└───────────┘  └──────────┘  └────────────┘
```

### Component Details

1. **Orchestrator (ADKAgent)**: 
   - Manages the entire workflow
   - Decides which tools to call and when
   - Coordinates between models and memory
   - Implements the Think-Act-Observe cycle

2. **Tools (Python Functions)**:
   - `fetch_price_data`: Multi-source data fetching (Yahoo Finance, Alpha Vantage, with synthetic fallback)
   - `compute_indicators`: Vectorized technical indicator calculations
   - `run_forecast`: Linear regression ML model for price prediction
   - `quantum_risk`: Proprietary risk assessment algorithm
   - `full_analysis`: Complete end-to-end stock analysis

3. **Models (LLM Panel)**:
   - Multiple open-source models running via Ollama
   - Each model provides independent analysis
   - Judge model (DeepSeek-R1) evaluates and ranks responses
   - Supports both local inference and cloud APIs (Google GenAI)

4. **Memory (ADKMemoryManager)**:
   - Session-based storage with unique IDs
   - Short-term: Last 10 interactions
   - Long-term: Persistent user preferences
   - Context injection for personalized responses

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running (for local LLM inference)
- Internet connection (for market data)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Sansyuh06/stocks.git
cd stocks
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `flask`: Web server and API
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `requests`: HTTP requests for data fetching
- `yfinance`: Yahoo Finance API wrapper (optional)

### Step 3: Install Ollama Models
Download the required LLM models:
```bash
ollama pull mistral:7b
ollama pull phi3:mini
ollama pull llama3.1:8b
ollama pull gemma:7b
ollama pull qwen2:7b
ollama pull deepseek-r1:7b
```

### Step 4: Run the Application
```bash
python app.py
```

The server will start on `http://localhost:5000`

### Step 5: Access the Dashboard
Open your browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage Guide

### Basic Analysis Workflow

1. **Select a Stock**: Choose from the dropdown (AAPL, MSFT, AMZN, TSLA, etc.) or enter any valid ticker symbol

2. **Click "Analyze Stock"**: The agent initiates its Think-Act-Observe cycle:
   - Fetches historical price data (1 year)
   - Calculates technical indicators
   - Runs ML price forecast
   - Computes quantum risk metrics

3. **View Results**: The dashboard displays:
   - Interactive price chart with MA20/MA50 overlays
   - 30-day price forecast with confidence bands
   - Key metrics grid (RSI, Sharpe Ratio, Volatility, etc.)
   - Recommendation badge (STRONG BUY, BUY, HOLD, SELL, STRONG SELL)
   - Trading signals with bullish/bearish indicators

### Advanced: Multi-Model Consensus

1. **Select AI Models**: Check the boxes for models you want to consult (e.g., Mistral + Llama 3 + Phi-3)

2. **Click "Get AI Consensus"**: The agent:
   - Sends the analysis to all selected models in parallel
   - Each model provides independent insights
   - The Judge model evaluates and scores each response
   - The best response is highlighted

3. **Review Responses**: Compare different perspectives and see the Judge's reasoning

### AI Canvas (Interactive Chat)

1. **Click "🎨 Open AI Canvas"**: Opens an immersive full-screen chat interface

2. **Ask Questions**: Natural language queries like:
   - "Is this a good long-term investment?"
   - "What are the main risks?"
   - "How does this compare to the sector average?"

3. **Get Contextual Answers**: The agent remembers previous interactions and provides personalized responses

## 🛠️ Technologies & Stack

### Backend
- **Python 3.8+**: Core application logic
- **Flask**: Lightweight web framework for API and routing
- **Pandas**: Time-series data manipulation and analysis
- **NumPy**: Numerical computing for indicators and forecasts

### AI/ML
- **Ollama**: Local LLM inference engine
- **Google GenAI SDK**: Cloud API integration (optional)
- **Custom ML Models**: Linear regression for price forecasting

### Frontend
- **Vanilla JavaScript**: No framework overhead, pure performance
- **Chart.js**: Interactive, responsive financial charts
- **HTML5/CSS3**: Modern, accessible UI with dark mode

### Data Sources
- **Yahoo Finance API**: Primary market data source
- **Alpha Vantage**: Fallback data provider
- **Synthetic Data Generator**: Demo mode when APIs are unavailable

### Architecture Patterns
- **Google ADK**: Agent Development Kit principles
- **Think-Act-Observe Loop**: Autonomous decision-making cycle
- **Panel of Experts**: Multi-model consensus pattern
- **Glass Box**: Full observability and transparency

## 📊 Example Output

For **AAPL** (Apple Inc.):
```
Recommendation: BUY (Score: 72/100)
Confidence: HIGH
Risk Level: LOW

Current Price: $180.45
30-Day Forecast: UP +5.2% → $189.83
Quantum Risk: 18.4%
Trade Probability: 87.1%

Key Signals:
✓ Price above MA20 (bullish)
✓ RSI neutral (52.3)
✓ MACD bullish crossover
✓ ML predicts UP
✓ Low volatility (15.3%)
```

## 🔒 Privacy & Security

- **No Data Storage**: Market data is fetched in real-time and not stored
- **Local Inference**: LLMs run locally via Ollama (no data sent to external APIs by default)
- **Session-Based Memory**: User data is ephemeral and session-scoped
- **Open Source**: Full code transparency for security auditing

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators (Fibonacci, Ichimoku Cloud)
- More sophisticated ML models (LSTM, Transformer-based forecasting)
- Real-time news sentiment integration
- Cryptocurrency support
- Portfolio optimization tools

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Google AI Agents Intensive**: For the ADK framework and agent design patterns
- **Ollama Team**: For making local LLM inference accessible
- **Open-Source Community**: For the incredible models (Mistral, Llama, Phi-3, etc.)

## 📞 Contact

- **Developer**: Sansyuh06
- **GitHub**: https://github.com/Sansyuh06/stocks
- **Project**: Google AI Agents Capstone 2025

---

**Orca: Trade Smarter, Not Harder** 🌊📈
