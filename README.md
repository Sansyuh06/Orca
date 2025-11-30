# Orca: Agentic Stock Analysis & Forecasting

**Google 5-Day AI Agents Intensive – Capstone Project**
**Track:** Concierge Agents

## 🌊 Overview

Orca is an intelligent, autonomous stock analysis agent that acts as your **personal quantitative analyst**. Unlike traditional static dashboards, Orca actively *investigates* stocks for you, combining quantitative finance with cutting-edge multi-LLM reasoning to deliver institutional-grade insights in plain English.

### The Problem We Solve
Retail investors face an information crisis: too much data, too little time. Orca bridges the gap between raw data and actionable wisdom by automating the entire analysis workflow—from data fetching to risk assessment and final recommendation.

---

## 🚀 Key Features (Competition Criteria)

Orca demonstrates the power of agentic AI by implementing **3 Key Concepts** from the course:

### 1. Multi-Agent System ("Panel of Experts")
Instead of relying on a single AI, Orca orchestrates a council of specialized agents:
- **Parallel Agents**: 5 specialized LLMs (Mistral, Llama 3, Phi-3, etc.) analyze the same data simultaneously to provide diverse perspectives.
- **Sequential Agent**: A "Judge" agent (DeepSeek-R1) evaluates all responses and selects the best one based on accuracy and logic.

### 2. Custom Tools
The agent is equipped with specialized Python tools to interact with the real world:
- **`fetch_data`**: Retrieves real-time market data (Yahoo Finance).
- **`calculate_indicators`**: Computes RSI, MACD, Bollinger Bands, and more.
- **`quantum_risk`**: A proprietary algorithm for advanced risk assessment.
- **`run_forecast`**: A hybrid ARIMA/Linear Regression ML model for price prediction.

### 3. Observability (Glass Box)
Orca provides full transparency into its decision-making:
- **AI Canvas**: A visual, interactive graph showing the agent's "Think-Act-Observe" loop in real-time.
- **Execution Traces**: Detailed logs of every tool call, model input, and judge score.

### ✨ Bonus: Gemini Integration
Orca leverages **Google Gemini 1.5 Pro** for its **Portfolio Analysis** feature. Users can upload PDF statements or images of their portfolio, and Gemini's multimodal capabilities analyze the holdings to provide personalized rebalancing advice.

---

## 🏗️ Architecture

Orca follows the **Model + Tools + Orchestration** pattern:

```mermaid
graph TD
    User[User Query] --> Orchestrator[ADK Agent Orchestrator]
    Orchestrator -->|Think| Plan[Determine Strategy]
    Plan -->|Act| Tools[Execute Tools]
    Tools -->|Fetch| Data[Market Data API]
    Tools -->|Calc| Math[Technical Indicators]
    Tools -->|Predict| ML[ARIMA Forecast]
    Orchestrator -->|Observe| Context[Update Memory]
    Context -->|Parallel| Experts[Panel of Experts (Mistral/Llama/Phi)]
    Experts -->|Evaluate| Judge[Judge Agent (DeepSeek)]
    Judge -->|Result| UI[Dashboard & Canvas]
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama (for local inference)
- Google API Key (optional, for Portfolio Analysis)

### Quick Start
1. **Clone the Repo**
   ```bash
   git clone https://github.com/Sansyuh06/stocks.git
   cd stocks
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull Models**
   ```bash
   ollama pull mistral:7b
   ollama pull llama3.1:8b
   ollama pull deepseek-r1:7b
   ```

4. **Run the App**
   ```bash
   python app.py
   ```
   Access the dashboard at `http://localhost:5000`

---

## 📚 Documentation

- **[Writeup](writeup.md)**: Detailed project explanation for Kaggle.
- **[Notebook](finsight_x_capstone.ipynb)**: Deep dive into the code and agent logic.
- **[User Guide](USER_GUIDE.md)**: Step-by-step instructions with screenshots.

---

**Developer**: Sansyuh06 | **License**: MIT
 
