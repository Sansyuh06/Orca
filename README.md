# FinSight-X: Agentic Stock Analysis & Forecasting

**Google 5-Day AI Agents Intensive – Capstone Project**

## Overview
FinSight-X is an intelligent stock analysis agent that combines quantitative finance with multi-LLM reasoning. It helps users understand stocks from multiple angles by orchestrating a team of AI models to analyze price history, technical indicators, and quantum-inspired risk metrics.

## Features
- **Multi-Agent Orchestration**: Uses a "Think-Act-Observe" loop to manage analysis.
- **Multi-Model Consensus**: Aggregates insights from Mistral, Phi-3, Llama 3, and others.
- **Judge-Evaluated Outputs**: Uses a "Judge" model (DeepSeek) to score and select the best analysis.
- **Advanced Tools**:
  - Technical Indicators (RSI, MACD, Bollinger Bands)
  - ML Forecasting (Linear Regression)
  - Quantum-Inspired Risk Assessment
- **Observability**: Full execution tracing and "Glass Box" transparency.
- **Session Memory**: Remembers user context and preferences.

## Architecture
The system follows the **Model + Tools + Orchestration** pattern:
1.  **Orchestrator**: Manages the workflow and tool execution.
2.  **Tools**: Python functions for data fetching and computation.
3.  **Models**: A panel of LLMs provides diverse perspectives.
4.  **Memory**: Stores session history for context-aware responses.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/finsight-x.git
    cd finsight-x
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    python app.py
    ```

4.  Open your browser at `http://localhost:5000`.

## Usage
- Enter a stock symbol (e.g., AAPL, MSFT).
- Ask a question (e.g., "Is this a good buy for the short term?").
- View the comprehensive report, including charts, forecasts, and the agent's reasoning trace.

## Technologies
- **Python**: Core logic and data processing.
- **Flask**: Web server and API.
- **Pandas/NumPy**: Data analysis.
- **Ollama**: Local LLM inference.
- **Google ADK**: Agent Development Kit integration.

## License
MIT
