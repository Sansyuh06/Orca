-   **Saves Time**: Condenses hours of research into a 30-second report.
-   **Reduces Bias**: The "Judge" architecture filters out emotional or hallucinated advice.
-   **Educates**: Explains the *reasoning* behind every signal, helping users become better investors.

## Technical Implementation
FinSight-X demonstrates advanced agentic patterns, implementing **4 key concepts** from the course:

### 1. Multi-Agent System (Model + Orchestration)
We implemented a **"Panel of Experts" with a Judge** pattern:
-   **Generators**: We simulate multiple expert personas (Fundamental Analyst, Technical Analyst, Risk Manager) using different system prompts/models.
-   **Evaluator (Judge)**: A separate "Judge" model (DeepSeek-R1) scores each expert's output on accuracy, depth, and actionability (0-10 scale). It selects the best response, ensuring the user only sees the highest-quality advice.

### 2. Tools (Function Calling)
The agent extends its capabilities with custom Python tools:
-   `fetch_price_data`: Real-time market data via yfinance.
-   `compute_indicators`: Vectorized calculation of RSI, MACD, and Bollinger Bands.
-   `run_forecast`: A linear regression ML model for price trend prediction.
-   `quantum_risk`: A novel tool that calculates "Quantum Risk" by amplifying volatility signals, providing a unique metric not found in standard tools.

### 3. Sessions & Memory
We implemented a persistent `ADKMemoryManager`:
-   **Short-term Memory**: Maintains the conversation context, allowing users to ask follow-up questions (e.g., "What about the risk?").
-   **Long-term Memory**: Stores user preferences (risk tolerance, favorite sectors) to personalize future analysis.

### 4. Observability (Glass Box)
Trust is critical in finance. We built a custom `ADKObservability` layer that logs every step of the agent's "Think-Act-Observe" cycle. These traces are visualized in the UI, allowing users to see exactly which tools were called and how the Judge made its decision.

## Architecture
The application is a Flask-based web app that integrates the agent logic.
-   **Frontend**: HTML/JS dashboard with interactive charts (Chart.js) and a real-time agent execution trace.
-   **Backend**: Python/Flask handling API requests and agent orchestration.
-   **AI Engine**: Local LLM inference via Ollama (or cloud APIs via Google GenAI).

## Team
-   **Sansyuh06**: Lead Developer & Architect

## Resources
-   **Code**: https://github.com/Sansyuh06/stocks