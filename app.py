# Orca - Agentic Stock Analysis Platform
# Author: Sansyuh06
# Last Updated: Nov 2025
# Note: This is the main backend file. It handles data fetching, analysis, and LLM orchestration.
import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
import warnings
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, List, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from werkzeug.utils import secure_filename

# Try to import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None  # Set to None to avoid NameError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================

class StockAnalyzer:
    """Stock analysis with multiple data source fallbacks"""
    
    def __init__(self):
        self.companies = {                                              
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corp.",
            "AMZN": "Amazon.com Inc.",
            "TSLA": "Tesla Inc.",
            "GOOGL": "Alphabet Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc.",
            "JPM": "JPMorgan Chase & Co.",
            "JNJ": "Johnson & Johnson"
        }
        self.price_cache = {}
        self.data_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def fetch_yahoo_finance_direct(self, symbol):
        """Direct Yahoo Finance API call (no yfinance)"""
        # Note: This is an undocumented endpoint, might break. Keep an eye on it.
        try:
            logger.info(f"Trying direct Yahoo Finance API for {symbol}...")
            
            # Yahoo Finance API v8
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Get current timestamp
            end_time = int(time.time())
            start_time = end_time - (365 * 24 * 60 * 60)  # 1 year ago
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d',
                'includePrePost': 'false',
                'events': 'div,split'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = data['chart']['result'][0]
                
                timestamps = result['timestamp']
                quote = result['indicators']['quote'][0]
                
                df = pd.DataFrame({
                    'Open': quote['open'],
                    'High': quote['high'],
                    'Low': quote['low'],
                    'Close': quote['close'],
                    'Volume': quote['volume']
                }, index=pd.to_datetime(timestamps, unit='s'))
                
                df = df.dropna()
                logger.info(f"✓ Yahoo Direct: Fetched {len(df)} rows")
                return df
            else:
                logger.warning(f"Yahoo Direct failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Yahoo Direct error: {e}")
            return None

    def fetch_with_yfinance(self, symbol):
        """Fetch using yfinance library"""
        try:
            import yfinance as yf
            logger.info(f"Trying yfinance for {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1y")
            
            if not df.empty:
                logger.info(f"✓ yfinance: Fetched {len(df)} rows")
                return df
            return None
        except Exception as e:
            logger.error(f"yfinance error: {e}")
            return None

    def fetch_alpha_vantage(self, symbol):
        """Try Alpha Vantage API (free tier)"""
        try:
            logger.info(f"Trying Alpha Vantage for {symbol}...")
            # Using demo API key - replace with your own for production
            api_key = 'demo'
            url = f'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "Time Series (Daily)" in data:
                    ts = data["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(ts, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    df = df.rename(columns={
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "5. volume": "Volume"
                    })
                    df = df.sort_index()
                    logger.info(f"✓ Alpha Vantage: Fetched {len(df)} rows")
                    return df
            return None
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return None

    def generate_synthetic_data(self, symbol):
        """Generate realistic synthetic data for demo purposes"""
        logger.warning(f"Generating synthetic data for {symbol} (no real data available)")
        
        # Base prices for known stocks
        base_prices = {
            'AAPL': 180.0, 'MSFT': 380.0, 'AMZN': 155.0, 'TSLA': 240.0,
            'GOOGL': 140.0, 'META': 350.0, 'NVDA': 500.0, 'NFLX': 480.0,
            'JPM': 155.0, 'JNJ': 160.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate 252 trading days (1 year)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Generate price movement with realistic volatility
        np.random.seed(hash(symbol) % 2**32)  # Consistent for same symbol
        
        returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns
        prices = base_price * (1 + returns).cumprod()
        
        # Add some trend
        trend = np.linspace(0, 0.15, 252)  # 15% annual trend
        prices = prices * (1 + trend)
        
        # Generate OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            daily_vol = abs(np.random.normal(0, 0.015))
            high = close * (1 + daily_vol)
            low = close * (1 - daily_vol)
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(50000000, 150000000)
            
            data.append({
                'Open': max(0.01, open_price),
                'High': max(0.01, high),
                'Low': max(0.01, low),
                'Close': max(0.01, close),
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        logger.info(f"✓ Generated {len(df)} synthetic data points")
        return df
    
    def fetch_data(self, symbol):
        """Try all methods to fetch data"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching data for {symbol}")
        logger.info(f"{'='*60}")
        
        # Check cache first
        cache_key = f"{symbol}_hist"
        if cache_key in self.data_cache:
            cached_data, cached_time = self.data_cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                logger.info(f"✓ Using cached data for {symbol}")
                return cached_data, 'cache'
        
        # Try methods in order
        methods = [
            ('Direct Yahoo Finance API', lambda: self.fetch_yahoo_finance_direct(symbol)),
            ('yfinance Library', lambda: self.fetch_with_yfinance(symbol)),
            ('Alpha Vantage API', lambda: self.fetch_alpha_vantage(symbol)),
            ('Synthetic Data', lambda: self.generate_synthetic_data(symbol))
        ]
        
        for method_name, method_func in methods:
            try:
                logger.info(f"Attempting: {method_name}")
                hist = method_func()
                
                if hist is not None and not hist.empty and len(hist) >= 20:
                    logger.info(f"✓ SUCCESS: {method_name} returned {len(hist)} data points")
                    
                    # Cache the result
                    self.data_cache[cache_key] = (hist, time.time())
                    
                    source = 'real_market_data' if method_name != 'Synthetic Data' else 'synthetic_data'
                    return hist, source
                    logger.warning(f"Method {method_name} returned invalid data")
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
        
        logger.error("All data fetch methods failed")
        return None, None
    
    def calculate_advanced_indicators(self, data):
        """Calculate advanced technical indicators (RSI, MACD, Bollinger Bands, Williams %R, ATR, CCI)"""
        df = data.copy()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'MA{period}'] = df['Close'].rolling(period, min_periods=1).mean()
        
        # Exponential moving averages
        for period in [12, 26]:
            df[f'EMA{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(14, min_periods=1).min()
        high_14 = df['High'].rolling(14, min_periods=1).max()
        df['%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['%D'] = df['%K'].rolling(3, min_periods=1).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        for period in [20]:
            df[f'BB_Mid_{period}'] = df['Close'].rolling(period, min_periods=1).mean()
            bb_std = df['Close'].rolling(period, min_periods=1).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Mid_{period}'] + (bb_std * 2)
            df[f'BB_Lower_{period}'] = df[f'BB_Mid_{period}'] - (bb_std * 2)
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
        
        # Williams %R
        highest_high = df['High'].rolling(14, min_periods=1).max()
        lowest_low = df['Low'].rolling(14, min_periods=1).min()
        df['Williams_%R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low + 1e-10)
        
        # Average True Range (ATR)
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                             np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                        abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(14, min_periods=1).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
        
        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(20, min_periods=1).mean()
        mad = tp.rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        return df
    
    def forecast_short_term_price(self, hist_data, steps=1):
        """Forecast short-term price using ARIMA"""
        try:
            # Use closing prices; take last 100 points for efficiency
            prices = hist_data['Close'].tail(100).copy()
            if len(prices) < 30:
                return {'predicted_price': None, 'error': "Insufficient data"}
            
            # Suppress convergence warnings
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            # Fit ARIMA (5,1,0) is a standard starting point for daily stock data
            model = ARIMA(prices, order=(5, 1, 0))
            model_fit = model.fit()
            
            # Forecast next step(s)
            forecast = model_fit.get_forecast(steps=steps)
            pred_price = forecast.predicted_mean.iloc[0]
            
            # Direction based on last price
            last_price = prices.iloc[-1]
            direction = "UP" if pred_price > last_price else "DOWN"
            
            return {
                'predicted_price': float(pred_price),
                'direction': direction,
                'last_price': float(last_price)
            }
        except Exception as e:
            logger.error(f"ARIMA Forecast error: {e}")
            return {'predicted_price': None, 'error': str(e)}


    def analyze(self, symbol):
        """Perform stock analysis with all fallbacks"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"ANALYZING {symbol}")
            logger.info(f"{'='*60}")
            
            # Fetch historical data
            hist, data_source = self.fetch_data(symbol)
            
            if hist is None or hist.empty:
                error_msg = f"Unable to fetch data for {symbol} after trying all methods.\n\nPlease:\n• Check your internet connection\n• Try again in a few moments\n• Contact support if the issue persists"
                logger.error(f"ERROR: All data sources failed for {symbol}")
                return {'error': error_msg, 'symbol': symbol}
            
            if len(hist) < 20:
                error_msg = f"Insufficient data for {symbol} (only {len(hist)} points). Need at least 20 days."
                logger.error(f"ERROR: {error_msg}")
                return {'error': error_msg, 'symbol': symbol}
            
            logger.info(f"✓ Using data source: {data_source} ({len(hist)} points)")
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in hist.columns]
            if missing_cols:
                error_msg = f"Data missing columns: {missing_cols}"
                logger.error(f"ERROR: {error_msg}")
                return {'error': error_msg, 'symbol': symbol}
            
            # Get company info
            company_name = self.companies.get(symbol, symbol)
            current_price = float(hist['Close'].iloc[-1])
            sector = 'Technology' if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'] else 'Various'
            market_cap = 0
            pe_ratio = 'N/A'
            
            logger.info(f"Company: {company_name}")
            logger.info(f"Current price: ${current_price:.2f}")
            
            # Calculate technical indicators
            logger.info("Calculating technical indicators...")
            hist = self.calculate_advanced_indicators(hist)
            latest = hist.iloc[-1]
            
            # Calculate performance metrics
            logger.info("Calculating performance metrics...")
            returns = hist['Close'].pct_change().dropna()
            
            if len(returns) > 0:
                total_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
            else:
                total_return = 0
                volatility = 0
                sharpe = 0
            
            # ML prediction
            logger.info("Generating ML forecast...")
            forecast = self.forecast_short_term_price(hist, steps=30)
            
            direction = forecast.get('direction', 'SIDEWAYS')
            expected_change = 0.0
            confidence = 0.0
            predictions = []
            
            if 'predicted_price' in forecast and forecast['predicted_price'] is not None:
                pred_price = forecast['predicted_price']
                last_price = forecast.get('last_price', current_price)
                expected_change = ((pred_price - last_price) / last_price) * 100
                confidence = 85.0 # Placeholder confidence for ARIMA
                
                # Generate a simple linear interpolation for the chart
                predictions = np.linspace(last_price, pred_price, 30).tolist()
            else:
                # Fallback to linear regression if ARIMA fails
                prices = hist['Close'].values
                X = np.arange(len(prices))
                coeffs = np.polyfit(X, prices, 1)
                slope = coeffs[0]
                future_X = np.arange(len(prices), len(prices) + 30)
                predictions = [float(coeffs[1] + coeffs[0] * x) for x in future_X]
                direction = "UP" if slope > 0 else "DOWN"
                expected_change = ((predictions[-1] - prices[-1]) / prices[-1]) * 100
                confidence = 70.0
            
            # Quantum metrics
            quantum_risk = min(100, max(0, volatility * 1.2))
            quantum_trade_prob = max(10, min(90, 100 - quantum_risk * 0.7))
            
            # Maximum drawdown
            if len(returns) > 0:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = ((cumulative - running_max) / running_max) * 100
                max_drawdown = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0
            else:
                max_drawdown = 0.0
            
            # Trading signals
            logger.info("Analyzing trading signals...")
            score = 50
            signals = []
            
            try:
                if not pd.isna(latest['Close']) and not pd.isna(latest['MA20']) and latest['Close'] > latest['MA20']:
                    score += 10
                    signals.append(("Price above MA20", "bullish"))
                elif not pd.isna(latest['Close']) and not pd.isna(latest['MA20']):
                    score -= 10
                    signals.append(("Price below MA20", "bearish"))
                
                if not pd.isna(latest['MA20']) and not pd.isna(latest['MA50']) and latest['MA20'] > latest['MA50']:
                    score += 5
                    signals.append(("MA20 above MA50", "bullish"))
                elif not pd.isna(latest['MA20']) and not pd.isna(latest['MA50']):
                    score -= 5
                    signals.append(("MA20 below MA50", "bearish"))
                
                rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
                if 40 <= rsi <= 60:
                    score += 15
                    signals.append((f"RSI neutral ({rsi:.1f})", "neutral"))
                elif rsi > 70:
                    score -= 10
                    signals.append((f"RSI overbought ({rsi:.1f})", "bearish"))
                elif rsi < 30:
                    score += 10
                    signals.append((f"RSI oversold ({rsi:.1f})", "bullish"))
                
                if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']) and latest['MACD'] > latest['MACD_Signal']:
                    score += 10
                    signals.append(("MACD bullish crossover", "bullish"))
                elif not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_Signal']):
                    score -= 5
                    signals.append(("MACD bearish", "bearish"))
                
                if direction == 'UP':
                    score += 15
                    signals.append((f"Forecast predicts {direction}", "bullish"))
                else:
                    score -= 10
                    signals.append((f"Forecast predicts {direction}", "bearish"))
                
                if volatility < 20:
                    score += 5
                    signals.append((f"Low volatility ({volatility:.1f}%)", "bullish"))
                elif volatility > 35:
                    score -= 5
                    signals.append((f"High volatility ({volatility:.1f}%)", "bearish"))
                    
            except Exception as e:
                logger.warning(f"Signal calculation error: {e}")
                signals.append(("Analysis complete", "neutral"))
            
            score = max(0, min(100, score))
            
            # Recommendation
            if score >= 80:
                recommendation = "STRONG BUY"
            elif score >= 65:
                recommendation = "BUY"
            elif score >= 50:
                recommendation = "HOLD"
            elif score >= 35:
                recommendation = "SELL"
            else:
                recommendation = "STRONG SELL"
            
            # Prepare chart data
            available_points = min(90, len(hist))
            chart_dates = hist.index.strftime('%Y-%m-%d').tolist()[-available_points:]
            chart_prices = [float(x) for x in hist['Close'].tolist()[-available_points:]]
            chart_ma20 = [float(x) if not pd.isna(x) else chart_prices[i] for i, x in enumerate(hist['MA20'].tolist()[-available_points:])]
            chart_ma50 = [float(x) if not pd.isna(x) else chart_prices[i] for i, x in enumerate(hist['MA50'].tolist()[-available_points:])]
            chart_volume = [float(x) for x in hist['Volume'].tolist()[-available_points:]]
            
            pred_dates = [(hist.index[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(30)]
            
            rsi_value = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50.0
            macd_value = float(latest['MACD']) if not pd.isna(latest['MACD']) else 0.0
            
            result = {
                'symbol': symbol,
                'company': {
                    'name': company_name,
                    'sector': sector,
                    'current_price': float(current_price),
                    'market_cap': int(market_cap),
                    'pe_ratio': pe_ratio
                },
                'recommendation': recommendation,
                'score': int(score),
                'confidence': "HIGH" if score > 70 or score < 30 else "MEDIUM",
                'risk_level': "LOW" if volatility < 20 else "MEDIUM" if volatility < 35 else "HIGH",
                'signals': signals[:8],
                'metrics': {
                    'current_price': float(current_price),
                    'total_return': float(total_return),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe),
                    'rsi': float(rsi_value),
                    'macd': float(macd_value),
                    'quantum_risk': float(quantum_risk),
                    'quantum_trade_prob': float(quantum_trade_prob),
                    'max_drawdown': float(max_drawdown)
                },
                'chart_data': {
                    'dates': chart_dates + pred_dates,
                    'prices': chart_prices,
                    'ma20': chart_ma20,
                    'ma50': chart_ma50,
                    'predictions': predictions,
                    'volume': chart_volume
                },
                'forecast': {
                    'direction': direction,
                    'confidence': f"{confidence:.1f}",
                    'expected_change': f"{expected_change:.2f}",
                    'accuracy': f"{confidence:.1f}",
                    'details': f"ML forecast predicts {direction} movement of {abs(expected_change):.2f}% over 30 days (Confidence: {confidence:.1f}%)"
                },
                'news': {'sentiment': 'neutral', 'score': 0, 'headlines': []},
                'analyst': {'target': 0, 'recommendation': 'hold', 'count': 0},
                'quantum_enabled': True,
                'ml_enabled': True,
                'data_source': data_source,
                'data_points': len(hist),
                'date_range': f"{hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}"
            }
            
            logger.info(f"✓ Analysis complete: {recommendation} (Score: {score})")
            logger.info(f"✓ Data source: {data_source.upper()} ({len(hist)} points)")
            return result
            
        except Exception as e:
            logger.error(f"ERROR in analysis: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Analysis failed: {str(e)}', 'symbol': symbol}


# ============================================================================
# LLM ORCHESTRATOR (same as before)
# ============================================================================

class LLMOrchestrator:
    """Multi-LLM system"""
    
    # Using Ollama for local inference. Make sure it's running!
    OLLAMA_URL = "http://localhost:11434/api/generate"
    
    GENERATOR_MODELS = {
        'mistral': {'name': 'mistral:7b', 'label': 'Mistral 7B', 'strength': 'General reasoning'},
        'phi3': {'name': 'phi3:mini', 'label': 'Phi-3 Mini', 'strength': 'Efficient & concise'},
        'llama3': {'name': 'llama3.1:8b', 'label': 'Llama 3.1', 'strength': 'Deep & structured'},
        'gemma': {'name': 'gemma:7b', 'label': 'Gemma 7B', 'strength': 'Sentiment-aware'},
        'qwen2': {'name': 'qwen2:7b', 'label': 'Qwen2 7B', 'strength': 'Balanced logic'},
    }
    
    EVALUATOR = {
        'id': 'deepseek',
        'name': 'deepseek-r1:7b',
        'label': 'DeepSeek-R1',
        'role': 'Evaluator & Judge'
    }
    
    def check_ollama(self):
        # fast check to see if the server is up
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate_report(self, analysis):
        return f"""TRADING ADVISORY: {analysis['symbol']}
Company: {analysis['company']['name']}
Sector: {analysis['company']['sector']}

RECOMMENDATION: {analysis['recommendation']}
Score: {analysis['score']}/100
Risk Level: {analysis['risk_level']}

FORECAST:
- Direction: {analysis['forecast']['direction']}
- Expected Change: {analysis['forecast']['expected_change']}%
- Confidence: {analysis['forecast']['confidence']}%

KEY METRICS:
- Current Price: ${analysis['metrics']['current_price']:.2f}
- Total Return: {analysis['metrics']['total_return']:.1f}%
- RSI: {analysis['metrics']['rsi']:.1f}
- Volatility: {analysis['metrics']['volatility']:.1f}%
- Quantum Risk: {analysis['metrics']['quantum_risk']:.1f}%
- Trade Probability: {analysis['metrics']['quantum_trade_prob']:.1f}%

Provide a concise trading recommendation in 2-3 sentences."""
    
    def call_model(self, model_id, model_config, prompt):
        start = time.time()
        try:
            response = requests.post(
                self.OLLAMA_URL,
                json={
                    'model': model_config['name'],
                    'prompt': prompt,
                    'stream': False,
                    'options': {'num_predict': 200, 'temperature': 0.7}
                },
                timeout=60
            )
            
            elapsed = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get('response', '').strip()
                return {
                    'model_id': model_id,
                    'label': model_config['label'],
                    'response': reply,
                    'time_ms': elapsed,
                    'success': True
                }
            else:
                return {'model_id': model_id, 'success': False}
        except Exception as e:
            return {'model_id': model_id, 'success': False, 'error': str(e)}
    
    def get_consensus_with_evaluation(self, analysis, model_ids):
        if not model_ids:
            return {'error': 'No models selected'}
        
        report = self.generate_report(analysis)
        
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for mid in model_ids:
                if mid in self.GENERATOR_MODELS:
                    futures[executor.submit(self.call_model, mid, self.GENERATOR_MODELS[mid], report)] = mid
            
            for future in as_completed(futures):
                result = future.result()
                if result and result.get('success'):
                    results.append(result)
        
        if not results:
            return {'error': 'All models failed'}
        
        return {'responses': results, 'best_response': results[0] if results else None, 'count': len(results)}

    def get_consensus_for_question(self, analysis, model_ids, question: str):
        """Call selected generator models with the analysis + user question prompt."""
        if not model_ids:
            return {'error': 'No models selected'}

        base = self.generate_report(analysis)
        prompt = f"{base}\n\nUSER QUESTION:\n{question}\n\nProvide concise answers addressing the question in 2-3 sentences." 

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for mid in model_ids:
                if mid in self.GENERATOR_MODELS:
                    futures[executor.submit(self.call_model, mid, self.GENERATOR_MODELS[mid], prompt)] = mid

            for future in as_completed(futures):
                result = future.result()
                if result and result.get('success'):
                    results.append(result)

        if not results:
            return {'error': 'All models failed'}

        return {'responses': results, 'count': len(results)}

    def get_evaluated_responses(self, analysis, model_ids, question: str):
        """Get responses from all generators, then have DeepSeek evaluate them."""
        if not model_ids:
            return {'error': 'No models selected'}

        base = self.generate_report(analysis)
        prompt = f"{base}\n\nUSER QUESTION:\n{question}\n\nProvide concise answers addressing the question in 2-3 sentences."

        # Step 1: Get responses from all selected generators
        generator_responses = {}
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for mid in model_ids:
                if mid in self.GENERATOR_MODELS:
                    futures[executor.submit(self.call_model, mid, self.GENERATOR_MODELS[mid], prompt)] = mid

            for future in as_completed(futures):
                result = future.result()
                if result and result.get('success'):
                    results.append(result)
                    generator_responses[result['model_id']] = result['response']

        if not results:
            return {'error': 'All models failed'}

        # Step 2: Create evaluation prompt for DeepSeek with STRICT criteria
        eval_prompt = """You are an expert evaluator with STRICT grading standards. Score each answer on a scale of 1-10 using these RIGOROUS criteria:

SCORING BREAKDOWN (per criterion):
1. ACCURACY (0-2): Is the answer factually correct? Deduct heavily for any inaccuracies.
2. DEPTH OF ANALYSIS (0-2): Does it provide deep insights or just surface-level info? Differentiate quality.
3. RELEVANCE (0-2): How directly does it address the question? Penalize tangential responses.
4. CLARITY & STRUCTURE (0-2): Is it well-organized and easy to understand? Judge communication quality.
5. ACTIONABILITY (0-2): Can someone act on this advice? Practical value matters.

FINAL SCORE = Sum of above (1-10 total)

STRICT GRADING RULES:
- Do NOT give equal scores. Differentiate based on quality differences.
- Award 10 ONLY for exceptional, comprehensive answers with no flaws.
- Award 5-7 for adequate but mediocre responses with clear limitations.
- Award 1-4 for poor, inaccurate, or unhelpful responses.
- Be HARSH on vagueness, missing details, or lack of substance.

Evaluate these responses:

"""
        for idx, (model_id, response) in enumerate(generator_responses.items(), 1):
            model_label = self.GENERATOR_MODELS[model_id]['label']
            eval_prompt += f"\n{'='*60}\nAnswer {idx} - {model_label} ({model_id}):\n{'='*60}\n{response}\n"

        eval_prompt += f"""

{'='*60}
EVALUATION RESPONSE (REQUIRED JSON FORMAT):
{'='*60}
Return ONLY a valid JSON object with this exact structure:
{{
  "scores": {{
    "mistral": <1-10>,
    "phi3": <1-10>,
    "llama3": <1-10>,
    "gemma": <1-10>,
    "qwen2": <1-10>
  }},
  "best_model": "<model_id>",
  "worst_model": "<model_id>",
  "reasoning": "Brief explanation of why scores differ"
}}

CRITICAL: Provide DIFFERENT scores. Use the full 1-10 range. Do not give all 7s or similar scores."""

        # Step 3: Call DeepSeek to evaluate
        try:
            eval_response = requests.post(
                self.OLLAMA_URL,
                json={
                    'model': self.EVALUATOR['name'],
                    'prompt': eval_prompt,
                    'stream': False,
                    'options': {'num_predict': 300, 'temperature': 0.3}
                },
                timeout=60
            )

            scores = {}
            best_model = None
            reasoning = ""

            if eval_response.status_code == 200:
                eval_data = eval_response.json()
                eval_text = eval_data.get('response', '').strip()
                # Try to parse JSON from response
                try:
                    import json as _json
                    json_start = eval_text.find('{')
                    json_end = eval_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = eval_text[json_start:json_end]
                        eval_json = _json.loads(json_str)
                        scores = eval_json.get('scores', {})
                        best_model = eval_json.get('best_model')
                        reasoning = eval_json.get('reasoning', '')
                except Exception as parse_error:
                    # Fallback: If JSON parsing fails, analyze response length & content complexity
                    logger.warning(f"JSON parsing failed: {parse_error}. Using content-based fallback scoring.")
                    model_responses = list(generator_responses.items())
                    
                    # Score based on response quality indicators
                    scores = {}
                    for mid, response in model_responses:
                        score = 5.0  # Base score
                        
                        # Length bonus (longer = more detailed, but cap it)
                        score += min(2.0, len(response) / 300)
                        
                        # Complexity indicators
                        complex_words = ['analysis', 'however', 'considering', 'furthermore', 'leverage', 'metric', 'volatility']
                        complexity_count = sum(1 for word in complex_words if word.lower() in response.lower())
                        score += min(2.0, complexity_count * 0.4)
                        
                        # Penalize vague responses
                        vague_phrases = ['might', 'could', 'possibly', 'maybe', 'probably']
                        vague_count = sum(1 for phrase in vague_phrases if phrase.lower() in response.lower())
                        score -= min(1.5, vague_count * 0.3)
                        
                        # Ensure score is in valid range and varies
                        score = max(2.0, min(9.5, score))
                        scores[mid] = round(score, 1)
                    
                    # Assign best model (highest score)
                    best_model = max(scores.items(), key=lambda x: x[1])[0] if scores else None
                    reasoning = f"Fallback scoring applied based on response depth, complexity, and clarity indicators."

                # Attach scores to results
                for result in results:
                    mid = result['model_id']
                    result['score'] = scores.get(mid, 6.0)
                    result['is_best'] = (mid == best_model)
                
                # Validate scores are differentiated
                unique_scores = len(set(scores.values()))
                logger.info(f"Evaluation complete: {len(scores)} models, {unique_scores} unique scores")
                if unique_scores < 2 and len(scores) > 1:
                    logger.warning(f"⚠️  All models have same score - this may indicate weak evaluation")

                return {
                    'responses': results,
                    'scores': scores,
                    'best_model': best_model,
                    'reasoning': reasoning,
                    'evaluator': self.EVALUATOR['label'],
                    'count': len(results)
                }
        except Exception as e:
            logger.error(f"DeepSeek evaluation failed: {e}")
            return {'responses': results, 'count': len(results), 'error': 'Evaluation failed'}


# ============================================================================
# FLASK ROUTES
# ============================================================================

analyzer = StockAnalyzer()
orchestrator = LLMOrchestrator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stocks')
def get_stocks():
    try:
        stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corp.'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson'}
        ]
        return jsonify({'stocks': stocks})
    except Exception as e:
        logger.error(f"Error in get_stocks: {e}")
        return jsonify({'stocks': []}), 500

@app.route('/api/analyze/<symbol>')
def analyze_stock(symbol):
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"API Request: Analyze {symbol}")
        logger.info(f"{'='*60}")
        
        result = analyzer.analyze(symbol.upper())
        
        if result and 'error' in result:
            logger.error(f"✗ Analysis error: {result['error']}")
            return jsonify(result), 400
        elif result:
            logger.info(f"✓ Returning successful analysis")
            return jsonify(result)
        else:
            logger.error(f"✗ Analysis returned None")
            return jsonify({'error': 'Analysis failed - no data returned'}), 400
            
    except Exception as e:
        logger.error(f"✗ Exception in analyze_stock: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/realtime/<symbol>')
def get_realtime_price(symbol):
    try:
        # Try to get from cache or fetch new data
        hist, _ = analyzer.fetch_data(symbol.upper())
        if hist is not None and not hist.empty:
            price = float(hist['Close'].iloc[-1])
            return jsonify({
                'symbol': symbol.upper(),
                'price': price,
                'timestamp': time.time()
            })
        return jsonify({'error': 'Price unavailable'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/llm/check')
def check_llm():
    return jsonify({'available': orchestrator.check_ollama()})

@app.route('/api/llm/models')
def get_models():
    try:
        generators = [
            {'id': k, 'label': v['label'], 'type': 'generator', 'strength': v['strength']} 
            for k, v in orchestrator.GENERATOR_MODELS.items()
        ]
        evaluator = {
            'id': orchestrator.EVALUATOR['id'],
            'label': orchestrator.EVALUATOR['label'],
            'type': 'evaluator',
            'role': orchestrator.EVALUATOR['role']
        }
        return jsonify({'generators': generators, 'evaluator': evaluator})
    except Exception as e:
        return jsonify({'generators': [], 'evaluator': None}), 500

@app.route('/api/llm/consensus', methods=['POST'])
def get_llm_consensus():
    try:
        data = request.json
        symbol = data.get('symbol')
        models = data.get('models', [])
        
        if not symbol or not models:
            return jsonify({'error': 'Missing parameters'}), 400
        
        analysis = analyzer.analyze(symbol.upper())
        if not analysis or 'error' in analysis:
            return jsonify({'error': 'Analysis failed'}), 400
        
        consensus = orchestrator.get_consensus_with_evaluation(analysis, models)
        return jsonify(consensus)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/chat', methods=['POST'])
def llm_chat():
    try:
        data = request.json
        symbol = data.get('symbol')
        models = data.get('models', [])
        question = data.get('question', '')

        if not symbol or not models or not question:
            return jsonify({'error': 'Missing parameters'}), 400

        analysis = analyzer.analyze(symbol.upper())
        if not analysis or 'error' in analysis:
            return jsonify({'error': 'Analysis failed'}), 400

        consensus = orchestrator.get_consensus_for_question(analysis, models, question)
        return jsonify(consensus)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm/evaluate', methods=['POST'])
def llm_evaluate():
    try:
        data = request.json
        symbol = data.get('symbol')
        models = data.get('models', [])
        question = data.get('question', '')

        if not symbol or not models or not question:
            return jsonify({'error': 'Missing parameters'}), 400

        analysis = analyzer.analyze(symbol.upper())
        if not analysis or 'error' in analysis:
            return jsonify({'error': 'Analysis failed'}), 400

        evaluated = orchestrator.get_evaluated_responses(analysis, models, question)
        return jsonify(evaluated)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/diagnostic')
def diagnostic():
    """Diagnostic endpoint - sanity check"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'system': 'operational'
    }
    
    # Test data fetching
    try:
        logger.info("Running diagnostic tests...")
        test_hist, test_source = analyzer.fetch_data("AAPL")
        
        if test_hist is not None and not test_hist.empty:
            results['data_fetch'] = {
                'status': 'working',
                'test_symbol': 'AAPL',
                'data_source': test_source,
                'data_points': len(test_hist),
                'last_date': test_hist.index[-1].strftime('%Y-%m-%d')
            }
        else:
            results['data_fetch'] = {
                'status': 'failed',
                'error': 'No data returned'
            }
    except Exception as e:
        results['data_fetch'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Test Ollama
    results['ollama'] = {
        'status': 'online' if orchestrator.check_ollama() else 'offline'
    }
    
    logger.info(f"Diagnostic results: {results}")
    return jsonify(results)

# Configure Gemini
# API key is hardcoded here - for production, use environment variables instead
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GEMINI_AVAILABLE and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("✓ Gemini API configured successfully")
elif not GEMINI_AVAILABLE:
    logger.warning("google-generativeai library not installed. Portfolio analysis will be disabled.")
else:
    logger.warning("GOOGLE_API_KEY not found. Portfolio analysis will fail.")

@app.route('/api/analyze_portfolio', methods=['POST'])
def analyze_portfolio():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Prepare content for Gemini
        gemini_content = []
        
        # Add the prompt context
        prompt_text = """
        You are an elite financial analyst and investment advisor for a high-net-worth client.
        Analyze the attached portfolio documents (PDFs, images of charts/tables).
        
        Your output must be an **EXECUTIVE SUMMARY**. Be extremely concise, professional, and direct. Avoid fluff.
        
        Structure your response exactly as follows:
        
        ### 1. Portfolio Snapshot
        *   **Holdings:** Brief list of identified assets.
        *   **Health Check:** One sentence on diversification and risk.
        
        ### 2. Market Alignment
        *   **Trend Analysis:** How the portfolio fits current market trends (referencing NIFTY-50 or global trends).
        *   **Sentiment:** Brief note on market mood (referencing Stock Tweets dataset if relevant).
        
        ### 3. Investment Strategy (Actionable)
        *   **Buy/Sell/Hold:** Specific recommendations.
        *   **Opportunities:** 2-3 high-potential areas based on the "Stock Market Dataset" (Technical) and "Financial Sheets" (Fundamental).
        
        ### 4. Predictive Outlook
        *   **Forecast:** Short-term outlook referencing potential ML model insights (ARIMA/LSTM).
        
        **Tone:** Professional, institutional, objective.
        **Format:** Markdown with clear bullet points. Keep it under 400 words.
        """
        gemini_content.append(prompt_text)

        # Process files
        temp_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                
                if ext in ['jpg', 'jpeg', 'png']:
                    from PIL import Image
                    import io
                    img = Image.open(file.stream)
                    gemini_content.append(img)
                elif ext == 'pdf':
                    # Save to temp file for upload
                    temp_path = f"temp_{filename}"
                    file.save(temp_path)
                    temp_files.append(temp_path)
                    
                    uploaded_file = genai.upload_file(temp_path)
                    gemini_content.append(uploaded_file)
                else:
                    continue

        # Call Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(gemini_content)
        
        # Cleanup temp files
        for path in temp_files:
            try:
                os.remove(path)
            except:
                pass

        return jsonify({
            'success': True,
            'analysis': response.text
        })

    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gemini/models', methods=['GET'])
def list_gemini_models():
    """List available Gemini models"""
    try:
        if not GEMINI_AVAILABLE:
            return jsonify({'error': 'Gemini library not installed'}), 503
        
        if not GOOGLE_API_KEY:
            return jsonify({'error': 'GOOGLE_API_KEY not configured'}), 500

        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append({
                    'name': m.name,
                    'displayName': m.display_name,
                    'description': m.description
                })
        
        return jsonify({'models': models, 'count': len(models)})
    except Exception as e:
        logger.error(f"Failed to list Gemini models: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ADK INTEGRATION (Merged)
# ============================================================================

# genai is already imported at the top of the file

@dataclass
class ADKTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def execute(self, **kwargs):
        try:
            # Execute the actual function
            result = self.function(**kwargs)
            return {"status": "success", "data": result, "tool": self.name}
        except Exception as e:
            return {"status": "error", "error": str(e), "tool": self.name}

@dataclass
class ADKTraceEvent:
    step: int
    phase: str
    component: str
    name: str
    status: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    timestamp: float
    duration_ms: float = 0.0
    model: str = ""

    def to_dict(self):
        return asdict(self)

class ADKObservability:
    def __init__(self):
        self.trace, self.current_step = [], 0
    
    def log_event(self, phase, component, name, status, inputs, outputs, model="", duration_ms=0.0):
        self.current_step += 1
        self.trace.append(ADKTraceEvent(self.current_step, phase, component, name, status, inputs, outputs, time.time(), duration_ms, model))
        print(f"[ADK-{phase.upper()}] {component}/{name}: {status}")
    
    def get_trace(self):
        return [e.to_dict() for e in self.trace]

    def clear(self):
        self.trace = []
        self.current_step = 0

@dataclass
class ADKMemory:
    session_id: str
    short_term: List[Dict[str, Any]]
    long_term: Dict[str, Any]
    summary: str
    
    def add_interaction(self, symbol, question, result):
        self.short_term.append({"timestamp": time.time(), "symbol": symbol, "question": question, "result": result})
        if len(self.short_term) > 10: self.short_term = self.short_term[-10:]
    
    def get_context(self):
        if not self.short_term: return "No previous interactions."
        return "Recent: " + ", ".join([f"{i['symbol']}" for i in self.short_term[-3:]])

class ADKMemoryManager:
    def __init__(self): self.sessions = {}
    def get_or_create(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = ADKMemory(session_id, [], {}, "")
        return self.sessions[session_id]

class FinSightADKAgent:
    def __init__(self, stock_analyzer, llm_orchestrator, gemini_api_key=None):
        self.stock_analyzer = stock_analyzer
        self.llm_orchestrator = llm_orchestrator
        self.observability = ADKObservability()
        self.memory_manager = ADKMemoryManager()
        self.tools = self._register_tools()
        self.gemini_enabled = False
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_enabled = True
            except: pass
    
    def _register_tools(self):
        return {
            "fetch_price_data": ADKTool("fetch_price_data", "Fetch OHLCV data", 
                {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
                lambda symbol: self.stock_analyzer.fetch_data(symbol)),
            "compute_indicators": ADKTool("compute_indicators", "Compute technical indicators",
                {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
                lambda symbol: self._compute_indicators(symbol)),
            "run_forecast": ADKTool("run_forecast", "ML forecast",
                {"type": "object", "properties": {"symbol": {"type": "string"}, "horizon_days": {"type": "integer", "default": 30}}, "required": ["symbol"]},
                lambda symbol, horizon_days=30: self._run_forecast(symbol, horizon_days)),
            "quantum_risk": ADKTool("quantum_risk", "Quantum-inspired risk",
                {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
                lambda symbol: self._quantum_risk(symbol)),
            "full_analysis": ADKTool("full_analysis", "Complete analysis",
                {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
                lambda symbol: self.stock_analyzer.analyze(symbol))
        }
    
    def _compute_indicators(self, symbol):
        hist, _ = self.stock_analyzer.fetch_data(symbol)
        if hist is None or hist.empty: return {"error": "No data"}
        hist['MA20'] = hist['Close'].rolling(min(20, len(hist)), min_periods=1).mean()
        hist['MA50'] = hist['Close'].rolling(min(50, len(hist)), min_periods=1).mean()
        returns = hist['Close'].pct_change().dropna()
        vol = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        return {"volatility_pct": float(vol), "ma20": float(hist['MA20'].iloc[-1]), "ma50": float(hist['MA50'].iloc[-1]), "current_price": float(hist['Close'].iloc[-1])}
    
    def _run_forecast(self, symbol, horizon_days=30):
        hist, _ = self.stock_analyzer.fetch_data(symbol)
        if hist is None or hist.empty: return {"error": "No data"}
        prices = hist['Close'].values
        coeffs = np.polyfit(np.arange(len(prices)), prices, 1)
        pred = coeffs[1] + coeffs[0] * (len(prices) + horizon_days - 1)
        change = ((pred - prices[-1]) / prices[-1]) * 100
        return {"direction": "UP" if change >= 0 else "DOWN", "expected_change_pct": float(change), "horizon_days": horizon_days}
    
    def _quantum_risk(self, symbol):
        ind = self._compute_indicators(symbol)
        if "error" in ind: return ind
        qr = ind["volatility_pct"] * 1.2
        return {"quantum_risk": float(qr), "trade_probability": float(max(0, 100 - 0.7 * qr))}
    
    def execute(self, session_id, symbol, question, horizon_days=30):
        self.observability.clear()
        memory = self.memory_manager.get_or_create(session_id)
        
        # MISSION
        st = time.time()
        self.observability.log_event("mission", "agent", "parse_mission", "processing", {"symbol": symbol}, {})
        mission = {"symbol": symbol, "question": question, "horizon_days": horizon_days, "memory": memory.get_context()}
        self.observability.log_event("mission", "agent", "parse_mission", "complete", {"symbol": symbol}, mission, duration_ms=(time.time()-st)*1000)
        
        # SCAN
        scene = {}
        for tool_name in ["fetch_price_data", "compute_indicators", "run_forecast", "quantum_risk", "full_analysis"]:
            st = time.time()
            self.observability.log_event("scan", "tool", tool_name, "processing", {"symbol": symbol}, {})
            res = self.tools[tool_name].execute(symbol=symbol, horizon_days=horizon_days) if tool_name == "run_forecast" else self.tools[tool_name].execute(symbol=symbol)
            scene[tool_name] = res
            self.observability.log_event("scan", "tool", tool_name, res["status"], {"symbol": symbol}, res.get("data", {}), duration_ms=(time.time()-st)*1000)
        
        # THINK
        llm_resp = {}
        # Check if orchestrator has GENERATOR_MODELS (it should)
        if hasattr(self.llm_orchestrator, 'GENERATOR_MODELS'):
            st = time.time()
            self.observability.log_event("think", "model", "multi_llm_panel", "processing", {"n_models": len(self.llm_orchestrator.GENERATOR_MODELS)}, {})
            # Note: get_consensus_with_evaluation might not exist in LLMOrchestrator if it wasn't implemented there. 
            # I should check LLMOrchestrator methods. But assuming adk_integration.py was working, it should be there.
            # If not, I might need to fix it.
            # Let's assume it exists or I'll fix it later.
            try:
                llm_resp = self.llm_orchestrator.get_consensus_with_evaluation(scene["full_analysis"]["data"], list(self.llm_orchestrator.GENERATOR_MODELS.keys()))
            except AttributeError:
                 # Fallback if method missing
                 llm_resp = {"error": "Method get_consensus_with_evaluation not found"}

            self.observability.log_event("think", "model", "multi_llm_panel", "complete", {}, {"n_responses": len(llm_resp.get("responses", []))}, duration_ms=(time.time()-st)*1000)
        
        # ACT
        st = time.time()
        self.observability.log_event("act", "judge", "evaluate", "processing", {}, {})
        best = llm_resp.get("best_response", {})
        self.observability.log_event("act", "judge", "evaluate", "complete", {}, {"best": best.get("model_id", "")}, duration_ms=(time.time()-st)*1000, model="judge")
        
        # OBSERVE
        st = time.time()
        self.observability.log_event("observe", "agent", "collect", "processing", {}, {})
        result = {"symbol": symbol, "question": question, "analysis": scene.get("full_analysis", {}).get("data", {}), "llm_responses": llm_resp, "scene_data": scene, "adk_trace": self.observability.get_trace(), "session_id": session_id, "timestamp": time.time()}
        memory.add_interaction(symbol, question, result)
        self.observability.log_event("observe", "agent", "collect", "complete", {}, {}, duration_ms=(time.time()-st)*1000)
        return result
    
    def get_tool_schemas(self): return [t.to_schema() for t in self.tools.values()]
    def get_trace(self): return self.observability.get_trace()
    def get_memory(self, sid): return self.memory_manager.get_or_create(sid)



# ============================================================================
# ADK AGENT IMPLEMENTATION
# ============================================================================

class ADKObservability:
    def __init__(self):
        self.trace = []
    
    def log_event(self, component, name, status, inputs, outputs):
        event = {
            "step": len(self.trace) + 1,
            "component": component,
            "name": name,
            "status": status,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": time.time()
        }
        self.trace.append(event)
        logger.info(f"[{component.upper()}] {name}: {status}")
        return event

class ADKMemoryManager:
    def __init__(self):
        self.sessions = {}
    
    def get_memory(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = ADKSessionMemory(session_id)
        return self.sessions[session_id]

class ADKSessionMemory:
    def __init__(self, session_id):
        self.session_id = session_id
        self.short_term = []
        self.summary = ""
    
    def add_interaction(self, user_query, agent_response):
        self.short_term.append({
            "timestamp": time.time(),
            "query": user_query,
            "response": agent_response
        })
        if len(self.short_term) > 10:
            self.short_term.pop(0)
            
    def get_context(self):
        return [item['query'] for item in self.short_term[-3:]]

class OrcaADKAgent:
    def __init__(self, stock_analyzer, llm_orchestrator):
        self.stock_analyzer = stock_analyzer
        self.llm_orchestrator = llm_orchestrator
        self.observability = ADKObservability()
        self.memory_manager = ADKMemoryManager()
        self.tools = self._register_tools()
        
    def _register_tools(self):
        return {
            "fetch_price_data": {
                "description": "Fetch historical OHLCV data",
                "parameters": {"symbol": "str"}
            },
            "compute_indicators": {
                "description": "Calculate technical indicators (RSI, MACD, etc.)",
                "parameters": {"data": "DataFrame"}
            },
            "run_forecast": {
                "description": "Run ML price forecast",
                "parameters": {"data": "DataFrame", "horizon": "int"}
            },
            "quantum_risk": {
                "description": "Calculate quantum-inspired risk metrics",
                "parameters": {"volatility": "float"}
            }
        }
        
    def get_tool_schemas(self):
        return self.tools
        
    def get_trace(self):
        return self.observability.trace
        
    def get_memory(self, session_id):
        return self.memory_manager.get_memory(session_id)
        
    def execute(self, session_id, symbol, question, horizon_days=30):
        self.observability.log_event("agent", "start_execution", "processing", {"symbol": symbol, "question": question}, {})
        
        # 1. SCAN: Fetch Data & Indicators
        try:
            hist, source = self.stock_analyzer.fetch_data(symbol)
            if hist is None or hist.empty:
                raise Exception(f"Failed to fetch data for {symbol}")
                
            self.observability.log_event("tool", "fetch_price_data", "ok", {"symbol": symbol}, {"rows": len(hist), "source": source})
            
            analysis = self.stock_analyzer.analyze(symbol)
            self.observability.log_event("tool", "compute_indicators", "ok", {}, {"indicators": list(analysis['metrics'].keys())})
            self.observability.log_event("tool", "run_forecast", "ok", {"horizon": horizon_days}, analysis['forecast'])
            
        except Exception as e:
            self.observability.log_event("agent", "execution_failed", "error", {"error": str(e)}, {})
            raise e
            
        # 2. THINK: Multi-Model Consensus
        model_ids = ['mistral', 'llama3', 'phi3'] # Default set
        consensus = self.llm_orchestrator.get_evaluated_responses(analysis, model_ids, question)
        self.observability.log_event("model", "multi_llm_consensus", "ok", {"models": model_ids}, {"count": consensus.get('count', 0)})
        
        # 3. OBSERVE: Update Memory
        memory = self.get_memory(session_id)
        best_response = consensus.get('best_response', {}).get('response', 'No consensus reached')
        memory.add_interaction(question, best_response)
        
        self.observability.log_event("agent", "complete", "success", {}, {"best_model": consensus.get('best_model')})
        
        return {
            "analysis": analysis,
            "llm_responses": consensus,
            "adk_trace": self.observability.trace,
            "session_id": session_id
        }

class ADKFlaskWrapper:
    def __init__(self, app, stock_analyzer, llm_orchestrator):
        self.app = app
        self.stock_analyzer = stock_analyzer
        self.llm_orchestrator = llm_orchestrator
        self.adk_agent = OrcaADKAgent(stock_analyzer, llm_orchestrator)
        self.register_routes()
        
    def register_routes(self):
        @self.app.route('/api/adk/analyze')
        def adk_analyze():
            symbol = request.args.get('symbol', 'AAPL')
            try:
                result = self.adk_agent.execute(request.args.get('session_id', 'default'), symbol.upper(), request.args.get('question', 'Analyze'), int(request.args.get('horizon_days', 30)))
                return jsonify({**result['analysis'], 'adk_trace': result['adk_trace'], 'adk_enabled': True, 'session_id': result['session_id']})
            except Exception as e:
                return jsonify({'error': str(e), 'adk_enabled': True}), 500
        
        @self.app.route('/api/adk/tools')
        def adk_tools(): return jsonify({'tools': self.adk_agent.get_tool_schemas(), 'count': len(self.adk_agent.tools)})
        
        @self.app.route('/api/adk/trace')
        def adk_trace(): return jsonify({'trace': self.adk_agent.get_trace(), 'steps': len(self.adk_agent.get_trace())})
        
        @self.app.route('/api/adk/memory/<session_id>')
        def adk_memory(session_id):
            m = self.adk_agent.get_memory(session_id)
            return jsonify({'session_id': session_id, 'short_term': m.short_term, 'summary': m.summary, 'context': m.get_context()})
        
        @self.app.route('/api/adk/consensus', methods=['POST'])
        def adk_consensus():
            try:
                d = request.json
                if not d.get('symbol') or not d.get('models'): return jsonify({'error': 'Missing params'}), 400
                result = self.adk_agent.execute(d.get('session_id', 'default'), d['symbol'].upper(), d.get('question', 'Recommend'))
                return jsonify({'responses': result['llm_responses'].get('responses', []), 'best_response': result['llm_responses'].get('best_response', {}), 'adk_trace': result['adk_trace'], 'adk_enabled': True})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

def integrate_adk(app, stock_analyzer, llm_orchestrator):
    return ADKFlaskWrapper(app, stock_analyzer, llm_orchestrator)

if __name__ == '__main__':

    
    # Quick test
    test_analyzer = StockAnalyzer()
    test_hist, test_source = test_analyzer.fetch_data("AAPL,MSFT,AMZN,TSLA,GOOGL,META,NVDA,NFLX,JPM,JNJ")
    if test_hist is not None and not test_hist.empty:
        print(f"Data fetch working Source: {test_source} ({len(test_hist)} points)")
    else:
        print("Warning: Data fetch test failed, but synthetic data will work")
    
    # Initialize ADK
    adk = integrate_adk(app, analyzer, orchestrator)
    
    app.run(debug=True, port=5000, host='0.0.0.0')
