"""
FinSight-X ADK Integration
Wraps the existing system in Google's Agent Development Kit (ADK) architecture.
This allows us to treat the whole app as a "Glass Box" agent.
"""
import os, time, json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
        
        analysis = scene["full_analysis"]["data"] if scene["full_analysis"]["status"] == "success" else {}
        
        # THINK
        llm_resp = {}
        if hasattr(self.llm_orchestrator, 'GENERATOR_MODELS'):
            st = time.time()
            self.observability.log_event("think", "model", "multi_llm_panel", "processing", {"n_models": len(self.llm_orchestrator.GENERATOR_MODELS)}, {})
            llm_resp = self.llm_orchestrator.get_consensus_with_evaluation(analysis, list(self.llm_orchestrator.GENERATOR_MODELS.keys()))
            self.observability.log_event("think", "model", "multi_llm_panel", "complete", {}, {"n_responses": len(llm_resp.get("responses", []))}, duration_ms=(time.time()-st)*1000)
        
        # ACT
        st = time.time()
        self.observability.log_event("act", "judge", "evaluate", "processing", {}, {})
        best = llm_resp.get("best_response", {})
        self.observability.log_event("act", "judge", "evaluate", "complete", {}, {"best": best.get("model_id", "")}, duration_ms=(time.time()-st)*1000, model="judge")
        
        # OBSERVE
        st = time.time()
        self.observability.log_event("observe", "agent", "collect", "processing", {}, {})
        result = {"symbol": symbol, "question": question, "analysis": analysis, "llm_responses": llm_resp, "scene_data": scene, "adk_trace": self.observability.get_trace(), "session_id": session_id, "timestamp": time.time()}
        memory.add_interaction(symbol, question, result)
        self.observability.log_event("observe", "agent", "collect", "complete", {}, {}, duration_ms=(time.time()-st)*1000)
        return result
    
    def get_tool_schemas(self): return [t.to_schema() for t in self.tools.values()]
    def get_trace(self): return self.observability.get_trace()
    def get_memory(self, sid): return self.memory_manager.get_or_create(sid)

class ADKFlaskWrapper:
    def __init__(self, app, stock_analyzer, llm_orchestrator):
        self.app, self.stock_analyzer, self.llm_orchestrator = app, stock_analyzer, llm_orchestrator
        self.adk_agent = FinSightADKAgent(stock_analyzer, llm_orchestrator, os.environ.get('GOOGLE_API_KEY'))
        self._register_routes()
        print("\n" + "="*60)
        print("ADK INTEGRATED - GLASS BOX MODE ACTIVE")
        print("="*60)
    
    def _register_routes(self):
        @self.app.route('/api/adk/analyze/<symbol>', methods=['GET', 'POST'])
        def adk_analyze(symbol):
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
    """Add to app.py: adk = integrate_adk(app, analyzer, orchestrator)"""
    return ADKFlaskWrapper(app, stock_analyzer, llm_orchestrator)
