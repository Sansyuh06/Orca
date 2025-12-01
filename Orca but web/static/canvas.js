// ============================================================================
// AI CANVAS - Visual Workflow Builder
// Inspired by n8n/Flowise but simpler.
// Note: This uses React via Babel Standalone for simplicity in this demo.
// ============================================================================

const { useState, useRef, useEffect } = React;

// SVG Icon Components
const MessageSquare = () => (
    React.createElement('svg', { width: 16, height: 16, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2 },
        React.createElement('path', { d: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z' })
    )
);

const ArrowRight = ({ size = 16 }) => (
    React.createElement('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2 },
        React.createElement('line', { x1: 5, y1: 12, x2: 19, y2: 12 }),
        React.createElement('polyline', { points: '12 5 19 12 12 19' })
    )
);

const Trash2 = ({ size = 16 }) => (
    React.createElement('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2 },
        React.createElement('polyline', { points: '3 6 5 6 21 6' }),
        React.createElement('path', { d: 'M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2' })
    )
);

const Play = ({ size = 16 }) => (
    React.createElement('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2 },
        React.createElement('polygon', { points: '5 3 19 12 5 21 5 3' })
    )
);

const X = ({ size = 16 }) => (
    React.createElement('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2 },
        React.createElement('line', { x1: 18, y1: 6, x2: 6, y2: 18 }),
        React.createElement('line', { x1: 6, y1: 6, x2: 18, y2: 18 })
    )
);

const Zap = ({ size = 16 }) => (
    React.createElement('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2 },
        React.createElement('polygon', { points: '13 2 3 14 12 14 11 22 21 10 12 10 13 2' })
    )
);

const Bot = ({ size = 16 }) => (
    React.createElement('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2 },
        React.createElement('rect', { x: 3, y: 11, width: 18, height: 10, rx: 2 }),
        React.createElement('circle', { cx: 12, cy: 5, r: 2 }),
        React.createElement('path', { d: 'M12 7v4' }),
        React.createElement('line', { x1: 8, y1: 16, x2: 8, y2: 16 }),
        React.createElement('line', { x1: 16, y1: 16, x2: 16, y2: 16 })
    )
);

// Main Canvas Component
const QuantumTradingCanvas = ({ currentAnalysis, onClose }) => {
    const [nodes, setNodes] = useState([]);
    const [connections, setConnections] = useState([]);
    const [draggingNode, setDraggingNode] = useState(null);
    const [connectingFrom, setConnectingFrom] = useState(null);
    const [selectedNode, setSelectedNode] = useState(null);
    const [chatMessages, setChatMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isExecuting, setIsExecuting] = useState(false);
    const [evaluationResults, setEvaluationResults] = useState(null);
    const [showScores, setShowScores] = useState(false);
    const canvasRef = useRef(null);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

    const models = [
        { id: 'mistral', name: 'Mistral 7B', color: '#06b6d4' },
        { id: 'phi3', name: 'Phi-3 Mini', color: '#8b5cf6' },
        { id: 'llama3', name: 'Llama 3.1', color: '#ec4899' },
        { id: 'gemma', name: 'Gemma 7B', color: '#10b981' },
        { id: 'qwen2', name: 'Qwen2 7B', color: '#f59e0b' }
    ];

    useEffect(() => {
        if (currentAnalysis && chatMessages.length === 0) {
            const welcomeMsg = {
                id: Date.now(),
                role: 'system',
                content: `Analysis loaded: ${currentAnalysis.symbol}\nData points: ${currentAnalysis.data_points || 'N/A'}\nDate range: ${currentAnalysis.date_range || 'N/A'}\nRecommendation: ${currentAnalysis.recommendation}\nScore: ${currentAnalysis.score}/100`
            };
            setChatMessages([welcomeMsg]);
        }
    }, [currentAnalysis]);

    const addNode = (type, config = {}) => {
        const existingOfType = nodes.filter(n => n.type === type);

        if (type === 'chat' && existingOfType.length > 0) return;
        if (type === 'evaluator' && existingOfType.length > 0) return;
        if (type === 'quantum' && existingOfType.length > 0) return;

        const newNode = {
            id: `${type}-${Date.now()}`,
            type: type,
            x: 200 + (nodes.length * 40),
            y: 150 + (nodes.length * 40),
            status: 'idle',
            ...config
        };

        // TODO: Add collision detection so nodes don't overlap
        setNodes([...nodes, newNode]);
    };

    const handleMouseDown = (e, nodeId) => {
        e.preventDefault();
        e.stopPropagation();
        const node = nodes.find(n => n.id === nodeId);
        if (node && canvasRef.current) {
            const rect = canvasRef.current.getBoundingClientRect();
            setDraggingNode(nodeId);
            setDragOffset({
                x: e.clientX - rect.left - node.x,
                y: e.clientY - rect.top - node.y
            });
        }
    };

    const handleMouseMove = (e) => {
        if (!draggingNode || !canvasRef.current) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const newX = Math.max(0, Math.min(rect.width - 240, e.clientX - rect.left - dragOffset.x));
        const newY = Math.max(0, Math.min(rect.height - 120, e.clientY - rect.top - dragOffset.y));

        setNodes(prevNodes => prevNodes.map(node =>
            node.id === draggingNode ? { ...node, x: newX, y: newY } : node
        ));
    };

    const handleMouseUp = () => {
        setDraggingNode(null);
        // Snap to grid? Maybe later.
    };

    const handlePortClick = (e, nodeId, isOutput) => {
        e.stopPropagation();

        if (isOutput) {
            setConnectingFrom(nodeId);
        } else {
            if (connectingFrom && connectingFrom !== nodeId) {
                const fromNode = nodes.find(n => n.id === connectingFrom);
                const toNode = nodes.find(n => n.id === nodeId);

                const validConnection =
                    (fromNode.type === 'model' && toNode.type === 'chat') ||
                    (fromNode.type === 'chat' && (toNode.type === 'evaluator' || toNode.type === 'quantum')) ||
                    (fromNode.type === 'quantum' && toNode.type === 'chat');

                if (validConnection) {
                    const exists = connections.some(c => c.from === connectingFrom && c.to === nodeId);
                    if (!exists) {
                        setConnections([...connections, {
                            id: `conn-${Date.now()}`,
                            from: connectingFrom,
                            to: nodeId
                        }]);
                    }
                }
            }
            setConnectingFrom(null);
        }
    };

    const deleteNode = (nodeId) => {
        setNodes(nodes.filter(n => n.id !== nodeId));
        setConnections(connections.filter(c => c.from !== nodeId && c.to !== nodeId));
        if (selectedNode === nodeId) setSelectedNode(null);
    };

    const executeWorkflow = async () => {
        setIsExecuting(true);

        const modelNodes = nodes.filter(n => n.type === 'model');
        setNodes(nodes.map(node =>
            node.type === 'model' ? { ...node, status: 'processing' } : node
        ));

        await new Promise(resolve => setTimeout(resolve, 1500));

        setNodes(nodes.map(node =>
            node.type === 'model' ? { ...node, status: 'complete' } : node
        ));

        const quantumNode = nodes.find(n => n.type === 'quantum');
        if (quantumNode && currentAnalysis) {
            setNodes(nodes.map(node =>
                node.type === 'quantum' ? { ...node, status: 'processing' } : node
            ));

            await new Promise(resolve => setTimeout(resolve, 1000));

            const quantumMsg = {
                id: Date.now(),
                role: 'quantum',
                content: `Quantum Analysis Results:\nTrade Success: ${currentAnalysis.metrics.quantum_trade_prob.toFixed(1)}%\nRisk Level: ${currentAnalysis.metrics.quantum_risk.toFixed(1)}%\nForecast: ${currentAnalysis.forecast.direction} ${currentAnalysis.forecast.expected_change}%\nConfidence: ${currentAnalysis.forecast.confidence}%`
            };
            setChatMessages(prev => [...prev, quantumMsg]);

            setNodes(nodes.map(node =>
                node.type === 'quantum' ? { ...node, status: 'complete' } : node
            ));
        }

        setIsExecuting(false);
    };



    // TODO: Clean up this massive function
    const sendMessage = () => {
        if (!inputMessage.trim()) return;

        const userMsg = { id: Date.now(), role: 'user', content: inputMessage };
        setChatMessages([...chatMessages, userMsg]);
        setInputMessage('');

        // Send question to backend LLM chat endpoint with connected models
        (async () => {
            try {
                const connectedModels = connections
                    .filter(c => c.to.startsWith('chat-'))
                    .map(c => nodes.find(n => n.id === c.from))
                    .filter(n => n && n.type === 'model')
                    .map(n => n.modelId)
                    .filter(Boolean);

                if (connectedModels.length === 0 || !currentAnalysis) return;

                const payload = {
                    symbol: currentAnalysis.symbol,
                    models: connectedModels,
                    question: userMsg.content
                };

                // Use evaluation endpoint which returns per-model responses and scores
                const resp = await fetch('/api/llm/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await resp.json();
                if (data.error) {
                    const errMsg = { id: Date.now() + Math.random(), role: 'system', content: 'LLM error: ' + data.error };
                    setChatMessages(prev => [...prev, errMsg]);
                    return;
                }

                // Data contains evaluated responses and scores
                // Example: { responses: [{model_id, label, response, score, is_best}], scores: {...}, best_model }
                const evaluated = data || {};

                // Normalize responses into a consistent array
                const responsesArr = (evaluated.responses || []).map(r => ({
                    model_id: r.model_id || r.model || r.id,
                    label: r.label || r.name || r.model || (r.model_id || r.model),
                    response: r.response || r.response_text || r.text || '',
                    score: typeof r.score !== 'undefined' ? Number(r.score) : null,
                    is_best: !!r.is_best
                }));

                // If evaluator provided scores object but responses lack scores, merge them
                if (evaluated.scores && responsesArr.length > 0) {
                    responsesArr.forEach(rr => {
                        if ((rr.score === null || isNaN(rr.score)) && evaluated.scores[rr.model_id]) {
                            rr.score = Number(evaluated.scores[rr.model_id]);
                        }
                    });
                }

                // Determine best model: prefer evaluator.best_model, else look for is_best flag, else highest score, else first
                let bestModel = evaluated.best_model || null;
                if (!bestModel) {
                    const flagged = responsesArr.find(r => r.is_best);
                    if (flagged) bestModel = flagged.model_id;
                }
                if (!bestModel) {
                    const byScore = responsesArr.filter(r => typeof r.score === 'number').sort((a, b) => b.score - a.score);
                    if (byScore.length) bestModel = byScore[0].model_id;
                }
                if (!bestModel && responsesArr.length) bestModel = responsesArr[0].model_id;

                // Append only the best model's response to the visible chat (final answer)
                const bestResp = responsesArr.find(r => r.model_id === bestModel);
                if (bestResp) {
                    const assistantMsg = {
                        id: Date.now() + Math.random(),
                        role: 'assistant',
                        content: bestResp.response,
                        model: { id: bestResp.model_id, name: bestResp.label },
                        score: bestResp.score,
                        is_best: true
                    };
                    setChatMessages(prev => [...prev, assistantMsg]);
                }

                // Store full evaluation results for dropdown and later inspection
                const scoresObj = evaluated.scores || responsesArr.reduce((acc, r) => { if (typeof r.score === 'number') acc[r.model_id] = r.score; return acc; }, {});
                setEvaluationResults({ responses: responsesArr, scores: scoresObj, best_model: bestModel, evaluator: evaluated.evaluator || 'DeepSeek-R1', reasoning: evaluated.reasoning });
                setShowScores(true);
            } catch (err) {
                const errMsg = { id: Date.now() + Math.random(), role: 'system', content: 'LLM request failed: ' + err.message };
                setChatMessages(prev => [...prev, errMsg]);
            }
        })();
    };

    // When user clicks a model in the dropdown, show that model's full response in chat
    const showModelResponse = (modelId) => {
        if (!evaluationResults || !evaluationResults.responses) return;
        const r = evaluationResults.responses.find(x => x.model_id === modelId);
        if (!r) return;
        const msg = {
            id: Date.now() + Math.random(),
            role: 'assistant',
            content: r.response,
            model: { id: r.model_id, name: r.label },
            score: r.score
        };
        setChatMessages(prev => [...prev, msg]);
        setShowScores(false);
    };

    const getNodePosition = (nodeId) => {
        const node = nodes.find(n => n.id === nodeId);
        return node ? { x: node.x + 100, y: node.y + 40 } : null;
    };

    const renderNode = (node) => {
        const nodeConfig = {
            chat: { label: 'Chat Interface', color: '#06b6d4', hasInput: true, hasOutput: true },
            model: { label: node.label || 'AI Model', color: node.color || '#8b5cf6', hasInput: false, hasOutput: true },
            evaluator: { label: 'DeepSeek-R1', color: '#ef4444', hasInput: true, hasOutput: false },
            quantum: { label: 'Quantum Predictor', color: '#8b5cf6', hasInput: true, hasOutput: true }
        };

        const config = nodeConfig[node.type];
        const isSelected = selectedNode === node.id;
        const statusColor = node.status === 'processing' ? '#f59e0b' : node.status === 'complete' ? '#10b981' : '#64748b';

        return (
            <div
                key={node.id}
                style={{
                    position: 'absolute',
                    left: `${node.x}px`,
                    top: `${node.y}px`,
                    zIndex: draggingNode === node.id ? 1000 : 10
                }}
            >
                <div style={{
                    width: '200px',
                    background: '#171a21',
                    border: `2px solid ${isSelected ? config.color : '#1f2937'}`,
                    borderRadius: '10px',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                    transition: 'all 0.15s'
                }}>
                    {/* Header */}
                    <div
                        style={{
                            padding: '10px',
                            background: `${config.color}20`,
                            borderBottom: '1px solid #2a303c',
                            cursor: draggingNode === node.id ? 'grabbing' : 'grab',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            borderRadius: '10px 10px 0 0',
                            userSelect: 'none'
                        }}
                        onMouseDown={(e) => handleMouseDown(e, node.id)}
                        onClick={() => setSelectedNode(node.id === selectedNode ? null : node.id)}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <div style={{
                                width: '8px',
                                height: '8px',
                                borderRadius: '50%',
                                background: statusColor,
                                boxShadow: `0 0 8px ${statusColor}`
                            }} />
                            <span style={{ fontSize: '13px', fontWeight: '600', color: '#e2e8f0' }}>
                                {config.label}
                            </span>
                        </div>
                        <button
                            onClick={(e) => { e.stopPropagation(); deleteNode(node.id); }}
                            style={{
                                background: 'none',
                                border: 'none',
                                color: '#9ca3af',
                                cursor: 'pointer',
                                padding: '2px',
                                display: 'flex'
                            }}
                        >
                            <Trash2 size={14} />
                        </button>
                    </div>

                    {/* Body */}
                    <div style={{ padding: '10px' }}>
                        <div style={{ fontSize: '10px', color: '#9ca3af', lineHeight: '1.5' }}>
                            {node.type === 'chat' && 'Central hub for AI communication'}
                            {node.type === 'model' && `AI model: ${node.label}`}
                            {node.type === 'evaluator' && 'Evaluates and ranks responses'}
                            {node.type === 'quantum' && 'Quantum risk & forecast analysis'}
                        </div>
                        {node.status === 'processing' && (
                            <div style={{ marginTop: '6px', fontSize: '10px', color: '#f59e0b' }}>
                                Processing...
                            </div>
                        )}
                        {node.status === 'complete' && (
                            <div style={{ marginTop: '6px', fontSize: '10px', color: '#10b981' }}>
                                Complete
                            </div>
                        )}
                    </div>

                    {/* Ports */}
                    {config.hasOutput && (
                        <div
                            style={{
                                position: 'absolute',
                                right: '-10px',
                                top: '50%',
                                transform: 'translateY(-50%)',
                                width: '18px',
                                height: '18px',
                                borderRadius: '50%',
                                background: config.color,
                                border: '2px solid #0f1419',
                                cursor: 'pointer',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                                zIndex: 10
                            }}
                            onClick={(e) => handlePortClick(e, node.id, true)}
                        />
                    )}
                    {config.hasInput && (
                        <div
                            style={{
                                position: 'absolute',
                                left: '-10px',
                                top: '50%',
                                transform: 'translateY(-50%)',
                                width: '18px',
                                height: '18px',
                                borderRadius: '50%',
                                background: config.color,
                                border: '2px solid #0f1419',
                                cursor: 'pointer',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                                zIndex: 10
                            }}
                            onClick={(e) => handlePortClick(e, node.id, false)}
                        />
                    )}
                </div>
            </div>
        );
    };

    // Helper: map a model id to a friendly label
    const getModelLabel = (mid) => {
        if (!mid) return mid;
        const found = models.find(m => m.id === mid || m.name === mid || m.id === (mid.split(':')[0]));
        return found ? found.name : mid;
    };

    // Prepare sorted scores for the dropdown (highest first).
    // Fallback: derive scores from evaluationResults.responses when evaluator didn't provide a scores object.
    const sortedScores = (() => {
        if (!evaluationResults) return [];
        if (evaluationResults.scores && Object.keys(evaluationResults.scores).length > 0) {
            return Object.entries(evaluationResults.scores).map(([mid, score]) => ({ id: mid, score: Number(score) }))
                .sort((a, b) => (b.score || 0) - (a.score || 0));
        }
        if (Array.isArray(evaluationResults.responses) && evaluationResults.responses.length > 0) {
            return evaluationResults.responses.map(r => ({ id: r.model_id, score: typeof r.score === 'number' ? r.score : 0 }))
                .sort((a, b) => (b.score || 0) - (a.score || 0));
        }
        return [];
    })();

    return (
        <div style={{
            position: 'fixed',
            inset: 0,
            display: 'flex',
            background: '#0d0d0d',
            color: '#ffffff',
            fontFamily: 'Inter, sans-serif',
            zIndex: 10000
        }}>
            {/* Sidebar */}
            <div style={{
                width: '200px',
                background: '#141414',
                borderRight: '1px solid #222',
                padding: '12px',
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column'
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '16px'
                }}>
                    <h2 style={{ fontSize: '15px', fontWeight: '700', margin: 0 }}>Canvas</h2>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'none',
                            border: 'none',
                            color: '#9ca3af',
                            cursor: 'pointer',
                            padding: '4px',
                            display: 'flex'
                        }}
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Core Nodes */}
                <div style={{ marginBottom: '24px' }}>
                    <h3 style={{
                        fontSize: '11px',
                        fontWeight: '700',
                        color: '#64748b',
                        textTransform: 'uppercase',
                        marginBottom: '12px'
                    }}>Core Nodes</h3>

                    <button
                        onClick={() => addNode('chat', { label: 'Chat Interface' })}
                        style={{
                            width: '100%',
                            padding: '8px',
                            background: '#14171b',
                            border: '1px solid #222',
                            borderRadius: '8px',
                            color: '#e2e8f0',
                            cursor: 'pointer',
                            textAlign: 'left',
                            marginBottom: '8px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '10px'
                        }}
                    >
                        <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '14px', fontWeight: '600' }}>Chat Interface</div>
                            <div style={{ fontSize: '11px', color: '#9ca3af' }}>Central hub</div>
                        </div>
                    </button>
                </div>

                {/* AI Models */}
                <div style={{ marginBottom: '24px' }}>
                    <h3 style={{
                        fontSize: '11px',
                        fontWeight: '700',
                        color: '#64748b',
                        textTransform: 'uppercase',
                        marginBottom: '12px'
                    }}>AI Models</h3>

                    {models.map(model => (
                        <button
                            key={model.id}
                            onClick={() => addNode('model', {
                                label: model.name,
                                color: model.color,
                                modelId: model.id
                            })}
                            style={{
                                width: '100%',
                                padding: '8px',
                                background: '#14171b',
                                border: '1px solid #222',
                                borderRadius: '8px',
                                color: '#e2e8f0',
                                cursor: 'pointer',
                                textAlign: 'left',
                                marginBottom: '8px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px'
                            }}
                        >
                            <div style={{
                                width: '28px',
                                height: '28px',
                                borderRadius: '6px',
                                background: `${model.color}18`,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '12px',
                                color: '#e2e8f0',
                                fontWeight: 700
                            }}>{model.name.charAt(0)}</div>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '13px', fontWeight: '600' }}>{model.name}</div>
                                <div style={{ fontSize: '11px', color: '#9ca3af' }}>Generator</div>
                            </div>
                        </button>
                    ))}
                </div>

                {/* Advanced */}
                <div>
                    <h3 style={{
                        fontSize: '11px',
                        fontWeight: '700',
                        color: '#64748b',
                        textTransform: 'uppercase',
                        marginBottom: '12px'
                    }}>Advanced</h3>

                    <button
                        onClick={() => addNode('evaluator', { label: 'DeepSeek-R1' })}
                        style={{
                            width: '100%',
                            padding: '12px',
                            background: '#1a1f2e',
                            border: '1px solid #2a303c',
                            borderRadius: '8px',
                            color: '#e2e8f0',
                            cursor: 'pointer',
                            textAlign: 'left',
                            marginBottom: '8px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '12px'
                        }}
                    >
                        <div style={{
                            width: '32px',
                            height: '32px',
                            borderRadius: '8px',
                            background: '#ef444420',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '12px',
                            color: '#fff',
                            fontWeight: 700
                        }}>Judge</div>
                        <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '14px', fontWeight: '600' }}>DeepSeek-R1</div>
                            <div style={{ fontSize: '11px', color: '#9ca3af' }}>Evaluator</div>
                        </div>
                    </button>

                    <button
                        onClick={() => addNode('quantum', { label: 'Quantum Predictor' })}
                        style={{
                            width: '100%',
                            padding: '12px',
                            background: '#1a1f2e',
                            border: '1px solid #2a303c',
                            borderRadius: '8px',
                            color: '#e2e8f0',
                            cursor: 'pointer',
                            textAlign: 'left',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '12px'
                        }}
                    >
                        <div style={{
                            width: '32px',
                            height: '32px',
                            borderRadius: '8px',
                            background: '#8b5cf620',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '12px',
                            color: '#fff',
                            fontWeight: 700
                        }}>Q</div>
                        <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '14px', fontWeight: '600' }}>Quantum Predictor</div>
                            <div style={{ fontSize: '11px', color: '#9ca3af' }}>Risk analysis</div>
                        </div>
                    </button>
                </div>

                {/* Info */}
                <div style={{
                    marginTop: 'auto',
                    padding: '16px',
                    background: '#1a1f2e',
                    borderRadius: '8px',
                    border: '1px solid #2a303c',
                    display: 'none'
                }}>
                    <h4 style={{ fontSize: '13px', fontWeight: '600', marginBottom: '8px' }}>
                        Quick Guide
                    </h4>
                    <ul style={{
                        fontSize: '11px',
                        color: '#9ca3af',
                        lineHeight: '1.8',
                        listStyle: 'none',
                        padding: 0,
                        margin: 0
                    }}>
                        <li>• Drag nodes to position</li>
                        <li>• Click ports to connect</li>
                        <li>• Models → Chat → Evaluator</li>
                        <li>• Chat ↔ Quantum (predictions)</li>
                        <li>• Click Execute to run</li>
                    </ul>
                </div>
            </div>

            {/* Main Canvas */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Toolbar */}
                <div style={{
                    height: '48px',
                    background: '#141414',
                    borderBottom: '1px solid #222',
                    display: 'flex',
                    alignItems: 'center',
                    padding: '0 16px',
                    gap: '12px'
                }}>
                    <button
                        onClick={executeWorkflow}
                        disabled={isExecuting || nodes.length === 0}
                        style={{
                            padding: '8px 14px',
                            background: isExecuting || nodes.length === 0
                                ? '#2f3740'
                                : 'linear-gradient(135deg, #3b82f6, #2563eb)',
                            border: 'none',
                            borderRadius: '8px',
                            color: 'white',
                            fontWeight: '600',
                            cursor: isExecuting || nodes.length === 0 ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            fontSize: '14px'
                        }}
                    >
                        <Play size={14} />
                        {isExecuting ? 'Executing...' : 'Execute Workflow'}
                    </button>

                    <div style={{ fontSize: '12px', color: '#9ca3af' }}>
                        {nodes.length} nodes • {connections.length} connections
                    </div>

                    {currentAnalysis && (
                        <div style={{
                            marginLeft: 'auto',
                            padding: '8px 16px',
                            background: '#242424',
                            border: '1px solid #333333',
                            borderRadius: '8px',
                            fontSize: '12px',
                            fontWeight: '600',
                            color: '#4a9eff',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                        }}>
                            <Zap size={14} />
                            {currentAnalysis.symbol}: {currentAnalysis.recommendation}
                        </div>
                    )}
                </div>

                {/* Canvas Area */}
                <div
                    ref={canvasRef}
                    style={{
                        flex: 1,
                        position: 'relative',
                        background: '#0d0d0d',
                        backgroundImage: 'radial-gradient(circle at 1px 1px, #242424 1px, transparent 0)',
                        backgroundSize: '40px 40px',
                        overflow: 'hidden'
                    }}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                >
                    {/* Connection Lines */}
                    <svg style={{
                        position: 'absolute',
                        inset: 0,
                        pointerEvents: 'none',
                        width: '100%',
                        height: '100%'
                    }}>
                        {connections.map(conn => {
                            const from = getNodePosition(conn.from);
                            const to = getNodePosition(conn.to);
                            if (!from || !to) return null;

                            const fromNode = nodes.find(n => n.id === conn.from);
                            const color = fromNode?.color || '#06b6d4';

                            const midX = (from.x + to.x) / 2;
                            const path = `M ${from.x} ${from.y} Q ${midX} ${from.y}, ${to.x} ${to.y}`;

                            return (
                                <g key={conn.id}>
                                    <path
                                        d={path}
                                        stroke={color}
                                        strokeWidth="2"
                                        fill="none"
                                        opacity="0.6"
                                    />
                                    <circle cx={to.x} cy={to.y} r="4" fill={color} />
                                </g>
                            );
                        })}
                    </svg>

                    {/* Nodes */}
                    {nodes.map(node => renderNode(node))}

                    {/* Connection Preview */}
                    {connectingFrom && (
                        <div style={{
                            position: 'absolute',
                            inset: 0,
                            background: 'rgba(59, 130, 246, 0.05)',
                            border: '2px dashed rgba(59, 130, 246, 0.3)',
                            borderRadius: '8px',
                            pointerEvents: 'none'
                        }} />
                    )}

                    {/* Empty State */}
                    {nodes.length === 0 && (
                        <div style={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            textAlign: 'center',
                            color: '#64748b'
                        }}>
                            <div style={{ fontSize: '64px', marginBottom: '16px', opacity: 0.6 }}>
                                <Bot size={64} />
                            </div>
                            <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '8px' }}>
                                Empty Canvas
                            </h3>
                            <p style={{ fontSize: '14px' }}>
                                Add nodes from the sidebar to start building your workflow
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Chat Panel */}
            {selectedNode && selectedNode.startsWith('chat-') && (
                <div style={{
                    width: '320px',
                    background: '#141414',
                    borderLeft: '1px solid #222',
                    display: 'flex',
                    flexDirection: 'column'
                }}>
                    <div style={{
                        height: '48px',
                        background: '#16161a',
                        borderBottom: '1px solid #222',
                        display: 'flex',
                        alignItems: 'center',
                        padding: '0 12px',
                        justifyContent: 'space-between'
                    }}>
                        <h3 style={{ fontSize: '14px', fontWeight: '700', margin: 0 }}>
                            Chat
                        </h3>
                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <div style={{
                                padding: '4px 10px',
                                background: '#242424',
                                borderRadius: '6px',
                                fontSize: '12px',
                                fontWeight: '600',
                                color: '#4a9eff',
                                border: '1px solid #333333'
                            }}>
                                {connections.filter(c => c.to === selectedNode).length} models
                            </div>
                            {evaluationResults && (
                                <div style={{ position: 'relative' }}>
                                    <button onClick={() => setShowScores(!showScores)} style={{
                                        padding: '6px 10px',
                                        borderRadius: '6px',
                                        background: '#1f2937',
                                        color: '#ffffff',
                                        border: '1px solid #333333',
                                        cursor: 'pointer'
                                    }}>Scores</button>
                                    {showScores && (
                                        <div style={{
                                            position: 'absolute',
                                            right: 0,
                                            marginTop: '8px',
                                            background: '#1a1a1a',
                                            border: '1px solid #333333',
                                            padding: '8px',
                                            borderRadius: '8px',
                                            width: '260px',
                                            boxShadow: '0 6px 20px rgba(0,0,0,0.6)'
                                        }}>
                                            <div style={{ fontSize: '12px', color: '#b0b0b0', marginBottom: '8px' }}><strong>{evaluationResults.evaluator || 'DeepSeek-R1'}</strong> • Scores</div>
                                            {sortedScores.length === 0 && (
                                                <div style={{ fontSize: '12px', color: '#9ca3af' }}>No scores available</div>
                                            )}

                                            {sortedScores.map((entry, idx) => {
                                                const isHighest = idx === 0;
                                                const isLowest = idx === sortedScores.length - 1;
                                                const percentile = (entry.score || 0) / 10;
                                                let scoreColor = '#b0b0b0';
                                                let barColor = '#3b82f6';

                                                if (isHighest && sortedScores.length > 1) {
                                                    scoreColor = '#4ade80';
                                                    barColor = '#4ade80';
                                                } else if (isLowest && sortedScores.length > 1 && (entry.score || 0) < 5) {
                                                    scoreColor = '#ff6b6b';
                                                    barColor = '#ff6b6b';
                                                } else if (percentile >= 0.8) {
                                                    scoreColor = '#10b981';
                                                    barColor = '#10b981';
                                                } else if (percentile >= 0.6) {
                                                    scoreColor = '#fbbf24';
                                                    barColor = '#fbbf24';
                                                }

                                                return (
                                                    <div key={entry.id} onClick={() => showModelResponse(entry.id)} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 4px', gap: '8px', cursor: 'pointer', opacity: isLowest && sortedScores.length > 1 ? 0.7 : 1 }}>
                                                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                                                            <div style={{ fontSize: '13px', color: scoreColor, fontWeight: isHighest ? 800 : 600 }}>{getModelLabel(entry.id)}{isHighest ? ' 🏆' : ''}{isLowest && sortedScores.length > 1 ? ' ⚠️' : ''}</div>
                                                            <div style={{ fontSize: '11px', color: '#9ca3af' }}>{entry.id}</div>
                                                        </div>
                                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                                                            <div style={{ fontSize: '13px', color: scoreColor, fontWeight: 800 }}>{(Math.round((entry.score || 0) * 10) / 10).toFixed(1)}</div>
                                                            <div style={{ width: '80px', height: '6px', background: '#0f1316', borderRadius: '4px', marginTop: '6px', overflow: 'hidden' }}>
                                                                <div style={{ width: `${Math.min(100, Math.max(0, (entry.score || 0) * 10))}%`, height: '100%', background: barColor, boxShadow: `0 0 8px ${barColor}80` }} />
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })}

                                            {evaluationResults.reasoning && (
                                                <div style={{ marginTop: '8px', fontSize: '12px', color: '#b0b0b0' }}>{evaluationResults.reasoning}</div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Messages */}
                    <div style={{
                        flex: 1,
                        overflowY: 'auto',
                        padding: '16px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '12px'
                    }}>
                        {chatMessages.length === 0 && (
                            <div style={{
                                textAlign: 'center',
                                color: '#808080',
                                padding: '60px 20px',
                                margin: 'auto'
                            }}>
                                <p style={{ fontSize: '15px', fontWeight: '600', marginBottom: '8px' }}>
                                    No messages yet
                                </p>
                                <p style={{ fontSize: '13px' }}>
                                    Connect AI models and start chatting
                                </p>
                            </div>
                        )}

                        {chatMessages.map(msg => (
                            <div
                                key={msg.id}
                                style={{
                                    padding: '12px',
                                    borderRadius: '12px',
                                    background: msg.role === 'user'
                                        ? '#242424'
                                        : msg.role === 'quantum'
                                            ? '#242424'
                                            : msg.role === 'system'
                                                ? '#242424'
                                                : '#1a1a1a',
                                    border: '1px solid #333333'
                                }}
                            >
                                {msg.model && (
                                    <div style={{
                                        fontSize: '11px',
                                        fontWeight: '700',
                                        marginBottom: '6px',
                                        color: '#4a9eff'
                                    }}>
                                        {msg.model.name}
                                    </div>
                                )}
                                {msg.role === 'quantum' && (
                                    <div style={{
                                        fontSize: '11px',
                                        fontWeight: '700',
                                        marginBottom: '6px',
                                        color: '#4a9eff',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '4px'
                                    }}>
                                        Quantum Predictor
                                    </div>
                                )}
                                {msg.role === 'system' && (
                                    <div style={{
                                        fontSize: '11px',
                                        fontWeight: '700',
                                        marginBottom: '6px',
                                        color: '#fbbf24'
                                    }}>
                                        System
                                    </div>
                                )}
                                <div style={{
                                    fontSize: '13px',
                                    lineHeight: '1.6',
                                    color: msg.role === 'user' ? '#ffffff' : '#b0b0b0',
                                    whiteSpace: 'pre-line'
                                }}>
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Input */}
                    <div style={{
                        padding: '16px',
                        borderTop: '1px solid #2a303c',
                        background: '#1a1f2e'
                    }}>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <input
                                type="text"
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                placeholder="Ask about the analysis..."
                                style={{
                                    flex: 1,
                                    padding: '12px',
                                    background: '#0f1419',
                                    border: '1px solid #2a303c',
                                    borderRadius: '8px',
                                    color: '#e2e8f0',
                                    fontSize: '14px',
                                    fontFamily: 'Inter, sans-serif',
                                    outline: 'none'
                                }}
                            />
                            <button
                                onClick={sendMessage}
                                disabled={!inputMessage.trim()}
                                style={{
                                    padding: '12px 16px',
                                    background: inputMessage.trim()
                                        ? 'linear-gradient(135deg, #10b981, #059669)'
                                        : '#374151',
                                    border: 'none',
                                    borderRadius: '8px',
                                    cursor: inputMessage.trim() ? 'pointer' : 'not-allowed',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: 'white'
                                }}
                            >
                                <ArrowRight size={16} />
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Global function to open canvas
window.openCanvas = function () {
    console.log('Opening AI Canvas...');
    const modal = document.getElementById('canvas-modal');
    if (!modal) {
        console.error('Canvas modal element not found');
        return;
    }

    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    const root = ReactDOM.createRoot(modal);
    root.render(
        React.createElement(QuantumTradingCanvas, {
            currentAnalysis: window.currentAnalysis || null,
            onClose: () => {
                console.log('Closing AI Canvas...');
                modal.classList.remove('active');
                document.body.style.overflow = '';
                setTimeout(() => {
                    root.unmount();
                }, 300);
            }
        })
    );

    console.log('AI Canvas opened successfully');
};

console.log('Canvas.js loaded successfully');