// ============================================================================
// QUANTUM TRADING PLATFORM - CLIENT SIDE LOGIC
// Author: Sansyuh06
// ============================================================================

let currentAnalysis = null;
let priceChart = null;
let selectedModels = new Set();
let availableModels = [];
let currentSymbol = null;
let initialPrice = 0;
let realtimeInterval = null;
let isRealtimeActive = false;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Application starting...'); // Fingers crossed

    try {
        // Load initial data in parallel to speed things up
        await loadStocks();
        await loadModels();
        checkLLMStatus();
        setupEventListeners();
        console.log('✓ Application initialized');
    } catch (error) {
        console.error('✗ Initialization error:', error);
        showError('Failed to initialize application: ' + error.message);
    }
});

// ============================================================================
// LOAD STOCKS
// ============================================================================

async function loadStocks() {
    try {
        const response = await fetch('/api/stocks');
        const data = await response.json();

        const select = document.getElementById('stock-select');
        select.innerHTML = '<option value="">Choose a stock...</option>';

        data.stocks.forEach(stock => {
            const option = document.createElement('option');
            option.value = stock.symbol;
            option.textContent = `${stock.symbol} - ${stock.name}`;
            select.appendChild(option);
        });

        console.log('✓ Stocks loaded');
    } catch (error) {
        console.error('✗ Error loading stocks:', error);
    }
}

// ============================================================================
// LOAD AI MODELS
// ============================================================================

async function loadModels() {
    try {
        const response = await fetch('/api/llm/models');
        const data = await response.json();

        if (data.generators) {
            availableModels = data.generators;
            renderModelList(data.generators);
            console.log('✓ AI models loaded');
        }
    } catch (error) {
        console.error('✗ Error loading models:', error);
    }
}

function renderModelList(models) {
    const container = document.getElementById('model-list');
    if (!container) return;

    container.innerHTML = '';

    models.forEach(model => {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.dataset.modelId = model.id;

        card.innerHTML = `
            <div class="model-checkbox"></div>
            <div style="flex: 1;">
                <div class="model-name">${model.label}</div>
                <div class="model-style">${model.strength}</div>
            </div>
        `;

        card.addEventListener('click', () => toggleModel(model.id, card));
        container.appendChild(card);
    });
}

function toggleModel(modelId, card) {
    if (selectedModels.has(modelId)) {
        selectedModels.delete(modelId);
        card.classList.remove('selected');
    } else {
        selectedModels.add(modelId);
        card.classList.add('selected');
    }

    updateModelCount();
    updateConsensusButton();
}

function updateModelCount() {
    const countEl = document.getElementById('model-count');
    if (countEl) {
        countEl.textContent = `${selectedModels.size} selected`;
    }
}

function updateConsensusButton() {
    const btn = document.getElementById('consensus-btn');
    if (btn) {
        btn.disabled = selectedModels.size === 0 || !currentAnalysis;
    }
}

// ============================================================================
// CHECK LLM STATUS
// ============================================================================

async function checkLLMStatus() {
    try {
        const response = await fetch('/api/llm/check');
        const data = await response.json();

        const statusEl = document.getElementById('ai-status');
        if (statusEl) {
            if (data.available) {
                statusEl.innerHTML = '<span style="color: var(--success);">✓ AI Online</span>';
            } else {
                statusEl.innerHTML = '<span style="color: var(--warning);">AI Offline</span>';
            }
        }
    } catch (error) {
        console.error('✗ LLM status check failed:', error);
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    const stockSelect = document.getElementById('stock-select');
    if (stockSelect) {
        stockSelect.addEventListener('change', (e) => {
            const analyzeBtn = document.getElementById('analyze-btn');
            if (analyzeBtn) {
                analyzeBtn.disabled = !e.target.value;
            }
            // Stop real-time updates when changing stock to avoid data pollution
            stopRealtimeUpdates();
        });
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', handleAnalyze);
    }

    const consensusBtn = document.getElementById('consensus-btn');
    if (consensusBtn) {
        consensusBtn.addEventListener('click', handleConsensus);
    }

    const canvasBtn = document.getElementById('open-agent-btn');
    if (canvasBtn) {
        canvasBtn.addEventListener('click', () => {
            console.log('Opening AI Canvas...');
            if (typeof window.openCanvas === 'function') {
                window.currentAnalysis = currentAnalysis;
                window.openCanvas();
            } else {
                console.error('Canvas function not available');
                showError('AI Canvas feature is not available');
            }
        });
    }

    // Stop real-time updates when page is hidden
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopRealtimeUpdates();
        } else if (currentSymbol && isRealtimeActive) {
            startRealtimeUpdates();
        }
    });

    // Stop updates before page unload
    window.addEventListener('beforeunload', () => {
        stopRealtimeUpdates();
    });
}

// ============================================================================
// ANALYZE STOCK
// ============================================================================

async function handleAnalyze() {
    const select = document.getElementById('stock-select');
    const symbol = select.value;

    if (!symbol) {
        showError('Please select a stock');
        return;
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
    }

    try {
        console.log(`📊 Analyzing ${symbol}...`);
        const response = await fetch(`/api/analyze/${symbol}`);
        const data = await response.json();

        if (data.error) {
            console.error('✗ Analysis error:', data.error);
            showError(data.error);
            return;
        }

        console.log('✓ Analysis complete:', data.recommendation);

        // Check if real data was used
        if (data.data_source === 'real_market_data') {
            console.log(`✓ Using REAL market data (${data.data_points} points)`);
            console.log(`✓ Date range: ${data.date_range}`);
        } else {
            console.warn('⚠ Real data not available');
        }

        currentAnalysis = data;
        window.currentAnalysis = data;
        currentSymbol = symbol;
        initialPrice = data.metrics.current_price;

        displayAnalysis(data);
        updateConsensusButton();

        const canvasBtnContainer = document.getElementById('agent-btn-container');
        if (canvasBtnContainer) {
            canvasBtnContainer.classList.remove('hidden');
        }

        // Start real-time updates
        startRealtimeUpdates();
        showSuccess(`Real-time updates activated for ${symbol}`);

    } catch (error) {
        console.error('✗ Analysis error:', error);
        showError('Analysis failed: ' + error.message);
    } finally {
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Stock';
        }
    }
}

// ============================================================================
// DISPLAY ANALYSIS
// ============================================================================

function displayAnalysis(data) {
    console.log('📈 Displaying analysis results...');

    const emptyState = document.getElementById('empty-state');
    if (emptyState) emptyState.classList.add('hidden');

    const analysisContent = document.getElementById('analysis-content');
    if (analysisContent) analysisContent.classList.remove('hidden');

    const metricsGrid = document.getElementById('metrics-grid');
    if (metricsGrid) metricsGrid.classList.remove('hidden');

    const forecastPanel = document.getElementById('forecast-panel');
    if (forecastPanel) forecastPanel.classList.remove('hidden');

    const chartTitle = document.getElementById('chart-title');
    if (chartTitle) {
        chartTitle.textContent = `${data.symbol} - ${data.company.name}`;
    }

    const currentPrice = document.getElementById('current-price');
    if (currentPrice) {
        currentPrice.textContent = `$${data.metrics.current_price.toFixed(2)}`;
    }

    const priceChange = document.getElementById('price-change');
    if (priceChange) {
        const change = data.metrics.total_return;
        priceChange.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
        priceChange.className = 'price-change ' + (change >= 0 ? 'positive' : 'negative');
    }

    const recBadge = document.getElementById('recommendation');
    if (recBadge) {
        recBadge.textContent = data.recommendation;
        recBadge.className = 'rec-badge';

        if (data.recommendation.includes('BUY')) {
            recBadge.classList.add('rec-buy');
        } else if (data.recommendation.includes('SELL')) {
            recBadge.classList.add('rec-sell');
        } else {
            recBadge.classList.add('rec-hold');
        }
    }

    updateForecast(data.forecast);
    updateMetrics(data.metrics, data.forecast);
    updateRecommendation(data);
    updateSignals(data.signals);
    updateNews(data.news);
    updateChart(data.chart_data, data.symbol);

    console.log('✓ Display complete');
}

function updateForecast(forecast) {
    const directionEl = document.getElementById('forecast-direction');
    const detailsEl = document.getElementById('forecast-details');

    if (directionEl) {
        directionEl.className = 'forecast-direction';
        if (forecast.direction === 'UP') {
            directionEl.classList.add('up');
            directionEl.textContent = '↗ ' + forecast.direction;
        } else if (forecast.direction === 'DOWN') {
            directionEl.classList.add('down');
            directionEl.textContent = '↘ ' + forecast.direction;
        } else {
            directionEl.textContent = forecast.direction;
        }
    }

    if (detailsEl) {
        detailsEl.textContent = forecast.details;
    }
}

function updateMetrics(metrics, forecast) {
    document.getElementById('metric-price').textContent = `$${metrics.current_price.toFixed(2)}`;

    const returnEl = document.getElementById('metric-return');
    returnEl.textContent = `${metrics.total_return.toFixed(1)}%`;
    returnEl.className = 'metric-value';
    if (metrics.total_return > 0) returnEl.classList.add('positive');
    else if (metrics.total_return < 0) returnEl.classList.add('negative');

    document.getElementById('metric-volatility').textContent = `${metrics.volatility.toFixed(1)}%`;
    document.getElementById('metric-rsi').textContent = metrics.rsi.toFixed(1);
    document.getElementById('metric-sharpe').textContent = metrics.sharpe_ratio.toFixed(2);

    if (forecast.accuracy) {
        document.getElementById('metric-accuracy').textContent = `${forecast.accuracy}%`;
    }

    document.getElementById('metric-quantum-risk').textContent = `${metrics.quantum_risk.toFixed(1)}%`;
    document.getElementById('metric-quantum-trade').textContent = `${metrics.quantum_trade_prob.toFixed(1)}%`;
    document.getElementById('metric-drawdown').textContent = `${metrics.max_drawdown.toFixed(1)}%`;
}

function updateRecommendation(data) {
    document.getElementById('rec-score').textContent = data.score;
    document.getElementById('rec-confidence').textContent = data.confidence;
    document.getElementById('rec-risk').textContent = data.risk_level;
}

function updateSignals(signals) {
    const container = document.getElementById('signals-list');
    if (!container) return;

    if (!signals || signals.length === 0) {
        container.innerHTML = '<div class="empty-state" style="padding: 20px;">No signals yet</div>';
        return;
    }

    container.innerHTML = '';

    signals.forEach(signal => {
        const item = document.createElement('div');
        item.className = 'signal-item';
        const [text, type] = signal;
        item.innerHTML = `
            <div class="signal-dot ${type}"></div>
            <div class="signal-text">${text}</div>
        `;
        container.appendChild(item);
    });
}

function updateNews(news) {
    const container = document.getElementById('news-content');
    if (!container) return;

    if (!news || !news.headlines || news.headlines.length === 0) {
        container.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">No recent news</div>';
        return;
    }

    let html = `<div class="sentiment-badge sentiment-${news.sentiment}">${news.sentiment} (${news.score})</div>`;
    news.headlines.forEach(headline => {
        html += `<div class="news-headline">${headline}</div>`;
    });
    container.innerHTML = html;
}

// ============================================================================
// CHART
// ============================================================================

function updateChart(chartData, symbol) {
    const canvas = document.getElementById('main-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    const datasets = [
        {
            label: 'Price',
            data: chartData.prices,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            borderWidth: 2,
            tension: 0.4,
            fill: true,
            pointRadius: 0
        },
        {
            label: 'MA20',
            data: chartData.ma20,
            borderColor: '#f59e0b',
            borderWidth: 2,
            tension: 0.4,
            fill: false,
            pointRadius: 0
        },
        {
            label: 'MA50',
            data: chartData.ma50,
            borderColor: '#8b5cf6',
            borderWidth: 2,
            tension: 0.4,
            fill: false,
            pointRadius: 0
        }
    ];

    if (chartData.predictions && chartData.predictions.length > 0) {
        const predictionData = new Array(chartData.prices.length).fill(null);
        predictionData[predictionData.length - 1] = chartData.prices[chartData.prices.length - 1];

        datasets.push({
            label: 'ML Forecast',
            data: [...predictionData, ...chartData.predictions],
            borderColor: '#fbbf24',
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.4,
            fill: false,
            pointRadius: 0
        });
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.dates,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { size: 11 },
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(20, 24, 36, 0.95)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#94a3b8',
                    borderColor: '#2a303c',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(30, 36, 51, 0.5)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: { size: 10 }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(30, 36, 51, 0.5)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: { size: 10 },
                        callback: function (value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });

    console.log('✓ Chart updated');
}

// ============================================================================
// REAL-TIME UPDATES
// ============================================================================

function startRealtimeUpdates() {
    // Stop any existing interval
    stopRealtimeUpdates();

    if (!currentSymbol) return;

    console.log('🔴 Starting real-time updates for', currentSymbol);
    isRealtimeActive = true;

    // Show streaming indicator
    const indicator = document.getElementById('streaming-indicator');
    if (indicator) {
        indicator.classList.add('active');
    }

    // Update every 2 seconds
    realtimeInterval = setInterval(async () => {
        await updateRealtimePrice();
    }, 2000);
}

function stopRealtimeUpdates() {
    if (realtimeInterval) {
        clearInterval(realtimeInterval);
        realtimeInterval = null;
    }

    isRealtimeActive = false;

    const indicator = document.getElementById('streaming-indicator');
    if (indicator) {
        indicator.classList.remove('active');
    }

    console.log('⏹ Real-time updates stopped');
}

async function updateRealtimePrice() {
    if (!currentSymbol || !priceChart) return;

    try {
        const response = await fetch(`/api/realtime/${currentSymbol}`);
        const data = await response.json();

        if (data.error) {
            console.warn('Real-time update failed:', data.error);
            return;
        }

        const newPrice = data.price;

        // Update current price display
        const currentPriceEl = document.getElementById('current-price');
        const metricPriceEl = document.getElementById('metric-price');

        if (currentPriceEl) {
            currentPriceEl.textContent = `$${newPrice.toFixed(2)}`;
        }

        if (metricPriceEl) {
            metricPriceEl.textContent = `$${newPrice.toFixed(2)}`;
        }

        // Calculate price change
        if (initialPrice > 0) {
            const change = ((newPrice - initialPrice) / initialPrice) * 100;
            const priceChangeEl = document.getElementById('price-change');

            if (priceChangeEl) {
                priceChangeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                priceChangeEl.className = 'price-change ' + (change >= 0 ? 'positive' : 'negative');
            }

            const metricReturnEl = document.getElementById('metric-return');
            if (metricReturnEl) {
                metricReturnEl.textContent = `${change.toFixed(1)}%`;
                metricReturnEl.className = 'metric-value';
                if (change > 0) metricReturnEl.classList.add('positive');
                else if (change < 0) metricReturnEl.classList.add('negative');
            }
        }

        // Add new price point to chart
        const chartData = priceChart.data;
        const now = new Date().toLocaleTimeString();

        // Add new data point
        chartData.labels.push(now);
        chartData.datasets[0].data.push(newPrice);

        // Keep only last 90 points
        if (chartData.labels.length > 90) {
            chartData.labels.shift();
            chartData.datasets.forEach(dataset => {
                if (dataset.data.length > 90) {
                    dataset.data.shift();
                }
            });
        }

        // Update moving averages
        const prices = chartData.datasets[0].data;
        if (prices.length >= 20) {
            const ma20 = prices.slice(-20).reduce((a, b) => a + b, 0) / 20;
            chartData.datasets[1].data.push(ma20);
            if (chartData.datasets[1].data.length > chartData.datasets[0].data.length) {
                chartData.datasets[1].data.shift();
            }
        }

        if (prices.length >= 50) {
            const ma50 = prices.slice(-50).reduce((a, b) => a + b, 0) / 50;
            chartData.datasets[2].data.push(ma50);
            if (chartData.datasets[2].data.length > chartData.datasets[0].data.length) {
                chartData.datasets[2].data.shift();
            }
        }

        // Update chart with smooth animation
        // Note: 'none' mode is crucial here for performance, otherwise it lags on updates
        priceChart.update('none');

        console.log(`📊 Updated: ${currentSymbol} = $${newPrice.toFixed(2)}`);

    } catch (error) {
        console.error('✗ Real-time update error:', error);
    }
}

// ============================================================================
// AI CONSENSUS
// ============================================================================

async function handleConsensus() {
    if (!currentAnalysis || selectedModels.size === 0) {
        showError('Please select models and analyze a stock first');
        return;
    }

    const btn = document.getElementById('consensus-btn');
    const loadingEl = document.getElementById('llm-loading');
    const responsesContainer = document.getElementById('responses-container');

    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Processing...';
    }

    if (loadingEl) {
        loadingEl.classList.add('active');
    }

    if (responsesContainer) {
        responsesContainer.innerHTML = '';
    }

    try {
        console.log('🤖 Getting AI consensus...');
        const response = await fetch('/api/llm/consensus', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: currentAnalysis.symbol,
                models: Array.from(selectedModels)
            })
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            displayConsensusResults(data);
            console.log('✓ AI consensus received');
        }

    } catch (error) {
        console.error('✗ Consensus error:', error);
        showError('AI Consensus failed: ' + error.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Get AI Consensus';
        }
        if (loadingEl) {
            loadingEl.classList.remove('active');
        }
    }
}

function displayConsensusResults(data) {
    const container = document.getElementById('responses-container');
    if (!container) return;

    container.innerHTML = '';

    if (!data.responses || data.responses.length === 0) {
        container.innerHTML = '<div class="empty-state">No responses received</div>';
        return;
    }

    data.responses.forEach(response => {
        const card = document.createElement('div');
        card.className = 'response-card';
        card.innerHTML = `
            <div class="response-header">
                <div class="response-model">${response.label}</div>
                <div class="response-time">${response.time_ms}ms</div>
            </div>
            <div class="response-text">${response.response}</div>
        `;
        container.appendChild(card);
    });
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    console.error('❌ Error:', message);
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: rgba(239, 68, 68, 0.95);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        z-index: 10000;
        max-width: 400px;
        font-size: 14px;
        font-weight: 600;
        animation: slideIn 0.3s ease-out;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.transition = 'opacity 0.3s';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function showSuccess(message) {
    console.log('✅ Success:', message);
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: rgba(16, 185, 129, 0.95);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        z-index: 10000;
        max-width: 400px;
        font-size: 14px;
        font-weight: 600;
        animation: slideIn 0.3s ease-out;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.transition = 'opacity 0.3s';
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

console.log('✓ App.js loaded with real-time features');