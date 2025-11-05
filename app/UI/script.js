class HateSpeechDetectorUI {
    constructor() {
        this.selectedModel = null;
        this.detector = null;
        this.initializeUI();
        this.bindEvents();
        this.loadDetector();
    }

    initializeUI() {
        this.elements = {
            textInput: document.getElementById('textInput'),
            modelCards: document.querySelectorAll('.model-card'),
            analyzeBtn: document.getElementById('analyzeBtn'),
            compareBtn: document.getElementById('compareBtn'),
            clearBtn: document.getElementById('clearBtn'),
            resultsContainer: document.getElementById('resultsContainer'),
            batchInput: document.getElementById('batchInput'),
            batchAnalyzeBtn: document.getElementById('batchAnalyzeBtn'),
            batchResults: document.getElementById('batchResults'),
            loadingSpinner: document.getElementById('loadingSpinner')
        };
    }

    bindEvents() {
        // Model selection
        this.elements.modelCards.forEach(card => {
            card.addEventListener('click', (e) => this.selectModel(e.target.closest('.model-card')));
        });

        // Button events
        this.elements.analyzeBtn.addEventListener('click', () => this.analyzeText());
        this.elements.compareBtn.addEventListener('click', () => this.compareModels());
        this.elements.clearBtn.addEventListener('click', () => this.clearAll());
        this.elements.batchAnalyzeBtn.addEventListener('click', () => this.analyzeBatch());

        // Enter key for text input
        this.elements.textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.analyzeText();
            }
        });
    }

    async loadDetector() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            if (data.success) {
                console.log('✅ Detector loaded successfully');
                console.log('📊 Available models:', data.available_models);
            }
        } catch (error) {
            console.error('❌ Failed to load detector:', error);
            this.showError('Failed to connect to the backend. Please make sure the Flask server is running.');
        }
    }

    selectModel(card) {
        // Remove previous selection
        this.elements.modelCards.forEach(c => c.classList.remove('selected'));

        // Add selection to clicked card
        card.classList.add('selected');
        this.selectedModel = card.dataset.model;

        // Enable analyze button
        this.elements.analyzeBtn.disabled = false;
    }

    async analyzeText() {
        const text = this.elements.textInput.value.trim();

        if (!text) {
            this.showError('Please enter some text to analyze');
            return;
        }

        if (!this.selectedModel) {
            this.showError('Please select a model first');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: this.selectedModel
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displaySingleResult(data.result, this.selectedModel);
            } else {
                this.showError('Analysis failed: ' + data.error);
            }
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        }

        this.showLoading(false);
    }

    async compareModels() {
        const text = this.elements.textInput.value.trim();

        if (!text) {
            this.showError('Please enter some text to analyze');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayComparisonResults(data.results);
            } else {
                this.showError('Comparison failed: ' + data.error);
            }
        } catch (error) {
            this.showError('Comparison failed: ' + error.message);
        }

        this.showLoading(false);
    }

    async analyzeBatch() {
        const batchText = this.elements.batchInput.value.trim();

        if (!batchText) {
            this.showError('Please enter texts for batch analysis');
            return;
        }

        if (!this.selectedModel) {
            this.showError('Please select a model first');
            return;
        }

        const texts = batchText.split('\n').filter(text => text.trim());

        if (texts.length === 0) {
            this.showError('Please enter valid texts (one per line)');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/predict/batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    texts: texts,
                    model: this.selectedModel
                })
            });

            const data = await response.json();

            if (data.success) {
                const formattedResults = texts.map((text, index) => ({
                    text: text,
                    result: data.results[index]
                }));
                this.displayBatchResults(formattedResults);
            } else {
                this.showError('Batch analysis failed: ' + data.error);
            }
        } catch (error) {
            this.showError('Batch analysis failed: ' + error.message);
        }

        this.showLoading(false);
    }


    displaySingleResult(result, modelName) {
        const modelDisplayNames = {
            'logistic_regression': 'Logistic Regression',
            'naive_bayes': 'Naive Bayes',
            'random_forest': 'Random Forest',
            'distilbert': 'DistilBERT'
        };

        const modelIcons = {
            'logistic_regression': 'fas fa-chart-line',
            'naive_bayes': 'fas fa-calculator',
            'random_forest': 'fas fa-tree',
            'distilbert': 'fas fa-robot'
        };

        this.elements.resultsContainer.innerHTML = `
            <div class="result-item">
                <div class="result-header">
                    <div class="model-name">
                        <i class="${modelIcons[modelName]}"></i>
                        ${modelDisplayNames[modelName]}
                    </div>
                    <div class="prediction-badge prediction-${result.prediction === 1 ? 'hate' : 'normal'}">
                        ${result.label}
                    </div>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill confidence-${result.prediction === 1 ? 'hate' : 'normal'}"
                         style="width: ${(result.confidence * 100).toFixed(1)}%"></div>
                </div>
                <div class="probability-details">
                    <div class="prob-item">
                        <span>Normal:</span>
                        <strong>${(result.probabilities.normal * 100).toFixed(1)}%</strong>
                    </div>
                    <div class="prob-item">
                        <span>Hate Speech:</span>
                        <strong>${(result.probabilities.hate_speech * 100).toFixed(1)}%</strong>
                    </div>
                </div>
            </div>
        `;
    }

    displayComparisonResults(results) {
        const modelDisplayNames = {
            'logistic_regression': 'Logistic Regression',
            'naive_bayes': 'Naive Bayes',
            'random_forest': 'Random Forest',
            'distilbert': 'DistilBERT'
        };

        const modelIcons = {
            'logistic_regression': 'fas fa-chart-line',
            'naive_bayes': 'fas fa-calculator',
            'random_forest': 'fas fa-tree',
            'distilbert': 'fas fa-robot'
        };

        let html = '';
        for (const [modelName, result] of Object.entries(results)) {
            html += `
                <div class="result-item">
                    <div class="result-header">
                        <div class="model-name">
                            <i class="${modelIcons[modelName]}"></i>
                            ${modelDisplayNames[modelName]}
                        </div>
                        <div class="prediction-badge prediction-${result.prediction === 1 ? 'hate' : 'normal'}">
                            ${result.label}
                        </div>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill confidence-${result.prediction === 1 ? 'hate' : 'normal'}"
                             style="width: ${(result.confidence * 100).toFixed(1)}%"></div>
                    </div>
                    <div class="probability-details">
                        <div class="prob-item">
                            <span>Normal:</span>
                            <strong>${(result.probabilities.normal * 100).toFixed(1)}%</strong>
                        </div>
                        <div class="prob-item">
                            <span>Hate Speech:</span>
                            <strong>${(result.probabilities.hate_speech * 100).toFixed(1)}%</strong>
                        </div>
                    </div>
                </div>
            `;
        }

        this.elements.resultsContainer.innerHTML = html;
    }

    displayBatchResults(results) {
        let html = '';
        results.forEach((item, index) => {
            html += `
                <div class="batch-item">
                    <div class="batch-text">${index + 1}. "${item.text}"</div>
                    <div class="batch-prediction">
                        <span class="prediction-badge prediction-${item.result.prediction === 1 ? 'hate' : 'normal'}">
                            ${item.result.label}
                        </span>
                        (${(item.result.confidence * 100).toFixed(1)}% confidence)
                    </div>
                </div>
            `;
        });

        this.elements.batchResults.innerHTML = html;
    }

    clearAll() {
        this.elements.textInput.value = '';
        this.elements.batchInput.value = '';
        this.elements.resultsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-chart-pie"></i>
                <p>Enter text and select a model to see analysis results</p>
            </div>
        `;
        this.elements.batchResults.innerHTML = '';

        // Clear model selection
        this.elements.modelCards.forEach(c => c.classList.remove('selected'));
        this.selectedModel = null;
        this.elements.analyzeBtn.disabled = true;
    }

    showLoading(show) {
        if (show) {
            this.elements.loadingSpinner.classList.add('show');
        } else {
            this.elements.loadingSpinner.classList.remove('show');
        }
    }

    showError(message) {
        alert(message);
    }
}

// Initialize the UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new HateSpeechDetectorUI();
});