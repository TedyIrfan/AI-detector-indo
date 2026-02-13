"""
Flask Web App untuk Demo Deteksi AI vs Manusia
API endpoint untuk prediksi teks
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load model
model = joblib.load('models/random_forest_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# HTML Template untuk demo
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deteksi AI vs Manusia</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 700px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .result.ai {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
        }
        .result.human {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        .result h2 {
            margin-bottom: 15px;
        }
        .confidence-bar {
            background: rgba(255,255,255,0.3);
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            margin: 15px 0;
        }
        .confidence-fill {
            background: white;
            height: 100%;
            transition: width 0.5s;
        }
        .details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        .detail-item {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
        }
        .detail-label {
            font-size: 12px;
            opacity: 0.8;
        }
        .detail-value {
            font-size: 20px;
            font-weight: bold;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Deteksi AI vs Manusia</h1>
        <p class="subtitle">Paste teks dan klik tombol analisis</p>

        <textarea id="textInput" placeholder="Masukkan teks di sini..."></textarea>

        <button class="btn" onclick="analyze()">🔍 Analisis Teks</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px;">Menganalisis...</p>
        </div>

        <div class="result" id="result">
            <h2 id="resultTitle">Hasil Analisis</h2>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidenceFill"></div>
            </div>
            <p id="confidenceText">Keyakinan: 0%</p>

            <div class="details">
                <div class="detail-item">
                    <div class="detail-label">Probabilitas AI</div>
                    <div class="detail-value" id="probAI">0%</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Probabilitas Manusia</div>
                    <div class="detail-value" id="probHuman">0%</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function analyze() {
            const text = document.getElementById('textInput').value;

            if (text.length < 10) {
                alert('Mohon masukkan minimal 10 karakter');
                return;
            }

            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                const resultDiv = document.getElementById('result');
                const resultTitle = document.getElementById('resultTitle');
                const confidenceFill = document.getElementById('confidenceFill');
                const confidenceText = document.getElementById('confidenceText');
                const probAI = document.getElementById('probAI');
                const probHuman = document.getElementById('probHuman');

                resultDiv.classList.remove('ai', 'human');
                resultDiv.classList.add(data.prediction === 'AI' ? 'ai' : 'human');

                if (data.prediction === 'AI') {
                    resultTitle.textContent = '🤖 Terdeteksi: TULISAN AI';
                } else {
                    resultTitle.textContent = '✍️ Terdeteksi: TULISAN MANUSIA';
                }

                confidenceFill.style.width = (data.confidence * 100) + '%';
                confidenceText.textContent = `Keyakinan: ${(data.confidence * 100).toFixed(1)}%`;

                probAI.textContent = (data.probabilities.AI * 100).toFixed(1) + '%';
                probHuman.textContent = (data.probabilities.Human * 100).toFixed(1) + '%';

                document.getElementById('loading').classList.remove('show');
                resultDiv.classList.add('show');

            } catch (error) {
                alert('Error: ' + error.message);
                document.getElementById('loading').classList.remove('show');
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')

        if len(text.strip()) < 10:
            return jsonify({
                'error': 'Text must be at least 10 characters long'
            }), 400

        # Vectorize and predict
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]

        return jsonify({
            'prediction': 'AI' if prediction == 1 else 'Human',
            'confidence': float(max(probability)),
            'probabilities': {
                'AI': float(probability[1]),
                'Human': float(probability[0])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'Random Forest'})

if __name__ == '__main__':
    print("="*60)
    print("FLASK WEB APP - DETEKSI AI vs MANUSIA")
    print("="*60)
    print("\nStarting server...")
    print("Open browser: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  GET  /         - Web interface")
    print("  POST /predict  - Prediction API")
    print("  GET  /health   - Health check")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
