from flask import Flask, request, jsonify, send_from_directory
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("⚠️  flask-cors not installed. Install with: pip install flask-cors")
import os
import sys

# Add current directory to path to import our detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hate_speech_detector import HateSpeechDetector

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)
else:
    # Add basic CORS headers manually
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# Initialize the detector
detector = HateSpeechDetector()

@app.route('/')
def index():
    """Serve the main UI page"""
    return send_from_directory('UI', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from UI directory"""
    return send_from_directory('UI', filename)

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    try:
        models = detector.get_available_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_single():
    """Predict hate speech for a single text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_name = data.get('model', '')

        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400

        if not model_name:
            return jsonify({
                'success': False,
                'error': 'Model name is required'
            }), 400

        result = detector.predict_single(text, model_name)

        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Predict hate speech for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        model_name = data.get('model', '')

        if not texts:
            return jsonify({
                'success': False,
                'error': 'Texts array is required'
            }), 400

        if not model_name:
            return jsonify({
                'success': False,
                'error': 'Model name is required'
            }), 400

        results = detector.predict_batch(texts, model_name)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """Compare predictions across all available models"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({
                'success': False,
                'error': 'Text is required'
            }), 400

        results = detector.compare_models(text)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'Hate Speech Detector API is running',
        'available_models': detector.get_available_models()
    })

if __name__ == '__main__':
    print("🚀 Starting Hate Speech Detector API...")
    print(f"📊 Available models: {detector.get_available_models()}")
    print("🌐 Open http://localhost:8500 to access the UI")
    app.run(debug=True, host='0.0.0.0', port=8500)