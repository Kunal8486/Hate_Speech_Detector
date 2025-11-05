import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class HateSpeechDetector:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.load_all_models()

    def load_all_models(self):
        """Load all available trained models"""
        try:
            # Load Logistic Regression
            if os.path.exists('../Logistic Regression/hate_speech_logreg.pkl'):
                self.models['logistic_regression'] = joblib.load('../Logistic Regression/hate_speech_logreg.pkl')
                self.vectorizers['logistic_regression'] = joblib.load('../Logistic Regression/tfidf_vectorizer.pkl')
                print("✅ Loaded Logistic Regression model")

            # Load Naive Bayes
            if os.path.exists('../Naive Bayes/hate_speech_nb.pkl'):
                self.models['naive_bayes'] = joblib.load('../Naive Bayes/hate_speech_nb.pkl')
                self.vectorizers['naive_bayes'] = joblib.load('../Naive Bayes/tfidf_vectorizer.pkl')
                print("✅ Loaded Naive Bayes model")

            # Load Random Forest
            if os.path.exists('../Random Forest/hate_speech_rf.pkl'):
                self.models['random_forest'] = joblib.load('../Random Forest/hate_speech_rf.pkl')
                self.vectorizers['random_forest'] = joblib.load('../Random Forest/tfidf_vectorizer.pkl')
                print("✅ Loaded Random Forest model")

            # Load DistilBERT
            if os.path.exists('../Distilbert/hate_speech_distilbert'):
                self.models['distilbert'] = AutoModelForSequenceClassification.from_pretrained('../Distilbert/hate_speech_distilbert')
                self.vectorizers['distilbert'] = AutoTokenizer.from_pretrained('../Distilbert/hate_speech_distilbert')
                self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                self.models['distilbert'].to(self.device)
                print("✅ Loaded DistilBERT model")

        except Exception as e:
            print(f"❌ Error loading models: {e}")

    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())

    def predict_single(self, text, model_name):
        """Predict hate speech for a single text using specified model"""
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not available"}

        try:
            if model_name == 'distilbert':
                return self._predict_distilbert(text)
            else:
                return self._predict_sklearn(text, model_name)
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _predict_sklearn(self, text, model_name):
        """Predict using sklearn models (Logistic Regression, Naive Bayes, Random Forest)"""
        model = self.models[model_name]
        vectorizer = self.vectorizers[model_name]

        # Transform text
        text_vectorized = vectorizer.transform([text])

        # Get prediction and probability
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]

        return {
            "prediction": int(prediction),
            "label": "Hate Speech" if prediction == 1 else "Normal",
            "confidence": float(max(probabilities)),
            "probabilities": {
                "normal": float(probabilities[0]),
                "hate_speech": float(probabilities[1])
            }
        }

    def _predict_distilbert(self, text):
        """Predict using DistilBERT model"""
        model = self.models['distilbert']
        tokenizer = self.vectorizers['distilbert']

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()

        probs = probabilities.cpu().numpy()[0]

        return {
            "prediction": int(prediction),
            "label": "Hate Speech" if prediction == 1 else "Normal",
            "confidence": float(max(probs)),
            "probabilities": {
                "normal": float(probs[0]),
                "hate_speech": float(probs[1])
            }
        }

    def predict_batch(self, texts, model_name):
        """Predict hate speech for multiple texts"""
        results = []
        for text in texts:
            result = self.predict_single(text, model_name)
            results.append(result)
        return results

    def compare_models(self, text):
        """Compare predictions across all available models"""
        results = {}
        for model_name in self.get_available_models():
            results[model_name] = self.predict_single(text, model_name)
        return results