import joblib
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

            # Load KNN
            if os.path.exists('../KNN/knn_hatespeech.pkl'):
                self.models['knn'] = joblib.load('../KNN/knn_hatespeech.pkl')
                self.vectorizers['knn'] = joblib.load('../KNN/knn_tfidf_vectorizer.pkl')
                print("✅ Loaded KNN model")

            # Load SVM
            if os.path.exists('../SVM/svm_hatespeech.pkl'):
                self.models['svm'] = joblib.load('../SVM/svm_hatespeech.pkl')
                self.vectorizers['svm'] = joblib.load('../SVM/svm_tfidf_vectorizer.pkl')
                print("✅ Loaded SVM model")


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
            return self._predict_sklearn(text, model_name)
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _predict_sklearn(self, text, model_name):
        """Predict using sklearn models (Logistic Regression, Naive Bayes, Random Forest, KNN, SVM)"""
        model = self.models[model_name]
        vectorizer = self.vectorizers[model_name]

        # Transform text
        text_vectorized = vectorizer.transform([text])

        # Get prediction
        prediction = model.predict(text_vectorized)[0]

        # Check if model supports predict_proba (SVM LinearSVC doesn't)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probabilities))
            prob_normal = float(probabilities[0])
            prob_hate = float(probabilities[1])
        else:
            # For models without predict_proba, use decision_function or default values
            if hasattr(model, "decision_function"):
                decision_score = model.decision_function(text_vectorized)[0]
                # Convert decision score to pseudo-probability
                confidence = min(0.99, max(0.51, abs(decision_score) / 10.0))
                if prediction == 1:
                    prob_hate = confidence
                    prob_normal = 1.0 - confidence
                else:
                    prob_normal = confidence
                    prob_hate = 1.0 - confidence
            else:
                # Default confidence for models without probability support
                confidence = 0.8
                prob_normal = 0.8 if prediction == 0 else 0.2
                prob_hate = 0.8 if prediction == 1 else 0.2

        return {
            "prediction": int(prediction),
            "label": "Hate Speech" if prediction == 1 else "Normal",
            "confidence": confidence,
            "probabilities": {
                "normal": prob_normal,
                "hate_speech": prob_hate
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