#!/usr/bin/env python3
"""
Test script for the Hate Speech Detector
"""

from hate_speech_detector import HateSpeechDetector
import time

def test_detector():
    """Test the hate speech detector with sample texts"""
    print("🧪 Testing Hate Speech Detector...")

    # Initialize detector
    detector = HateSpeechDetector()

    # Check available models
    available_models = detector.get_available_models()
    print(f"📊 Available models: {available_models}")

    if not available_models:
        print("❌ No models found! Please train models first.")
        return

    # Test texts
    test_texts = [
        "I love this beautiful day!",
        "You are such an idiot and I hate you",
        "This is a normal conversation between friends",
        "I hope you die you stupid moron"
    ]

    print("\n" + "="*50)
    print("Testing individual predictions:")
    print("="*50)

    # Test each model with sample texts
    for model_name in available_models:
        print(f"\n🧠 Testing {model_name.replace('_', ' ').title()}:")
        print("-" * 30)

        for i, text in enumerate(test_texts, 1):
            result = detector.predict_single(text, model_name)

            if 'error' in result:
                print(f"  {i}. ERROR: {result['error']}")
            else:
                label = result['label']
                confidence = result['confidence'] * 100
                print(f"  {i}. \"{text[:40]}...\" → {label} ({confidence:.1f}%)")

    print("\n" + "="*50)
    print("Testing model comparison:")
    print("="*50)

    # Test comparison feature
    test_text = "You are so stupid, I hate you!"
    print(f"\nComparing models for: \"{test_text}\"")

    comparison_results = detector.compare_models(test_text)

    for model_name, result in comparison_results.items():
        if 'error' in result:
            print(f"  {model_name}: ERROR - {result['error']}")
        else:
            label = result['label']
            confidence = result['confidence'] * 100
            print(f"  {model_name}: {label} ({confidence:.1f}%)")

    print("\n✅ Testing completed!")

if __name__ == "__main__":
    test_detector()