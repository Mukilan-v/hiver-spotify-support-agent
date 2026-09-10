"""
Intent Classification Module for Spotify Customer Support.
Provides:
1. Fast, highly calibrated multi-class classifier using TF-IDF (word & char n-grams) + Logistic Regression.
2. Confidence score estimation and margin calculation.
3. Optional LLM zero-shot / few-shot prompt formulation for API-backed classification.
"""

import json
import os
import re
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

INTENT_CLASSES = [
    "BILLING_SUBSCRIPTION",
    "PLAYBACK_AUDIO",
    "APP_CRASH_TECHNICAL",
    "ACCOUNT_SECURITY",
    "DEVICE_CONNECTIVITY",
    "LIBRARY_CONTENT",
    "FEEDBACK_CHITCHAT"
]

INTENT_DESCRIPTIONS = {
    "BILLING_SUBSCRIPTION": "Disputes over charges, double billing, student discount verification, family plan address mismatch, cancellation, or pricing.",
    "PLAYBACK_AUDIO": "Music buffering, songs pausing/skipping, greyed-out unavailable songs, offline playback sync failures, or sound quality.",
    "APP_CRASH_TECHNICAL": "App crashing on launch, infinite loading screens, Windows/Mac freezes, black screen after update, or memory leaks.",
    "ACCOUNT_SECURITY": "Hacked accounts, unauthorized email changes, foreign logins, password reset email failures, or credential takeover.",
    "DEVICE_CONNECTIVITY": "Bluetooth disconnects, CarPlay/Android Auto issues, smart speaker (Alexa/Sonos) linking, or console integration.",
    "LIBRARY_CONTENT": "Deleted playlist recovery, missing albums/tracks, local MP3 Wi-Fi sync, or collaborative playlist vandalism.",
    "FEEDBACK_CHITCHAT": "Feature requests, UI design critiques, brand praise/compliments, or casual social interactions."
}

def clean_tweet_text(text: str) -> str:
    """Normalize tweet text by removing mentions, normalizing whitespace and urls."""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class IntentClassifier:
    def __init__(self, model_path: str = None):
        self.classes = INTENT_CLASSES
        self.model: Pipeline = None
        self.is_trained = False
        self._init_model()

    def _init_model(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                analyzer='word',
                sublinear_tf=True,
                min_df=1,
                strip_accents='unicode',
                token_pattern=r'(?u)\b\w+\b'
            )),
            ('clf', LogisticRegression(
                C=2.5,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])

    def train(self, texts: List[str], labels: List[str]):
        """Train classifier on labeled text samples."""
        cleaned_texts = [clean_tweet_text(t) for t in texts]
        self.model.fit(cleaned_texts, labels)
        self.is_trained = True

    def train_on_golden_set(self, golden_set_path: str = None):
        """Convenience loader to train or bootstrap on golden set."""
        if not golden_set_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            golden_set_path = os.path.join(base_dir, "data", "gold", "golden_eval_set.json")
        
        with open(golden_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        texts = [d["incoming_tweet"] for d in data]
        labels = [d["gold_intent"] for d in data]
        self.train(texts, labels)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent with calibrated probability distribution.
        Returns:
            {
                "intent": str,
                "confidence": float,
                "margin": float,
                "probabilities": dict
            }
        """
        if not self.is_trained:
            self.train_on_golden_set()

        cleaned = clean_tweet_text(text)
        probs = self.model.predict_proba([cleaned])[0]
        classes = self.model.classes_
        
        # Sort descending
        ranked_indices = np.argsort(probs)[::-1]
        top_idx = ranked_indices[0]
        second_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx
        
        top_intent = classes[top_idx]
        confidence = float(probs[top_idx])
        margin = float(probs[top_idx] - probs[second_idx])
        
        prob_dict = {classes[i]: round(float(probs[i]), 4) for i in ranked_indices}
        
        return {
            "intent": top_intent,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "probabilities": prob_dict
        }

    def build_llm_prompt(self, tweet: str) -> str:
        """Constructs zero-shot / few-shot prompt for LLM classification."""
        prompt = (
            "You are an expert customer support intent classifier for @SpotifyCares.\n"
            "Classify the following incoming customer tweet into EXACTLY ONE of the following 7 intents:\n\n"
        )
        for intent, desc in INTENT_DESCRIPTIONS.items():
            prompt += f"- {intent}: {desc}\n"
        
        prompt += f"\nCustomer Tweet: \"{tweet}\"\n\n"
        prompt += (
            "Respond in strictly valid JSON format with keys:\n"
            "{\n"
            '  "intent": "<ONE_OF_THE_7_INTENTS>",\n'
            '  "confidence": <FLOAT_BETWEEN_0_AND_1>,\n'
            '  "reasoning": "<BRIEF_EXPLANATION>"\n'
            "}"
        )
        return prompt
