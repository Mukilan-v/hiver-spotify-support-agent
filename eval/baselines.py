"""
Baseline Models for Benchmark Comparison.

Baseline 1 (Trivial):
- Intent: Majority Class ('PLAYBACK_AUDIO')
- Reply: Static canned generic macro reply
- Escalation: Always AUTO_HANDLE (or Always ESCALATE)

Baseline 2 (Simple):
- Intent: Simple CountVectorizer + MultinomialNB
- Reply: Top-1 nearest neighbor verbatim reply from historical corpus
- Escalation: Naive single keyword check ("refund" or "hack")
"""

from typing import Dict, Any, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import json
import os

class TrivialBaselineAgent:
    """Baseline 1: Trivial Canned Macro System."""
    def __init__(self, majority_intent: str = "PLAYBACK_AUDIO"):
        self.majority_intent = majority_intent
        self.canned_reply = "Hey there! Thanks for reaching out to Spotify. Please try restarting your device or check https://support.spotify.com. Send us a DM if you need more help /SC"

    def handle_tweet(self, tweet_text: str) -> Dict[str, Any]:
        return {
            "incoming_tweet": tweet_text,
            "intent": self.majority_intent,
            "confidence": 0.20,
            "margin": 0.0,
            "action": "AUTO_HANDLE",
            "stated_reason": "Trivial default policy: auto-handle all queries.",
            "requires_dm": False,
            "reply": self.canned_reply,
            "reply_length": len(self.canned_reply),
            "under_280_chars": len(self.canned_reply) <= 280
        }

class SimpleBaselineAgent:
    """Baseline 2: Simple Naive Bayes + Top-1 Corpus Match + Naive Keyword Escalation."""
    def __init__(self, golden_set_path: str = None):
        self.model = Pipeline([
            ('cv', CountVectorizer(ngram_range=(1, 1))),
            ('nb', MultinomialNB())
        ])
        self.historical_pairs: List[Dict[str, str]] = []
        self._init_data(golden_set_path)

    def _init_data(self, path: str = None):
        if not path:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, "data", "gold", "golden_eval_set.json")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        texts = [d["incoming_tweet"] for d in data]
        labels = [d["gold_intent"] for d in data]
        self.model.fit(texts, labels)
        
        for d in data:
            self.historical_pairs.append({
                "tweet": d["incoming_tweet"],
                "reply": d["historical_reference_reply"]
            })

    def handle_tweet(self, tweet_text: str) -> Dict[str, Any]:
        # 1. Naive Bayes classification
        pred_intent = str(self.model.predict([tweet_text])[0])
        probs = self.model.predict_proba([tweet_text])[0]
        conf = float(max(probs))

        # 2. Naive keyword escalation
        text_lower = tweet_text.lower()
        if "refund" in text_lower or "hack" in text_lower:
            action = "ESCALATE"
            reason = "Simple keyword match: contains 'refund' or 'hack'."
            requires_dm = True
        else:
            action = "AUTO_HANDLE"
            reason = "No high-risk keywords detected."
            requires_dm = False

        # 3. Simple top-1 word overlap search from historical pairs
        query_words = set(text_lower.split())
        best_match = self.historical_pairs[0]
        max_overlap = -1
        for pair in self.historical_pairs:
            pair_words = set(pair["tweet"].lower().split())
            overlap = len(query_words.intersection(pair_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = pair

        reply = best_match["reply"]

        return {
            "incoming_tweet": tweet_text,
            "intent": pred_intent,
            "confidence": round(conf, 4),
            "margin": 0.0,
            "action": action,
            "stated_reason": reason,
            "requires_dm": requires_dm,
            "reply": reply,
            "reply_length": len(reply),
            "under_280_chars": len(reply) <= 280
        }
