"""
Unified Pipeline for Spotify Customer Support Agent.
Integrates:
- IntentClassifier (calibrated classification)
- ResolutionRetriever (contextual RAG)
- EscalationEngine (multi-factor decision making)
- ReplyGenerator (brand-aligned, length-guarded synthesis)
"""

import os
from typing import Dict, Any, Optional
from src.intent_classifier import IntentClassifier
from src.retriever import ResolutionRetriever
from src.escalation_engine import EscalationEngine
from src.reply_generator import ReplyGenerator

class SpotifySupportAgent:
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[ResolutionRetriever] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        reply_generator: Optional[ReplyGenerator] = None
    ):
        self.classifier = classifier or IntentClassifier()
        if not self.classifier.is_trained:
            self.classifier.train_on_golden_set()

        self.retriever = retriever or ResolutionRetriever()
        self.escalation_engine = escalation_engine or EscalationEngine()
        self.reply_generator = reply_generator or ReplyGenerator()

    def handle_tweet(self, tweet_text: str) -> Dict[str, Any]:
        """
        Process incoming customer tweet and return structured classification,
        escalation action, stated reason, and drafted reply.
        """
        # 1. Classify Intent
        classification = self.classifier.predict(tweet_text)
        intent = classification["intent"]
        confidence = classification["confidence"]
        margin = classification["margin"]

        # 2. Retrieve Grounded Playbooks
        playbooks = self.retriever.retrieve(tweet_text, intent=intent, top_k=2)
        top_playbook = playbooks[0] if playbooks else None

        # 3. Evaluate Escalation
        decision = self.escalation_engine.evaluate(
            tweet_text=tweet_text,
            predicted_intent=intent,
            confidence=confidence,
            margin=margin,
            retrieved_playbook=top_playbook
        )
        action = decision["action"]
        stated_reason = decision["stated_reason"]
        requires_dm = decision["requires_dm"]

        # 4. Draft Grounded Reply
        reply = self.reply_generator.generate_reply(
            tweet_text=tweet_text,
            intent=intent,
            action=action,
            retrieved_playbooks=playbooks,
            requires_dm=requires_dm
        )

        return {
            "incoming_tweet": tweet_text,
            "intent": intent,
            "confidence": confidence,
            "margin": margin,
            "action": action,
            "stated_reason": stated_reason,
            "requires_dm": requires_dm,
            "reply": reply,
            "reply_length": len(reply),
            "under_280_chars": len(reply) <= 280,
            "retrieved_playbooks": playbooks
        }
