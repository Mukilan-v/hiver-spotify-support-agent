"""
Unit Tests for Spotify Customer Support AI Agent.
Tests:
- IntentClassifier training and probability calibration
- ResolutionRetriever indexing and retrieval ranking
- EscalationEngine multi-factor triggers (PII, repeat failure, churn)
- ReplyGenerator length limit (< 280 chars) and sign-off tag (/SC)
- SpotifySupportAgent unified pipeline end-to-end execution
- Baseline agents execution
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.intent_classifier import IntentClassifier, INTENT_CLASSES
from src.retriever import ResolutionRetriever
from src.escalation_engine import EscalationEngine
from src.reply_generator import ReplyGenerator
from src.pipeline import SpotifySupportAgent
from eval.baselines import TrivialBaselineAgent, SimpleBaselineAgent
from eval.llm_judge import LLMJudge

class TestSpotifySupportAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = SpotifySupportAgent()

    def test_intent_classifier_prediction(self):
        res = self.agent.classifier.predict("Why was I charged twice for my Spotify Family subscription?")
        self.assertIn(res["intent"], INTENT_CLASSES)
        self.assertEqual(res["intent"], "BILLING_SUBSCRIPTION")
        self.assertGreater(res["confidence"], 0.3)
        self.assertIn("BILLING_SUBSCRIPTION", res["probabilities"])

    def test_resolution_retriever(self):
        playbooks = self.agent.retriever.retrieve(
            query="offline songs won't play on airplane mode",
            intent="PLAYBACK_AUDIO",
            top_k=2
        )
        self.assertGreaterEqual(len(playbooks), 1)
        self.assertIn("canonical_response", playbooks[0])

    def test_escalation_security_hacked(self):
        res = self.agent.handle_tweet("Someone hacked my Spotify account and changed the email to mail.ru! Help!")
        self.assertEqual(res["action"], "ESCALATE")
        self.assertTrue(res["requires_dm"])
        self.assertIn("security", res["stated_reason"].lower())
        self.assertIn("DM", res["reply"])

    def test_escalation_repeat_failure(self):
        res = self.agent.handle_tweet("I already reinstalled and restarted my phone twice, but songs still pause every 10 seconds!")
        self.assertEqual(res["action"], "ESCALATE")
        self.assertTrue(res["requires_dm"])
        self.assertIn("already", res["stated_reason"].lower())

    def test_auto_handle_self_service(self):
        res = self.agent.handle_tweet("How do I cancel my Spotify Premium subscription before next Tuesday?")
        self.assertEqual(res["action"], "AUTO_HANDLE")
        self.assertFalse(res["requires_dm"])
        self.assertIn("spotify.com/account", res["reply"])

    def test_reply_length_guardrail(self):
        test_queries = [
            "Why did you charge me twice on my credit card?",
            "My app is crashing every single second on Windows 11",
            "Can I recover a playlist I accidentally deleted yesterday?",
            "Love the new Daylist feature, great job team!"
        ]
        for query in test_queries:
            res = self.agent.handle_tweet(query)
            self.assertLessEqual(len(res["reply"]), 280, f"Reply exceeded 280 chars: {res['reply']}")
            self.assertTrue(res["reply"].strip().endswith("/SC"), f"Reply missing /SC sign-off: {res['reply']}")

    def test_baselines_execution(self):
        t_agent = TrivialBaselineAgent()
        s_agent = SimpleBaselineAgent()
        t_res = t_agent.handle_tweet("Where are my songs?")
        s_res = s_agent.handle_tweet("Where are my songs?")
        self.assertIn("action", t_res)
        self.assertIn("action", s_res)

    def test_llm_judge_execution(self):
        judge = LLMJudge()
        scores = judge.judge_reply(
            incoming_tweet="Where is my student discount?",
            gold_intent="BILLING_SUBSCRIPTION",
            gold_escalation="AUTO_HANDLE",
            pred_intent="BILLING_SUBSCRIPTION",
            pred_escalation="AUTO_HANDLE",
            draft_reply="Hey! You can re-verify at https://spotify.com/student /SC",
            historical_reference="Check spotify.com/student /SC"
        )
        self.assertEqual(scores["relevance"], 5)
        self.assertEqual(scores["groundedness"], 5)
        self.assertEqual(scores["brand_voice"], 5)
        self.assertEqual(scores["escalation_correctness"], 5)

if __name__ == "__main__":
    unittest.main()
