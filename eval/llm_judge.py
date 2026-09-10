"""
LLM-as-a-Judge Evaluation Module for Customer Support Replies.
Implements a 4-dimensional evaluation rubric:
1. Relevance (1-5)
2. Groundedness / Faithfulness (1-5)
3. Brand Voice & Tone (1-5)
4. Escalation Correctness (1-5)

Provides:
- Fast, calibrated heuristic judge (deterministic, offline, reproducible without API keys).
- API-ready prompt template for external LLM judges (Gemini, Claude, GPT-4).
"""

from typing import Dict, Any, List, Optional
import re

RUBRIC = {
    "relevance": {
        5: "Directly and comprehensively addresses the customer's specific technical/billing query.",
        4: "Addresses the primary issue with minor missing detail.",
        3: "Partially relevant; generic troubleshooting that loosely touches the topic.",
        2: "Largely tangential or misses the core question.",
        1: "Completely irrelevant or unresponsive to the query."
    },
    "groundedness": {
        5: "Factually accurate and strictly grounded in official Spotify playbooks and verified workflows.",
        4: "Accurate guidance with minor wording variations from official articles.",
        3: "Plausible advice but includes unsupported claims or unverified steps.",
        2: "Hallucinates nonexistent Spotify settings, policies, or faulty links.",
        1: "Blatantly contradictory, dangerous, or fabricated guidance."
    },
    "brand_voice": {
        5: "Exemplary Spotify tone: friendly, warm, empathetic, concise (<280 chars), and includes /SC sign-off.",
        4: "Polite and helpful, under 280 chars, includes proper sign-off.",
        3: "Slightly robotic or generic corporate tone; within length limit.",
        2: "Cold, dismissive, or exceeds 280 characters.",
        1: "Rude, aggressive, heavily truncated, or unintelligible."
    },
    "escalation_correctness": {
        5: "Perfect operational decision: accurately auto-handles self-service or escalates high-risk cases with clear reason and DM callout.",
        4: "Correct operational decision with slightly suboptimal stated reason.",
        3: "Acceptable decision but borderline risk handling (e.g. escalating a simple FAQ).",
        2: "Incorrect decision (e.g. auto-handling an active security hack or refund).",
        1: "Catastrophic error: attempting to handle fraudulent billing publicly on Twitter or ignoring compromised account."
    }
}

class LLMJudge:
    def __init__(self, use_llm_api: bool = False, api_key: Optional[str] = None):
        self.use_llm_api = use_llm_api
        self.api_key = api_key

    def judge_reply(
        self,
        incoming_tweet: str,
        gold_intent: str,
        gold_escalation: str,
        pred_intent: str,
        pred_escalation: str,
        draft_reply: str,
        historical_reference: str,
        requires_dm_gold: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single agent response against the ground truth and rubric.
        Returns score dictionary with explanation.
        """
        # Calibrated rubric scoring logic
        scores = {}
        critique = []

        # 1. Relevance Score (1-5)
        # Check intent alignment and key overlap
        if pred_intent == gold_intent:
            rel_score = 5
        elif self._are_intents_related(pred_intent, gold_intent):
            rel_score = 4
            critique.append("Intent partially aligned with related category.")
        else:
            rel_score = 2
            critique.append(f"Intent mismatch: predicted {pred_intent} vs gold {gold_intent}.")
        scores["relevance"] = rel_score

        # 2. Groundedness Score (1-5)
        # Check if reply uses valid spotify URLs and playbooks
        has_spotify_link = "spotify.com" in draft_reply
        has_invalid_link = "t.co" in draft_reply and "http" not in draft_reply
        if "hallucinate" in draft_reply.lower():
            gro_score = 1
        elif has_spotify_link or "/SC" in draft_reply:
            gro_score = 5 if pred_intent == gold_intent else 4
        else:
            gro_score = 3
        scores["groundedness"] = gro_score

        # 3. Brand Voice Score (1-5)
        bv_score = 5
        if len(draft_reply) > 280:
            bv_score -= 2
            critique.append("Exceeds 280-character Twitter limit.")
        if "/SC" not in draft_reply and not draft_reply.endswith("/"):
            bv_score -= 1
            critique.append("Missing signature /SC agent tag.")
        if not any(draft_reply.startswith(g) for g in ["Hey", "Thanks", "Do not", "Don't", "We hear", "We take", "That"]):
            bv_score -= 1
        scores["brand_voice"] = max(1, bv_score)

        # 4. Escalation Correctness Score (1-5)
        if pred_escalation == gold_escalation:
            esc_score = 5
            if gold_escalation == "ESCALATE":
                # Check if customer was instructed to DM
                if "DM" in draft_reply or "direct message" in draft_reply.lower():
                    esc_score = 5
                else:
                    esc_score = 4
                    critique.append("Escalated correctly but failed to prompt for DM.")
        else:
            if gold_escalation == "ESCALATE" and pred_escalation == "AUTO_HANDLE":
                esc_score = 1  # High-risk failure: missed human escalation
                critique.append("CRITICAL: Missed escalation for sensitive/repeat issue.")
            else:
                esc_score = 2  # False positive: unnecessary human escalation
                critique.append("Suboptimal: Escalated a resolvable self-service issue.")
        scores["escalation_correctness"] = esc_score

        scores["mean_score"] = round(sum(scores.values()) / len(scores), 2)
        scores["critique"] = "; ".join(critique) if critique else "Response meets all rubric standards."

        return scores

    def _are_intents_related(self, p: str, g: str) -> bool:
        related_pairs = [
            ("PLAYBACK_AUDIO", "APP_CRASH_TECHNICAL"),
            ("DEVICE_CONNECTIVITY", "PLAYBACK_AUDIO"),
            ("ACCOUNT_SECURITY", "BILLING_SUBSCRIPTION"),
            ("LIBRARY_CONTENT", "PLAYBACK_AUDIO")
        ]
        return (p, g) in related_pairs or (g, p) in related_pairs

    def build_judge_prompt(
        self,
        tweet: str,
        pred_intent: str,
        pred_escalation: str,
        stated_reason: str,
        draft_reply: str,
        gold_ref: str
    ) -> str:
        """Prompt to send to an LLM evaluator."""
        return f"""You are an impartial expert Customer Experience Auditor for @SpotifyCares.
Evaluate the following support agent's reply to an incoming customer tweet according to our rubric.

CUSTOMER TWEET: "{tweet}"
HISTORICAL HUMAN BENCHMARK REPLY: "{gold_ref}"

AGENT EVALUATION CANDIDATE:
- Classified Intent: {pred_intent}
- Escalation Decision: {pred_escalation}
- Stated Escalation Reason: {stated_reason}
- Draft Reply: "{draft_reply}"

RUBRIC AXES (Score 1 to 5 each):
1. RELEVANCE: Does the reply directly address the query?
2. GROUNDEDNESS: Is the advice factually sound and aligned with Spotify official procedures?
3. BRAND VOICE: Is it warm, friendly, <280 characters, and signed off with /SC?
4. ESCALATION CORRECTNESS: Was the decision to auto-handle or escalate to human correct?

Respond in JSON format:
{{
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "brand_voice": <1-5>,
  "escalation_correctness": <1-5>,
  "rationale": "<BRIEF_REASONING>"
}}"""
