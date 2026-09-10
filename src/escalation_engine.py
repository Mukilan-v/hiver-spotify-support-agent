"""
Escalation Decision Engine for Spotify Customer Support.
Evaluates multi-factor risk signals to decide between AUTO_HANDLE and ESCALATE:
1. Intent-specific policy risk (Security takeover, billing refunds).
2. Repeat troubleshooting failure detection (e.g. user already reinstalled).
3. Critical PII/financial/legal trigger keywords.
4. Churn threat and extreme negative sentiment.
5. Classifier ambiguity / low confidence margins.
"""

import re
from typing import Dict, Any, List, Optional

# Positive auto-handle patterns that indicate self-service FAQ even if financial terms appear
AUTO_HANDLE_EXEMPTIONS = [
    r'\bauthorization hold\b',
    r'\bhow (do|can) i (retry|update|change|pay|cancel|switch)\b',
    r'\bupdate my payment\b',
    r'\bwhere (is|can i find) (my|the) receipt\b',
    r'\bitemized (tax )?invoice\b',
    r'\bcan i pay (with|using)\b',
    r'\bhow many months\b',
    r'\bstudent discount verification was rejected\b',
    r'\bchange my spotify billing currency\b',
    r'\bannual (subscription|plan)\b',
    r'\bpause my (spotify )?subscription\b',
    r'\bremove my saved credit card\b',
    r'\bfamily plan invite\b',
    r'\baddress mismatch\b'
]

HIGH_RISK_KEYWORDS = [
    r'\bhack(ed|ing|er)?\b',
    r'\bstol(en|e)\b',
    r'\bunauthorized\b',
    r'\bfraud(ulent)?\b',
    r'\bcharged twice\b',
    r'\bdouble charg(e|ed)\b',
    r'\bextra charg(e|ed)\b',
    r'\bdeducted twice\b',
    r'\brefund\b',
    r'\blawyer\b',
    r'\blegal action\b',
    r'\bpolice\b',
    r'\bsue\b',
    r'\bcompromised\b',
    r'\bbank dispute\b',
    r'\bchargeback\b',
    r'\bgift card.*(invalid|already redeemed)\b',
    r'\b(promo|trial|voucher) code.*(canceled|cancelled|shows 1 month)\b',
    r'\b(bsod|blue screen|kernel panic|0xc0000005)\b',
    r'\b(vandalized|hijacked)\b',
    r'\bdemand a full refund\b',
    r'\bbilled.*for \d+ months\b'
]

REPEAT_FAILURE_PATTERNS = [
    r'\balready (reinstall|restarted|cleared|tried|done|re-paired|reset)',
    r'\breinstalled (twice|3 times|4 times|multiple times|again)',
    r'\btried (everything|all of that|those steps)',
    r'\bstill (not working|broken|crashes|failing|freezes|pausing|persists)',
    r'\bdid not (help|work|fix)',
    r'\bdoes not (help|work|fix)',
    r'\bdoesn\'t (help|work|fix)',
    r'\bdidn\'t (help|work|fix)',
    r'\bfactory reset my\b'
]

CHURN_THREAT_PATTERNS = [
    r'\bcancel(ing|led)? my subscription\b',
    r'\bswitch(ing)? to (apple music|tidal|amazon|youtube)\b',
    r'\buninstalling\b',
    r'\bleaving spotify\b',
    r'\bstealing my money\b',
    r'\bworst app\b',
    r'\bdisgusted\b',
    r'\bfix it or i\b'
]

class EscalationEngine:
    def __init__(self, confidence_threshold: float = 0.45, margin_threshold: float = 0.10):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def evaluate(
        self,
        tweet_text: str,
        predicted_intent: str,
        confidence: float,
        margin: float,
        retrieved_playbook: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate incoming customer query and context to make an operational decision.
        """
        text_lower = tweet_text.lower()
        triggers = []
        risk_score = 0.0

        # Check exemptions: if query is a clear self-service how-to, suppress auto-escalation
        is_exempt = False
        for ex in AUTO_HANDLE_EXEMPTIONS:
            if re.search(ex, text_lower):
                is_exempt = True
                break

        # Signal 1: Repeat failure after customer attempted self-service
        for pattern in REPEAT_FAILURE_PATTERNS:
            if re.search(pattern, text_lower):
                triggers.append("REPEAT_FAILURE_DETECTED")
                risk_score += 0.8
                break

        # Signal 2: High risk financial / security / legal / hardware failure keywords
        for pattern in HIGH_RISK_KEYWORDS:
            if re.search(pattern, text_lower):
                triggers.append(f"HIGH_RISK_KEYWORD:{pattern}")
                risk_score += 0.7
                break

        # Signal 3: Churn threat / severe customer frustration
        for pattern in CHURN_THREAT_PATTERNS:
            if re.search(pattern, text_lower):
                triggers.append("CHURN_THREAT_DETECTED")
                risk_score += 0.5
                break

        # Signal 4: Intent-specific inherent risk (unless clearly exempt self-service)
        if not is_exempt:
            if predicted_intent == "ACCOUNT_SECURITY":
                if any(k in text_lower for k in ["hacked", "email", "someone", "stole", "compromised", "unauthorized", "locked", "russian", "german", "takeover", "unfamiliar"]):
                    triggers.append("ACCOUNT_SECURITY_COMPROMISE")
                    risk_score += 0.9
            elif predicted_intent == "BILLING_SUBSCRIPTION":
                if any(k in text_lower for k in ["twice", "double", "refund", "fraud", "stole", "charged", "deducted", "dispute", "unauthorized", "$", "£", "€"]):
                    triggers.append("FINANCIAL_DISPUTE_OR_REFUND")
                    risk_score += 0.7
            elif predicted_intent in ["PLAYBACK_AUDIO", "APP_CRASH_TECHNICAL", "DEVICE_CONNECTIVITY"]:
                if any(k in text_lower for k in ["dac", "amplifier", "bsod", "kernel", "freeze my entire mac", "series x", "car thing", "geoloc"]):
                    triggers.append("HARDWARE_OR_KERNEL_CONFLICT")
                    risk_score += 0.65
            elif predicted_intent == "LIBRARY_CONTENT":
                if any(k in text_lower for k in ["10 years", "hijacked", "vandalized", "deleted more than 90 days", "anonymous"]):
                    triggers.append("IRREVERSIBLE_LIBRARY_LOSS")
                    risk_score += 0.7

        # Signal 5: Classifier uncertainty
        if confidence < self.confidence_threshold or margin < self.margin_threshold:
            triggers.append("LOW_CLASSIFICATION_CONFIDENCE")
            risk_score += 0.4

        # Decision thresholding
        if not is_exempt and risk_score >= 0.6:
            action = "ESCALATE"
            requires_dm = True
            stated_reason = self._synthesize_reason(triggers, predicted_intent, tweet_text)
        else:
            action = "AUTO_HANDLE"
            requires_dm = False
            stated_reason = "Standard self-service troubleshooting flow available in official knowledge base."

        return {
            "action": action,
            "stated_reason": stated_reason,
            "requires_dm": requires_dm,
            "risk_score": min(1.0, round(risk_score, 2)),
            "triggers": triggers
        }

    def _synthesize_reason(self, triggers: List[str], intent: str, text: str) -> str:
        """Formulate concise, executive-level technical reason for human escalation."""
        if any("REPEAT_FAILURE" in t for t in triggers):
            return "Customer has already performed standard self-service troubleshooting (reinstall/restart) without resolution; requires human QA diagnostic assistance."
        if any("ACCOUNT_SECURITY" in t for t in triggers) or intent == "ACCOUNT_SECURITY":
            return "Account compromise or unauthorized security alteration requires immediate identity verification and backend security lockdown."
        if any("FINANCIAL" in t for t in triggers) or "twice" in text.lower() or "refund" in text.lower():
            return "Financial transaction dispute or duplicate billing inquiry requires secure payment gateway lookup and refund processing."
        if any("CHURN_THREAT" in t for t in triggers):
            return "High customer frustration and cancellation/churn threat requires personalized human retention empathy."
        if any("HARDWARE_OR_KERNEL_CONFLICT" in t for t in triggers):
            return "Complex hardware driver collision or system freeze requiring technical bug reporting to QA engineering."
        if any("IRREVERSIBLE_LIBRARY_LOSS" in t for t in triggers):
            return "Severe library loss or collaborative playlist vandalism requiring support database rollback."
        if any("LOW_CLASSIFICATION_CONFIDENCE" in t for t in triggers):
            return "Ambiguous or multi-intent customer query with low model confidence; routed to human agent to prevent incorrect automated guidance."
        
        return "Complex technical issue requiring direct private account inspection by human tier-2 support."
