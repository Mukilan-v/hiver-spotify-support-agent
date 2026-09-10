"""
Grounded Reply Generator for @SpotifyCares.
Produces brand-aligned, concise (<280 chars), and factually grounded responses.
Supports:
1. Fast, high-quality RAG template synthesis grounded in retrieved playbooks.
2. Twitter length guardrail (<280 characters).
3. LLM prompt formulation for zero-shot / few-shot generation via external API.
"""

from typing import Dict, Any, List, Optional
import re

TWITTER_CHAR_LIMIT = 280

class ReplyGenerator:
    def __init__(self):
        pass

    def generate_reply(
        self,
        tweet_text: str,
        intent: str,
        action: str,
        retrieved_playbooks: List[Dict[str, Any]],
        requires_dm: bool = False
    ) -> str:
        # Select playbook aligned with the operational action
        matched_pb = None
        for pb in retrieved_playbooks:
            if pb.get("resolution_type") == action:
                matched_pb = pb
                break
        if not matched_pb and retrieved_playbooks:
            matched_pb = retrieved_playbooks[0]

        if action == "ESCALATE":
            reply = self._synthesize_escalate_reply(tweet_text, intent, matched_pb, requires_dm)
        else:
            reply = self._synthesize_autohandle_reply(tweet_text, intent, matched_pb)

        # Enforce Twitter 280-character limit guardrail
        if len(reply) > TWITTER_CHAR_LIMIT:
            reply = self._truncate_gracefully(reply)

        return reply

    def _synthesize_autohandle_reply(
        self,
        tweet: str,
        intent: str,
        pb: Optional[Dict[str, Any]]
    ) -> str:
        if pb and pb.get("canonical_response"):
            base = pb["canonical_response"]
            if len(base) <= TWITTER_CHAR_LIMIT:
                return base

        # Fallback synthesis based on intent
        if intent == "PLAYBACK_AUDIO":
            return "Hey! Try heading to Spotify Settings -> Storage -> Clear Cache, then restart your device. A quick logout and login helps clear playback stalls too! Let us know /SC"
        elif intent == "BILLING_SUBSCRIPTION":
            return "Hey! You can manage subscriptions, update cards, or view receipts anytime from a web browser at https://spotify.com/account /SC"
        elif intent == "APP_CRASH_TECHNICAL":
            return "Hey! We recommend doing a full clean reinstall of the app to clear out corrupted files: https://support.spotify.com/article/reinstall-spotify/ /SC"
        elif intent == "DEVICE_CONNECTIVITY":
            return "Hey! Try forgetting the device in your phone's Bluetooth settings, restarting both devices, and pairing them again. Let us know if that helps! /SC"
        elif intent == "LIBRARY_CONTENT":
            return "Don't panic! You can recover deleted playlists by logging into https://spotify.com/account on a web browser and clicking 'Recover playlists' /SC"
        elif intent == "FEEDBACK_CHITCHAT":
            return "Thanks for sharing your thoughts with us! You can post and upvote feature ideas directly on our Community Idea Exchange at https://community.spotify.com /SC"
        else:
            return "Hey! Check out our official troubleshooting steps at https://support.spotify.com, or let us know your device details so we can assist! /SC"

    def _synthesize_escalate_reply(
        self,
        tweet: str,
        intent: str,
        pb: Optional[Dict[str, Any]],
        requires_dm: bool
    ) -> str:
        t_lower = tweet.lower()
        if "already" in t_lower or "reinstalled" in t_lower or "restarted" in t_lower:
            return "Thanks for letting us know you already tried those steps! Could you drop us a DM with your device model, OS, and Spotify build so our QA team can dig in? /SC"
        
        if intent == "ACCOUNT_SECURITY" or "hack" in t_lower or "stolen" in t_lower:
            return "Hey! We take account security very seriously. Please DM us your registered email address right away so our security team can secure your account immediately /SC"
            
        if intent == "BILLING_SUBSCRIPTION" or "twice" in t_lower or "refund" in t_lower or "charge" in t_lower:
            return "Hey! We'd love to check into this charge backstage. Please drop us a DM with your account email and the charge details so our billing team can assist /SC"

        # General escalation
        return "We hear you and want to look into this backstage. Please send us a quick DM with your account username or email so our support specialists can assist /SC"

    def _truncate_gracefully(self, text: str) -> str:
        """Truncate text cleanly at sentence or word boundary while preserving /SC sign-off."""
        signoff = " /SC"
        budget = TWITTER_CHAR_LIMIT - len(signoff)
        truncated = text[:budget]
        # Find last space
        last_space = truncated.rfind(' ')
        if last_space != -1:
            truncated = truncated[:last_space]
        return truncated + signoff

    def build_llm_prompt(
        self,
        tweet: str,
        intent: str,
        action: str,
        reason: str,
        playbook: Optional[Dict[str, Any]]
    ) -> str:
        """Constructs prompt for LLM grounded response generation."""
        pb_context = ""
        if playbook:
            pb_context = (
                f"Reference Title: {playbook.get('title')}\n"
                f"Resolution Type: {playbook.get('resolution_type')}\n"
                f"Key Actions: {', '.join(playbook.get('key_actions', []))}\n"
                f"Official Links: {', '.join(playbook.get('links', []))}\n"
            )

        prompt = (
            "You are the official customer support voice for @SpotifyCares on Twitter.\n"
            "Your task is to draft a grounded, empathetic, and concise reply to an incoming customer tweet.\n\n"
            f"CUSTOMER TWEET: \"{tweet}\"\n"
            f"CLASSIFIED INTENT: {intent}\n"
            f"ACTION: {action} (Stated Reason: {reason})\n\n"
            f"GROUNDING KNOWLEDGE BASE:\n{pb_context}\n\n"
            "CONSTRAINTS:\n"
            "1. Must be STRICTLY under 280 characters (Twitter limit).\n"
            "2. Adopt Spotify's brand voice: friendly, casual, empathetic, and proactive.\n"
            "3. If ESCALATE, instruct customer to DM their account email or details so support can look 'backstage'.\n"
            "4. If AUTO_HANDLE, provide exact self-service instructions and canonical links.\n"
            "5. End with Spotify agent sign-off tag: ' /SC'.\n\n"
            "Draft the reply directly without quotes or preamble:"
        )
        return prompt
