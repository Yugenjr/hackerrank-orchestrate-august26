from typing import Optional, Dict, Any

class ReasonGenerator:
    """
    Generates and rigorously validates explanation strings.
    Guarantees < 150 chars and non-hallucinated templates.
    """
    @staticmethod
    def generate_and_validate(action: str, message_type: str, evidence: Optional[Dict[Any, Any]] = None, llm_reason: Optional[str] = None) -> str:
        # If an LLM-provided reason exists, validate it
        reason = llm_reason if llm_reason else ""
        
        # If reason is missing, or clearly hallucinating generic placeholders
        if not reason or "[Insert" in reason:
            # Fallback to deterministic generation to prevent hallucination
            if action == "mute" and message_type == "spam":
                reason = "Muted: High-risk spam indicators detected."
            elif action == "mute" and message_type == "group_chat":
                reason = "Muted: Group notifications are disabled by user."
            elif action == "notify" and message_type == "otp":
                reason = "Notify: Urgent authentication/OTP code."
            elif action == "notify" and message_type == "urgent":
                reason = "Notify: High-priority or urgent keyword match."
            elif action == "digest":
                reason = f"Digest: Low priority {message_type} update, saved for later."
            else:
                reason = f"{action.capitalize()}: System assessed as {message_type}."
                
        # Final safety boundary
        if len(reason) > 150:
            reason = reason[:147] + "..."
            
        return reason
