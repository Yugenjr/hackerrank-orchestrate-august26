from schemas_intelligence import NotificationFeatureVector, Signal

class NotificationSignalEngine:
    @staticmethod
    def extract(text: str = "", ocr_text: str = "", asr_text: str = "") -> NotificationFeatureVector:
        combined = (text + " " + ocr_text + " " + asr_text).lower()
        
        # Urgency
        urgency_val = "low"
        if any(w in combined for w in ["urgent", "emergency", "asap", "immediately"]):
            urgency_val = "high"
        elif any(w in combined for w in ["soon", "today", "tomorrow"]):
            urgency_val = "medium"
            
        # Importance
        importance_val = "routine"
        if "final notice" in combined or "hospital" in combined or "911" in combined:
            importance_val = "critical"
        elif "reminder" in combined or "update" in combined:
            importance_val = "notable"
            
        return NotificationFeatureVector(
            urgency=Signal[str](value=urgency_val, confidence=0.9, evidence=urgency_val),
            importance=Signal[str](value=importance_val, confidence=0.85, evidence=importance_val),
            requires_action=Signal[bool](value="please" in combined or "action" in combined, confidence=0.9, evidence="action keywords"),
            deadline_detected=Signal[bool](value="due" in combined or "deadline" in combined, confidence=0.8, evidence="due keyword"),
            payment_related=Signal[bool](value="pay" in combined or "$" in combined or "invoice" in combined, confidence=0.95, evidence="money keywords"),
            event_related=Signal[bool](value="invite" in combined or "rsvp" in combined, confidence=0.9, evidence="event keywords"),
            personal_related=Signal[bool](value="mom" in combined or "dad" in combined or "friend" in combined, confidence=0.7, evidence="family keywords"),
            business_related=Signal[bool](value="invoice" in combined or "account" in combined, confidence=0.9, evidence="business keywords"),
            scam_probability=Signal[float](value=0.8 if "winner" in combined and "free" in combined else 0.05, confidence=0.8, evidence="scam heuristics"),
            spam_probability=Signal[float](value=0.9 if "unsubscribe" in combined else 0.1, confidence=0.8, evidence="spam heuristics")
        )
