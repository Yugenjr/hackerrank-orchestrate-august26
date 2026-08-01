import re

class QueryExpander:
    """
    Normalizes queries and expands them by adding token representations 
    of common patterns (OTP, URLs) to aid in sparse retrieval (BM25).
    """
    ABBRV_MAP = {
        "ty": "thank you",
        "thx": "thanks",
        "pls": "please",
        "np": "no problem",
        "u": "you",
        "r": "are",
        "ur": "your",
        "brb": "be right back",
        "omg": "oh my god",
        "idk": "i don't know",
        "asap": "as soon as possible",
        "fyi": "for your information",
        "msg": "message"
    }

    OTP_PATTERN = re.compile(r'\b\d{4,8}\b', re.IGNORECASE)
    URL_PATTERN = re.compile(r'https?://\S+', re.IGNORECASE)
    CURRENCY_PATTERN = re.compile(r'(\$|rs\.?|₹|€|£)\s*\d+(?:,\d{3})*(?:\.\d{2})?', re.IGNORECASE)

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text or not isinstance(text, str) or str(text).lower() == "nan":
            return ""
            
        text = text.lower()
        
        words = text.split()
        expanded_words = [cls.ABBRV_MAP.get(w, w) for w in words]
        text = " ".join(expanded_words)
        
        expansions = []
        # Find 4-8 digit numbers that look like OTPs or verifications
        if cls.OTP_PATTERN.search(text) and ("code" in text or "otp" in text or "verification" in text):
            expansions.append("auth_otp_code")
        
        if cls.URL_PATTERN.search(text):
            expansions.append("hyperlink_url")
            
        if cls.CURRENCY_PATTERN.search(text):
            expansions.append("payment_amount_reference")
            
        if expansions:
            text = text + " " + " ".join(expansions)
            
        return text.strip()
