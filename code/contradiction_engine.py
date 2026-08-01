from schemas_intelligence import Contradiction

class ContradictionEngine:
    @staticmethod
    def detect(text: str = "", ocr_text: str = "", asr_text: str = "") -> Contradiction:
        """
        Cross-validates modalities. If text claims one thing but OCR claims another, flags it.
        """
        text = text or ""
        ocr_text = ocr_text or ""
        asr_text = asr_text or ""
        
        has_contradiction = False
        desc = "No contradiction detected."
        conf = 1.0
        

        # Heuristic 1: Payment vs Free
        if ("pay" in text.lower() or "invoice" in text.lower()) and ("free" in ocr_text.lower() or "gift" in ocr_text.lower()):
            has_contradiction = True
            desc = "Text discusses payment/invoice but image OCR claims 'free' or 'gift'."
            conf = 0.85
            
        # Heuristic 2: Urgent vs Casual
        if "urgent" in text.lower() and ("joke" in asr_text.lower() or "haha" in asr_text.lower()):
            has_contradiction = True
            desc = "Text claims urgency but ASR sentiment is casual/joking."
            conf = 0.90
            
        return Contradiction(
            has_contradiction=has_contradiction,
            description=desc,
            confidence=conf
        )
