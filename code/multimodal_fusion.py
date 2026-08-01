from schemas_intelligence import MediaContext
from ner_engine import NEREngine
from structural_extractors import StructuralExtractor
from contradiction_engine import ContradictionEngine
from notification_signals import NotificationSignalEngine

class MultimodalFusionEngine:
    @staticmethod
    def process(text: str = "", ocr_text: str = "", asr_text: str = "") -> MediaContext:
        """
        Fuses all input modalities (text, vision, audio) into the unified MediaContext schema.
        Extracts entities, structures, and notification features.
        """
        text = text or ""
        ocr_text = ocr_text or ""
        asr_text = asr_text or ""
        
        combined = (text + " " + ocr_text + " " + asr_text).strip()
        has_media = bool(ocr_text or asr_text)
        
        entities = NEREngine.extract_entities(combined)
        structures = StructuralExtractor.extract(combined)
        features = NotificationSignalEngine.extract(text, ocr_text, asr_text)
        contradiction = ContradictionEngine.detect(text, ocr_text, asr_text)
        
        return MediaContext(
            has_media=has_media,
            entities=entities,
            structures=structures,
            features=features,
            contradiction=contradiction
        )
