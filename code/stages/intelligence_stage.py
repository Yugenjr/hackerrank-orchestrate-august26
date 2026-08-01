import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.stage import PipelineStage
from orchestrator.context import PipelineContext
from multimodal_fusion import MultimodalFusionEngine
from schemas_intelligence import MediaContext, Entities, StructuralElements, NotificationFeatureVector, Contradiction

class NotificationIntelligenceStage(PipelineStage):
    @property
    def name(self) -> str:
        return "NotificationIntelligenceStage"
        
    @property
    def dependencies(self):
        return ["FeatureExtractionStage"]
        
    @property
    def is_critical(self):
        return False  # Will fallback to empty MediaContext
        
    async def execute(self, context: PipelineContext) -> MediaContext:
        """
        Runs the Multimodal Fusion (OCR/ASR/Text) intelligence layer.
        """
        row = context.payload
        text = str(row.get("message_text", ""))
        if text.lower() == "nan": text = ""
        
        ocr_text = str(row.get("ocr_text", ""))
        if ocr_text.lower() == "nan": ocr_text = ""
        
        asr_text = str(row.get("asr_transcript", ""))
        if asr_text.lower() == "nan": asr_text = ""
        
        return MultimodalFusionEngine.process(text, ocr_text, asr_text)
        
    async def fallback(self, context: PipelineContext, error: Exception) -> MediaContext:
        """Graceful degradation: Return empty/neutral MediaContext."""
        return MediaContext(
            has_media=False,
            entities=Entities(),
            structures=StructuralElements(),
            features=NotificationFeatureVector(),
            contradiction=Contradiction()
        )
