import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.stage import PipelineStage
from orchestrator.context import PipelineContext
from retrieval import RetrievalEngine

class RetrievalStage(PipelineStage):
    def __init__(self, retrieval_engine: RetrievalEngine):
        self.engine = retrieval_engine
        
    @property
    def name(self) -> str:
        return "RetrievalStage"
        
    @property
    def dependencies(self):
        return ["FeatureExtractionStage"]
        
    @property
    def is_critical(self):
        return False  # Will fallback to no evidence
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """
        Runs the hybrid embedding and retrieval pipeline.
        Depends on the features extracted in FeatureExtractionStage.
        """
        features = context.get_result("FeatureExtractionStage")
        if features is None:
            features = {}
            
        result = self.engine.retrieve(context.payload, features)
        return result
        
    async def fallback(self, context: PipelineContext, error: Exception) -> Dict[str, Any]:
        """Graceful degradation: if retrieval crashes, return empty evidence."""
        return {
            "evidence_message_ids": ["none"],
            "retrieval_confidence": 0.0,
            "retrieval_reason": f"Retrieval failed and degraded: {str(error)}",
            "retrieval_metadata": {}
        }
