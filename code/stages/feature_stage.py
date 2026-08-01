import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.stage import PipelineStage
from orchestrator.context import PipelineContext
from feature_engineering import ContextBuilder

class FeatureExtractionStage(PipelineStage):
    def __init__(self, context_builder: ContextBuilder):
        self.context_builder = context_builder
        
    @property
    def name(self) -> str:
        return "FeatureExtractionStage"
        
    @property
    def dependencies(self):
        return []
        
    @property
    def is_critical(self):
        return True
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """
        Extracts structural JSON features from the raw payload and data infrastructure.
        """
        features = self.context_builder.build_context(context.payload)
        return features
