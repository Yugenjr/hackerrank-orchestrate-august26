import os
import sys
import json
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.stage import PipelineStage
from orchestrator.context import PipelineContext
from decision_engine import DecisionEngine
from schemas import FinalDecisionOutput

class DecisionStage(PipelineStage):
    def __init__(self, decision_engine: DecisionEngine):
        self.engine = decision_engine
        
    @property
    def name(self) -> str:
        return "DecisionStage"
        
    @property
    def dependencies(self):
        return ["RetrievalStage", "NotificationIntelligenceStage", "SafetyStage"]
        
    @property
    def is_critical(self):
        return True
        
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """
        Executes the final Decision Engine fusing retrieval, intelligence, and safety.
        """
        features = context.get_result("FeatureExtractionStage") or {}
        retrieval_res = context.get_result("RetrievalStage") or {}
        safety_res = context.get_result("SafetyStage") or {}
        intelligence_res = context.get_result("NotificationIntelligenceStage")
        
        # Format retrieval metadata for Decision Engine
        ret_meta = {
            "evidence": ";".join(retrieval_res.get("evidence_message_ids", ["none"])),
            "confidence": retrieval_res.get("retrieval_confidence", 0.0)
        }
        
        # Format context (merging features and intelligence signals)
        if intelligence_res:
            features["urgency"] = intelligence_res.features.urgency.value
            features["importance"] = intelligence_res.features.importance.value
            features["scam_probability"] = intelligence_res.features.scam_probability.value
            features["spam_probability"] = intelligence_res.features.spam_probability.value
            features["requires_action"] = intelligence_res.features.requires_action.value
            features["payment_related"] = intelligence_res.features.payment_related.value
            features["event_related"] = intelligence_res.features.event_related.value
            features["personal_related"] = intelligence_res.features.personal_related.value
            features["business_related"] = intelligence_res.features.business_related.value
            
        features["safety"] = safety_res
        
        mode = context.metadata.get("evaluation_mode", "C")
        payload_str = json.dumps(context.payload)
        
        decision, trace = self.engine.process(
            msg_id=context.message_id,
            payload=payload_str,
            context=features,
            retrieval_meta=ret_meta,
            mode=mode
        )
        
        return {
            "decision": decision.model_dump(),
            "decision_trace": trace
        }
