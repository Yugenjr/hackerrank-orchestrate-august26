import logging
import json
from typing import Dict, Any, Tuple, Literal, cast, List

from schemas import FinalDecisionOutput, DecisionRouterOutput, SafetyAgentOutput
from policies import PolicyEngine
from reason_generator import ReasonGenerator

# Structured JSON logging for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DecisionEngine")

class DecisionEngine:
    def __init__(self, policy_engine: PolicyEngine):
        self.policy_engine = policy_engine
        
    def get_safety_assessment(self, payload: str, context: Dict) -> SafetyAgentOutput:
        # Mock LLM API call for hackathon tests
        return SafetyAgentOutput(is_safe=True, risk_category="none", risk_reason="")
        
    def get_llm_decision(self, payload: str, context: Dict) -> DecisionRouterOutput:
        # Mock LLM API call for hackathon tests
        return DecisionRouterOutput(action="digest", message_type="personal", llm_confidence=0.8)

    def calculate_confidence(self, llm_conf: float, retrieval_conf: float, trust_score: float) -> float:
        """Deterministically calibrates final confidence."""
        final = (0.5 * llm_conf) + (0.3 * retrieval_conf) + (0.2 * trust_score)
        return min(1.0, max(0.0, final))
        
    def process(self, msg_id: str, payload: str, context: Dict, retrieval_meta: Dict) -> Tuple[FinalDecisionOutput, Dict]:
        """
        Executes the 6-stage reasoning pipeline.
        Returns the Pydantic FinalDecisionOutput and an internal trace dictionary.
        """
        trace: Dict[str, Any] = {"message_id": msg_id, "steps": []}
        
        # 1. Deterministic Pre-flight
        policy_result = self.policy_engine.evaluate(payload, context)
        if policy_result:
            p_action, msg_type, conf, reason = policy_result
            action = cast(Literal["notify", "digest", "mute"], p_action)
            trace["steps"].append({"step": "deterministic_policy", "matched": True, "action": action, "rule_reason": reason})
            validated_reason = ReasonGenerator.generate_and_validate(action, msg_type, None, reason)
            
            logger.info(json.dumps({"event": "decision", "type": "deterministic", "action": action, "msg_id": msg_id}))
            return FinalDecisionOutput(
                message_id=msg_id, action=action, message_type=msg_type, 
                reason=validated_reason, confidence=conf, evidence_message_ids=retrieval_meta.get("evidence", "none")
            ), trace
            
        trace["steps"].append({"step": "deterministic_policy", "matched": False})
        
        # 2. Safety Check (Overrides ALL downstream systems)
        safety = self.get_safety_assessment(payload, context)
        trace["steps"].append({"step": "safety_check", "is_safe": safety.is_safe})
        if not safety.is_safe:
            action = cast(Literal["notify", "digest", "mute"], "mute")
            msg_type, conf = safety.risk_category, 0.95
            reason = ReasonGenerator.generate_and_validate(action, msg_type, None, safety.risk_reason)
            logger.info(json.dumps({"event": "decision", "type": "safety_override", "action": action, "msg_id": msg_id}))
            return FinalDecisionOutput(
                message_id=msg_id, action=action, message_type=msg_type, 
                reason=reason, confidence=conf, evidence_message_ids=retrieval_meta.get("evidence", "none")
            ), trace
            
        # 3. LLM Router
        llm_decision = self.get_llm_decision(payload, context)
        trace["steps"].append({"step": "llm_router", "raw_decision": llm_decision.model_dump()})
        
        # 4. Conflict Resolution & Confidence
        trust_score = context.get("sender_trust_score", 0.5)
        retrieval_conf = retrieval_meta.get("confidence", 0.0)
        final_conf = self.calculate_confidence(llm_decision.llm_confidence, retrieval_conf, trust_score)
        
        action_str = llm_decision.action
        
        # Conflict Resolution Rule: Safety > LLM > History > Cautious Default
        if final_conf < 0.40 and action_str != "digest":
            trace["steps"].append({"step": "conflict_resolution", "override": "low_confidence_digest"})
            action_str = "digest"
            
        action = cast(Literal["notify", "digest", "mute"], action_str)
            
        # 5. Reason Generation
        reason = ReasonGenerator.generate_and_validate(action, llm_decision.message_type)
        trace["steps"].append({"step": "final_output", "action": action, "confidence": final_conf})
        
        logger.info(json.dumps({"event": "decision", "type": "llm_routed", "action": action, "msg_id": msg_id, "conf": final_conf}))
        return FinalDecisionOutput(
            message_id=msg_id, action=action, message_type=llm_decision.message_type, 
            reason=reason, confidence=final_conf, evidence_message_ids=retrieval_meta.get("evidence", "none")
        ), trace
        
    def counterfactual_analysis(self, msg_id: str, payload: str, context: Dict, retrieval_meta: Dict) -> Dict:
        """
        Utility for debugging: Re-runs the pipeline altering single context variables 
        to see if the decision flips (e.g. what if trust was 0? what if group was muted?).
        """
        baseline, _ = self.process(msg_id, payload, context, retrieval_meta)
        
        ctx_no_trust = dict(context)
        ctx_no_trust["sender_trust_score"] = 0.0
        r1, _ = self.process(msg_id, payload, ctx_no_trust, retrieval_meta)
        
        ctx_muted = dict(context)
        ctx_muted["group_muted"] = True
        r2, _ = self.process(msg_id, payload, ctx_muted, retrieval_meta)
        
        return {
            "baseline_action": baseline.action,
            "if_zero_trust": r1.action,
            "if_group_muted": r2.action
        }
