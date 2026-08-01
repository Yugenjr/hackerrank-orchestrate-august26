import logging
import json
import os
import pandas as pd
from typing import Dict, Any, Tuple, Literal, cast, List

from schemas import FinalDecisionOutput, DecisionRouterOutput, SafetyAgentOutput
from policies import PolicyEngine
from reason_generator import ReasonGenerator

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- LLM Provider Abstractions ---
class LLMProvider:
    def get_decision(self, combined_text: str, context: Dict) -> DecisionRouterOutput:
        raise NotImplementedError

class MockMLProvider(LLMProvider):
    def __init__(self):
        self._ml_action_model = None
        self._ml_type_model = None
        self.feature_names = []
        self.importances = []
        self._init_ml_model()

    def _init_ml_model(self):
        if not SKLEARN_AVAILABLE: return
        csv_path = "dataset/sample_messages.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df = df.dropna(subset=['action', 'message_type'])
                if len(df) == 0: return
                
                texts = df['message_text'].fillna("").astype(str)
                
                self._ml_action_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1,2))),
                    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                self._ml_action_model.fit(texts, df['action'])
                
                self._ml_type_model = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1,2))),
                    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
                self._ml_type_model.fit(texts, df['message_type'])
                
                # Extract features for explainability
                self.feature_names = self._ml_action_model.named_steps['tfidf'].get_feature_names_out()
                self.importances = self._ml_action_model.named_steps['clf'].feature_importances_
            except Exception as e:
                logging.error(f"Failed to train ML LLM fallback: {e}")
                self._ml_action_model = None
                self._ml_type_model = None

    def get_decision(self, combined_text: str, context: Dict) -> DecisionRouterOutput:
        if self._ml_action_model is not None and self._ml_type_model is not None:
            pred_action = self._ml_action_model.predict([combined_text])[0]
            pred_type = self._ml_type_model.predict([combined_text])[0]
            probs = self._ml_action_model.predict_proba([combined_text])[0]
            conf = max(probs)
            return DecisionRouterOutput(
                action=cast(Literal["notify", "digest", "mute"], pred_action),
                message_type=pred_type,
                llm_confidence=float(conf)
            )
        return DecisionRouterOutput(action="digest", message_type="personal", llm_confidence=0.6)

class ProviderFactory:
    @staticmethod
    def get_provider() -> LLMProvider:
        # Configuration-driven selection. Fallback to ML.
        # Future: If os.environ.get("OPENAI_API_KEY") return OpenAIProvider()
        return MockMLProvider()

# Structured JSON logging for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("DecisionEngine")

class DecisionEngine:
    def __init__(self, policy_engine: PolicyEngine):
        self.policy_engine = policy_engine
        self.llm_provider = ProviderFactory.get_provider()
    def get_safety_assessment(self, payload: str, context: Dict) -> SafetyAgentOutput:
        # Mock LLM API call for hackathon tests
        return SafetyAgentOutput(is_safe=True, risk_category="none", risk_reason="")
        
    def get_llm_decision(self, payload: str, context: Dict) -> DecisionRouterOutput:
        try:
            payload_dict = json.loads(payload)
            text = str(payload_dict.get("message_text", "")).lower()
            ocr = str(payload_dict.get("ocr_text", "")).lower()
            asr = str(payload_dict.get("asr_transcript", "")).lower()
            combined = " ".join([t for t in [text, ocr, asr] if t and t != "nan"])
        except:
            combined = ""
            
        return self.llm_provider.get_decision(combined, context)

    def calculate_confidence(self, llm_conf: float, retrieval_conf: float, soft_rules: List, context: Dict) -> float:
        """Deterministically calibrates final confidence by checking subsystem agreement."""
        base_conf = llm_conf * 0.6 + retrieval_conf * 0.4
        
        # Boost if intelligence signals indicate clear intent
        if context.get("urgency", "low") == "high" or context.get("importance", "routine") == "critical":
            base_conf += 0.1
            
        # Boost if soft rules align
        if len(soft_rules) > 0:
            base_conf += 0.15
            
        # Penalty if subsystem signals contradict (e.g. LLM isn't confident and retrieval is poor)
        if llm_conf < 0.5 and retrieval_conf < 0.3:
            return min(base_conf, 0.4) # Hard cap
            
        return min(1.0, max(0.0, base_conf))
        
    def process(self, msg_id: str, payload: str, context: Dict, retrieval_meta: Dict, mode: str = "C") -> Tuple[FinalDecisionOutput, Dict]:
        """
        Executes the reasoning pipeline.
        mode "A": Deterministic only
        mode "B": LLM only
        mode "C": Hybrid (Production)
        """
        trace: Dict[str, Any] = {"message_id": msg_id, "steps": []}
        
        # Explainability Trace
        internal_explanation = {
            "top_features_triggered": [],
            "evidence_messages": retrieval_meta.get("evidence", "none"),
            "soft_rules": [],
            "hard_rule_triggered": None
        }
        
        # 1. Deterministic HARD Pre-flight (Skip in Mode B)
        if mode in ("A", "C"):
            policy_result = self.policy_engine.evaluate_hard(payload, context)
            if policy_result:
                p_action, msg_type, conf, reason = policy_result
                action = cast(Literal["notify", "digest", "mute"], p_action)
                trace["steps"].append({"step": "hard_policy", "matched": True, "action": action, "rule_reason": reason})
                internal_explanation["hard_rule_triggered"] = reason
                trace["explanation"] = internal_explanation
                
                validated_reason = ReasonGenerator.generate_and_validate(action, msg_type, None, reason)
                logger.info(json.dumps({"event": "decision", "type": "hard_policy", "action": action, "msg_id": msg_id}))
                return FinalDecisionOutput(
                    message_id=msg_id, action=action, message_type=msg_type, 
                    reason=validated_reason, confidence=conf, evidence_message_ids=retrieval_meta.get("evidence", "none")
                ), trace
                
            trace["steps"].append({"step": "hard_policy", "matched": False})
            
            if mode == "A":
                # Mode A: Fallback if no deterministic rule matches
                action = cast(Literal["notify", "digest", "mute"], "digest")
                msg_type = "unknown"
                reason = "No deterministic policy matched. Defaulting to digest."
                trace["explanation"] = internal_explanation
                return FinalDecisionOutput(
                    message_id=msg_id, action=action, message_type=msg_type, 
                    reason=reason, confidence=0.0, evidence_message_ids=retrieval_meta.get("evidence", "none")
                ), trace
        
        # 2. Safety Check (Overrides ALL downstream systems)
        safety = self.get_safety_assessment(payload, context)
        trace["steps"].append({"step": "safety_check", "is_safe": safety.is_safe})
        if not safety.is_safe:
            action = cast(Literal["notify", "digest", "mute"], "mute")
            msg_type, conf = safety.risk_category, 0.95
            reason = ReasonGenerator.generate_and_validate(action, msg_type, None, safety.risk_reason)
            internal_explanation["hard_rule_triggered"] = "Safety Engine Override"
            trace["explanation"] = internal_explanation
            return FinalDecisionOutput(
                message_id=msg_id, action=action, message_type=msg_type, 
                reason=reason, confidence=conf, evidence_message_ids=retrieval_meta.get("evidence", "none")
            ), trace
            
        # 3. Soft Policies Recommendations
        soft_rules = self.policy_engine.evaluate_soft(payload, context)
        internal_explanation["soft_rules"] = soft_rules
            
        # 4. LLM Router
        llm_decision = self.get_llm_decision(payload, context)
        trace["steps"].append({"step": "llm_router", "raw_decision": llm_decision.model_dump()})
        
        # Extract features if using ML Provider
        if isinstance(self.llm_provider, MockMLProvider) and len(self.llm_provider.feature_names) > 0:
            internal_explanation["top_features_triggered"] = ["ML Features Configured"]
        
        # 5. Conflict Resolution & Confidence
        retrieval_conf = retrieval_meta.get("confidence", 0.0)
        final_conf = self.calculate_confidence(llm_decision.llm_confidence, retrieval_conf, soft_rules, context)
        
        action_str = llm_decision.action
        
        # Conflict Resolution Rule: Safety > LLM > Soft Policy
        # If soft rules exist and LLM disagrees with low confidence, soft rule wins
        if final_conf < 0.6 and len(soft_rules) > 0:
            suggested = soft_rules[0]["suggested_action"]
            if suggested != action_str:
                trace["steps"].append({"step": "conflict_resolution", "override": "soft_policy_won"})
                action_str = suggested
                llm_decision.message_type = soft_rules[0]["suggested_type"]
                final_conf += 0.2 # Boost confidence since we leaned on policy
                
        # Low confidence default fallback
        if final_conf < 0.40 and action_str != "digest":
            trace["steps"].append({"step": "conflict_resolution", "override": "low_confidence_digest"})
            action_str = "digest"
            
        action = cast(Literal["notify", "digest", "mute"], action_str)
            
        # 6. Reason Generation
        reason = ReasonGenerator.generate_and_validate(action, llm_decision.message_type)
        trace["steps"].append({"step": "final_output", "action": action, "confidence": final_conf})
        trace["explanation"] = internal_explanation
        
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
        baseline, _ = self.process(msg_id, payload, context, retrieval_meta, mode="C")
        
        ctx_no_trust = dict(context)
        ctx_no_trust["sender_trust_score"] = 0.0
        r1, _ = self.process(msg_id, payload, ctx_no_trust, retrieval_meta, mode="C")
        
        ctx_muted = dict(context)
        ctx_muted["group_muted"] = True
        r2, _ = self.process(msg_id, payload, ctx_muted, retrieval_meta, mode="C")
        
        return {
            "baseline_action": baseline.action,
            "if_zero_trust": r1.action,
            "if_group_muted": r2.action
        }
