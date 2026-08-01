import yaml
import re
import os
from typing import Dict, Any, Optional, Tuple

class PolicyEngine:
    def __init__(self, yaml_path: str = "policies.yaml"):
        # Allow loading from same directory
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(base_dir, yaml_path)
        with open(full_path, "r", encoding="utf-8") as f:
            self.policies = yaml.safe_load(f)
            
    def evaluate_hard(self, msg_payload: str, context: Dict[str, Any]) -> Optional[Tuple[str, str, float, str]]:
        """
        Returns (action, message_type, confidence, reason) if a HARD deterministic policy overrides the LLM.
        Otherwise returns None.
        """
        overrides = self.policies.get("deterministic_overrides", {})
        
        # Filter for hard rules and sort by priority (highest first)
        hard_rules = {k: v for k, v in overrides.items() if v.get("rule_type") == "hard"}
        sorted_rules = sorted(hard_rules.items(), key=lambda x: x[1].get("priority", 0), reverse=True)
        
        for rule_name, rule in sorted_rules:
            # Check family emergency
            if rule_name == "family_emergency":
                if context.get("interaction_strength", 0.0) >= rule.get("interaction_strength_min", 0.7):
                    if re.search(rule.get("keyword_match", ""), msg_payload, re.IGNORECASE):
                        return (rule["force_action"], rule["force_type"], 1.0, "Deterministic match: Family emergency keyword.")
                        
            # Check OTP
            if rule_name == "otp_2fa":
                if re.search(rule.get("regex_match", ""), msg_payload):
                    return (rule["force_action"], rule["force_type"], 1.0, "Deterministic match: Authentication/OTP pattern.")
                    
            # Check Payment Reminder
            if rule_name == "payment_reminder":
                if context.get("business_trust_score", 0.0) >= rule.get("business_trust_min", 0.8):
                    if re.search(rule.get("intent_match", ""), msg_payload, re.IGNORECASE):
                        return (rule["force_action"], rule["force_type"], 0.95, "Deterministic match: Trusted business payment reminder.")
                        
            # Check Muted Group
            if rule_name == "muted_group":
                if context.get("group_muted", False) == rule.get("group_muted", True):
                    return (rule["force_action"], "group_chat", 1.0, "Deterministic match: User has muted this group.")
                    
        return None

    def evaluate_soft(self, msg_payload: str, context: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Returns a list of soft policy recommendations.
        """
        overrides = self.policies.get("deterministic_overrides", {})
        soft_rules = {k: v for k, v in overrides.items() if v.get("rule_type") == "soft"}
        
        recommendations = []
        for rule_name, rule in soft_rules.items():
            if rule_name == "payment_reminder":
                if context.get("business_trust_score", 0.0) >= rule.get("business_trust_min", 0.8):
                    if re.search(rule.get("intent_match", ""), msg_payload, re.IGNORECASE):
                        recommendations.append({
                            "rule_name": rule_name,
                            "suggested_action": rule["force_action"],
                            "suggested_type": rule["force_type"],
                            "reason": "Deterministic soft match: Trusted business payment reminder."
                        })
                        
            if rule_name == "forwarded_spam":
                fw_count = int(context.get("forwarded_count", 0))
                trust = context.get("sender_trust_score", 0.5)
                if fw_count >= rule.get("forward_count_min", 5) and trust <= rule.get("sender_trust_max", 0.2):
                    recommendations.append({
                        "rule_name": rule_name,
                        "suggested_action": rule["force_action"],
                        "suggested_type": rule["force_type"],
                        "reason": "Deterministic soft match: Heavily forwarded from low-trust sender."
                    })
                    
        return recommendations
