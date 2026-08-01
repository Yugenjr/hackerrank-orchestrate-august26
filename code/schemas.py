from typing import Literal
from pydantic import BaseModel, Field

class SafetyAgentOutput(BaseModel):
    is_safe: bool = Field(description="True if message is completely safe")
    risk_category: Literal["none", "phishing", "spam", "impersonation", "high_forward_noise"]
    risk_reason: str = Field(max_length=100)

class DecisionRouterOutput(BaseModel):
    action: Literal["notify", "digest", "mute"]
    message_type: str = Field(description="e.g. personal, urgent, event, payment, business_update, promotion, spam")
    # Rule: LLM does not generate final system confidence, only its internal reasoning confidence.
    llm_confidence: float = Field(ge=0.0, le=1.0)

class FinalDecisionOutput(BaseModel):
    message_id: str
    action: Literal["notify", "digest", "mute"]
    message_type: str
    reason: str = Field(max_length=150)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: str
