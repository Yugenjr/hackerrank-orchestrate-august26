from typing import TypeVar, Generic, List
from pydantic import BaseModel, Field

T = TypeVar('T')

class Signal(BaseModel, Generic[T]):
    value: T
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="Snippet of text that justifies this signal")

class Entities(BaseModel):
    people: List[Signal[str]] = Field(default_factory=list)
    organizations: List[Signal[str]] = Field(default_factory=list)
    dates: List[Signal[str]] = Field(default_factory=list)
    times: List[Signal[str]] = Field(default_factory=list)
    locations: List[Signal[str]] = Field(default_factory=list)
    money: List[Signal[str]] = Field(default_factory=list)
    urls: List[Signal[str]] = Field(default_factory=list)
    phones: List[Signal[str]] = Field(default_factory=list)
    qr_codes: List[Signal[str]] = Field(default_factory=list)

class StructuralElements(BaseModel):
    deadlines: List[Signal[str]] = Field(default_factory=list)
    action_items: List[Signal[str]] = Field(default_factory=list)
    events: List[Signal[str]] = Field(default_factory=list)

class NotificationFeatureVector(BaseModel):
    urgency: Signal[str] = Field(default=Signal(value="low", confidence=1.0, evidence="fallback"), description="Enum: low, medium, high")
    importance: Signal[str] = Field(default=Signal(value="routine", confidence=1.0, evidence="fallback"), description="Enum: routine, notable, critical")
    requires_action: Signal[bool] = Field(default=Signal(value=False, confidence=1.0, evidence="fallback"))
    deadline_detected: Signal[bool] = Field(default=Signal(value=False, confidence=1.0, evidence="fallback"))
    payment_related: Signal[bool] = Field(default=Signal(value=False, confidence=1.0, evidence="fallback"))
    event_related: Signal[bool] = Field(default=Signal(value=False, confidence=1.0, evidence="fallback"))
    personal_related: Signal[bool] = Field(default=Signal(value=False, confidence=1.0, evidence="fallback"))
    business_related: Signal[bool] = Field(default=Signal(value=False, confidence=1.0, evidence="fallback"))
    scam_probability: Signal[float] = Field(default=Signal(value=0.0, confidence=1.0, evidence="fallback"))
    spam_probability: Signal[float] = Field(default=Signal(value=0.0, confidence=1.0, evidence="fallback"))

class Contradiction(BaseModel):
    has_contradiction: bool = False
    description: str = "fallback"
    confidence: float = 1.0

class MediaContext(BaseModel):
    has_media: bool
    entities: Entities
    structures: StructuralElements
    features: NotificationFeatureVector
    contradiction: Contradiction
