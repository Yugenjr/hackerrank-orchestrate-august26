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
    urgency: Signal[str] = Field(description="Enum: low, medium, high")
    importance: Signal[str] = Field(description="Enum: routine, notable, critical")
    requires_action: Signal[bool]
    deadline_detected: Signal[bool]
    payment_related: Signal[bool]
    event_related: Signal[bool]
    personal_related: Signal[bool]
    business_related: Signal[bool]
    scam_probability: Signal[float]
    spam_probability: Signal[float]

class Contradiction(BaseModel):
    has_contradiction: bool
    description: str
    confidence: float

class MediaContext(BaseModel):
    has_media: bool
    entities: Entities
    structures: StructuralElements
    features: NotificationFeatureVector
    contradiction: Contradiction
