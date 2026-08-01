from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TraceEntry(BaseModel):
    stage: str
    latency_ms: float
    status: str
    retries: int = 0
    degraded: bool = False
    outputs: Optional[Any] = None
    cache_hit: bool = False
    error: Optional[str] = None

class PipelineContext(BaseModel):
    message_id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"
    
    # Observability
    trace: List[TraceEntry] = Field(default_factory=list)
    
    # State accumulated across DAG nodes
    results: Dict[str, Any] = Field(default_factory=dict)

    def record_trace(self, entry: TraceEntry) -> None:
        self.trace.append(entry)
        
    def set_result(self, key: str, value: Any) -> None:
        self.results[key] = value
        
    def get_result(self, key: str) -> Any:
        return self.results.get(key)
