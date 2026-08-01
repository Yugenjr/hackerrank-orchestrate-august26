from abc import ABC
from typing import Any
from .context import PipelineContext

class PipelineHooks(ABC):
    """
    Interface for injecting side-effects into the DAG lifecycle.
    """
    async def before_stage(self, stage_name: str, context: PipelineContext) -> None:
        pass
        
    async def after_stage(self, stage_name: str, context: PipelineContext, result: Any) -> None:
        pass
        
    async def on_failure(self, stage_name: str, context: PipelineContext, error: Exception) -> None:
        pass
        
    async def on_retry(self, stage_name: str, context: PipelineContext, attempt: int) -> None:
        pass
        
    async def on_complete(self, context: PipelineContext) -> None:
        pass
