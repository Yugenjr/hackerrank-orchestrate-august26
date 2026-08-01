from abc import ABC, abstractmethod
from typing import Any, List
from .context import PipelineContext

class PipelineStage(ABC):
    """
    Contract for any module executing in the DAG orchestrator.
    Enforces async execution, dependency listing, and fallback logic.
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__
        
    @property
    def is_critical(self) -> bool:
        """If True, failure halts the pipeline. If False, failure causes graceful degradation."""
        return True
        
    @property
    def max_retries(self) -> int:
        return 0
        
    @property
    def dependencies(self) -> List[str]:
        """List of stage names that must complete before this one executes."""
        return []
        
    @abstractmethod
    async def execute(self, context: PipelineContext) -> Any:
        """Primary business logic for the stage."""
        pass
        
    async def fallback(self, context: PipelineContext, error: Exception) -> Any:
        """
        Attempt a fallback action before total degradation.
        By default, it re-raises the error to trigger degradation logic in the runner.
        """
        raise error
