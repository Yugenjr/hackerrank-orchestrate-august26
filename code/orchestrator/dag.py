import asyncio
import time
import logging
from typing import Dict, Any, Set, Optional

from .context import PipelineContext, TraceEntry
from .stage import PipelineStage
from .hooks import PipelineHooks

logger = logging.getLogger("DAGOrchestrator")

class DAGOrchestrator:
    """
    Asynchronous DAG executor traversing registered PipelineStages.
    Handles dynamic dependency resolution, retries, and graceful degradation.
    """
    def __init__(self, hooks: Optional[PipelineHooks] = None):
        self.stages: Dict[str, PipelineStage] = {}
        self.hooks = hooks
        
    def register_stage(self, stage: PipelineStage) -> None:
        self.stages[stage.name] = stage
        
    async def _execute_stage(self, stage: PipelineStage, context: PipelineContext) -> Any:
        start_time = time.time()
        attempt = 0
        
        while attempt <= stage.max_retries:
            try:
                if self.hooks:
                    await self.hooks.before_stage(stage.name, context)
                    
                result = await stage.execute(context)
                
                if self.hooks:
                    await self.hooks.after_stage(stage.name, context, result)
                    
                latency_ms = (time.time() - start_time) * 1000
                context.record_trace(TraceEntry(
                    stage=stage.name, latency_ms=latency_ms, status="success",
                    retries=attempt, outputs=result
                ))
                return result
                
            except Exception as e:
                attempt += 1
                if self.hooks:
                    await self.hooks.on_failure(stage.name, context, e)
                    
                if attempt <= stage.max_retries:
                    if self.hooks:
                        await self.hooks.on_retry(stage.name, context, attempt)
                    continue
                    
                # Retry exhausted -> Attempt fallback hook
                try:
                    result = await stage.fallback(context, e)
                    latency_ms = (time.time() - start_time) * 1000
                    context.record_trace(TraceEntry(
                        stage=stage.name, latency_ms=latency_ms, status="fallback_success",
                        retries=attempt-1, degraded=True, outputs=result
                    ))
                    return result
                except Exception as fallback_e:
                    # Total failure of stage
                    latency_ms = (time.time() - start_time) * 1000
                    context.record_trace(TraceEntry(
                        stage=stage.name, latency_ms=latency_ms, status="failed",
                        retries=attempt-1, degraded=True, error=str(fallback_e)
                    ))
                    if stage.is_critical:
                        raise fallback_e  # Halt pipeline
                    return None  # Graceful degradation
                    
    async def run(self, context: PipelineContext) -> PipelineContext:
        """
        Executes the DAG via topological readiness. Independent stages run via asyncio.
        """
        pending: Set[str] = set(self.stages.keys())
        completed: Set[str] = set()
        running_tasks: Dict[str, asyncio.Task] = {}
        
        while pending or running_tasks:
            # 1. Identify ready stages (dependencies met)
            ready = [s for s in pending if all(d in completed for d in self.stages[s].dependencies)]
            
            # 2. Schedule ready stages
            for s in ready:
                pending.remove(s)
                task = asyncio.create_task(self._execute_stage(self.stages[s], context))
                running_tasks[s] = task
                
            if not running_tasks:
                if pending:
                    raise RuntimeError(f"Deadlock detected! Unresolved dependencies for: {pending}")
                break
                
            # 3. Wait for at least one task to finish
            done, _ = await asyncio.wait(running_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                # Map task back to stage name
                stage_name = next(name for name, t in running_tasks.items() if t == task)
                del running_tasks[stage_name]
                
                try:
                    result = task.result()
                    context.set_result(stage_name, result)
                    completed.add(stage_name)
                except Exception as e:
                    # Critical stage failure bubbled up
                    if self.hooks:
                        await self.hooks.on_complete(context)
                    raise e
                    
        if self.hooks:
            await self.hooks.on_complete(context)
            
        return context
