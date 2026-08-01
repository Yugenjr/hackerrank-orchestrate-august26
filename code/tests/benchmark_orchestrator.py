import os
import sys
import time
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.context import PipelineContext
from orchestrator.stage import PipelineStage
from orchestrator.dag import DAGOrchestrator

class FastStage(PipelineStage):
    def __init__(self, name: str, deps=None):
        self._name = name
        self._deps = deps or []
    
    @property
    def name(self): return self._name
    @property
    def dependencies(self): return self._deps
    
    async def execute(self, context):
        # Near-zero execution time to measure pure DAG overhead
        return True

async def run_benchmark(iterations=10000):
    orch = DAGOrchestrator()
    # A -> (B, C) -> D
    orch.register_stage(FastStage("A"))
    orch.register_stage(FastStage("B", deps=["A"]))
    orch.register_stage(FastStage("C", deps=["A"]))
    orch.register_stage(FastStage("D", deps=["B", "C"]))
    
    start = time.time()
    for i in range(iterations):
        ctx = PipelineContext(message_id=f"msg_{i}", payload={})
        await orch.run(ctx)
    
    return time.time() - start

if __name__ == "__main__":
    t = asyncio.run(run_benchmark(10000))
    print("--- Orchestrator Overhead Benchmarks ---")
    print(f"10000 DAG executions: {t:.4f} seconds ({10000/t:,.0f} orchestrations/sec)")
