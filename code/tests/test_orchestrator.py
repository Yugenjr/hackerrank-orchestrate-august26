import os
import sys
import pytest
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.context import PipelineContext
from orchestrator.stage import PipelineStage
from orchestrator.dag import DAGOrchestrator

class DummyStage(PipelineStage):
    def __init__(self, name: str, deps=None, duration=0.01, fails=False, critical=True):
        self._name = name
        self._deps = deps or []
        self._duration = duration
        self._fails = fails
        self._critical = critical
        self.retry_count = 0
        
    @property
    def name(self): return self._name
    @property
    def dependencies(self): return self._deps
    @property
    def max_retries(self): return 1
    @property
    def is_critical(self): return self._critical
    
    async def execute(self, context):
        await asyncio.sleep(self._duration)
        if self._fails:
            self.retry_count += 1
            raise ValueError(f"Simulated failure in {self._name}")
        return f"{self._name}_done"

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.anyio
async def test_dag_sequential_and_parallel():
    orch = DAGOrchestrator()
    # A runs first. B and C depend on A and run in parallel. D depends on B and C.
    orch.register_stage(DummyStage("A", duration=0.01))
    orch.register_stage(DummyStage("B", deps=["A"], duration=0.01))
    orch.register_stage(DummyStage("C", deps=["A"], duration=0.01))
    orch.register_stage(DummyStage("D", deps=["B", "C"], duration=0.01))
    
    ctx = PipelineContext(message_id="msg_1", payload={})
    ctx = await orch.run(ctx)
    
    assert ctx.get_result("A") == "A_done"
    assert ctx.get_result("D") == "D_done"
    
    # Check trace logic
    assert len(ctx.trace) == 4
    stages_run = [t.stage for t in ctx.trace]
    assert stages_run[0] == "A"
    assert stages_run[-1] == "D"
    assert set(stages_run[1:3]) == {"B", "C"}  # B and C completed between A and D

@pytest.mark.anyio
async def test_dag_retry_and_degrade():
    orch = DAGOrchestrator()
    s1 = DummyStage("NonCrit", fails=True, critical=False)
    orch.register_stage(s1)
    
    ctx = PipelineContext(message_id="msg_2", payload={})
    ctx = await orch.run(ctx)
    
    assert s1.retry_count == 2 # Initial attempt + 1 retry
    assert ctx.get_result("NonCrit") is None
    assert ctx.trace[0].status == "failed"
    assert ctx.trace[0].degraded

@pytest.mark.anyio
async def test_dag_critical_failure():
    orch = DAGOrchestrator()
    s1 = DummyStage("Crit", fails=True, critical=True)
    orch.register_stage(s1)
    
    ctx = PipelineContext(message_id="msg_3", payload={})
    with pytest.raises(ValueError, match="Simulated failure"):
        await orch.run(ctx)
