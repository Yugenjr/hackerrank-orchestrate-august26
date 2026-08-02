import asyncio
import os
import time
import tracemalloc
import pandas as pd
from orchestrator.context import PipelineContext
from orchestrator.dag import DAGOrchestrator
from data_loader import DataLoader
from feature_engineering import ContextBuilder
from retrieval import RetrievalEngine
from decision_engine import DecisionEngine
from policies import PolicyEngine

from stages.feature_stage import FeatureExtractionStage
from stages.retrieval_stage import RetrievalStage
from stages.intelligence_stage import NotificationIntelligenceStage
from stages.safety_stage import SafetyStage
from stages.decision_stage import DecisionStage
from stages.serialization_stage import SerializationStage

async def run_benchmark():
    tracemalloc.start()
    
    t0 = time.time()
    # 1. Startup phase
    loader = DataLoader("dataset")
    loader.load_all()
    ctx_builder = ContextBuilder(loader)
    ret_engine = RetrievalEngine(loader, use_mocks=False)
    pol_engine = PolicyEngine()
    dec_engine = DecisionEngine(pol_engine)
    
    output_file = "dataset/benchmark_output.csv"
    if os.path.exists(output_file):
        os.remove(output_file)
    
    orch = DAGOrchestrator()
    orch.register_stage(FeatureExtractionStage(ctx_builder))
    orch.register_stage(RetrievalStage(ret_engine))
    orch.register_stage(NotificationIntelligenceStage())
    orch.register_stage(SafetyStage())
    orch.register_stage(DecisionStage(dec_engine))
    ser_stage = SerializationStage(output_file)
    orch.register_stage(ser_stage)
    
    startup_time = time.time() - t0
    
    # 2. Process phase
    df = pd.read_csv("dataset/messages.csv")
    total = len(df)
    
    t1 = time.time()
    
    serialization_times = []
    
    for idx, row in df.iterrows():
        msg_id = str(row["message_id"])
        payload = {k: v for k, v in row.to_dict().items() if pd.notnull(v)}
        ctx = PipelineContext(message_id=msg_id, payload=payload)
        
        ctx = await orch.run(ctx)
        
        # We can estimate serialization time as the last stage latency from trace
        trace = [t for t in ctx.trace if t.stage == "SerializationStage"]
        if trace:
            serialization_times.append(trace[0].latency_ms)
            
    total_time = time.time() - t1
    avg_latency = (total_time / total) * 1000
    throughput = total / total_time
    avg_ser_time = sum(serialization_times) / len(serialization_times) if serialization_times else 0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_ram_mb = peak_mem / 10**6
    
    with open('performance_report.md', 'w') as f:
        f.write("# Performance Benchmark Report\n\n")
        f.write(f"- **Startup Time**: {startup_time:.2f}s\n")
        f.write(f"- **Total Processing Time ({total} msgs)**: {total_time:.2f}s\n")
        f.write(f"- **Average Latency**: {avg_latency:.2f}ms/msg\n")
        f.write(f"- **Throughput**: {throughput:.2f} msgs/sec\n")
        f.write(f"- **Average Serialization Time**: {avg_ser_time:.2f}ms/msg\n")
        f.write(f"- **Peak RAM Usage**: {peak_ram_mb:.2f}MB\n")
        f.write("- **Cache Hit Ratio**: Not applicable (mock caches in python)\n")

if __name__ == '__main__':
    asyncio.run(run_benchmark())
