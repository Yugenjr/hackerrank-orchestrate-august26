import os
import sys
import json
import time
import asyncio
import pandas as pd
from typing import Optional

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

def setup_pipeline(output_file: str) -> DAGOrchestrator:
    loader = DataLoader("dataset")
    loader.load_all()
    ctx_builder = ContextBuilder(loader)
    ret_engine = RetrievalEngine(loader, use_mocks=True) # use mocks for CI/CD speed testing
    pol_engine = PolicyEngine("policies.yaml")
    dec_engine = DecisionEngine(pol_engine)
    
    orch = DAGOrchestrator()
    orch.register_stage(FeatureExtractionStage(ctx_builder))
    orch.register_stage(RetrievalStage(ret_engine))
    orch.register_stage(NotificationIntelligenceStage())
    orch.register_stage(SafetyStage())
    orch.register_stage(DecisionStage(dec_engine))
    orch.register_stage(SerializationStage(output_file))
    
    return orch

async def process_batch(file_path: str, output_file: str, limit: Optional[int] = None):
    start = time.time()
    # Ensure fresh output
    if os.path.exists(output_file):
        os.remove(output_file)
        
    orch = setup_pipeline(output_file)
    
    # Load dataset
    try:
        df = pd.read_csv(file_path)
    except Exception:
        # Fallback to sample if primary missing in this test env
        df = pd.read_csv("dataset/sample_messages.csv")
        
    if limit:
        df = df.head(limit)
        
    os.makedirs("replay_examples", exist_ok=True)
    
    success = 0
    degraded = 0
    failures = 0
    
    for idx, row in df.iterrows():
        msg_id = str(row["message_id"])
        payload = {str(k): v for k, v in row.to_dict().items() if pd.notnull(v)}
        
        ctx = PipelineContext(message_id=msg_id, payload=payload)
        try:
            ctx = await orch.run(ctx)
            success += 1
            if any(t.degraded for t in ctx.trace):
                degraded += 1
                
            # Save replay artifact
            replay = {
                "message_id": msg_id,
                "pipeline_version": ctx.version,
                "trace": [t.model_dump() for t in ctx.trace],
                "decision": ctx.get_result("DecisionStage")
            }
            with open(f"replay_examples/replay_{msg_id}.json", "w", encoding='utf-8') as f:
                json.dump(replay, f, indent=2, default=str)
                
        except Exception as e:
            print(f"FATAL Pipeline crash for {msg_id}: {e}")
            failures += 1
            
    runtime = time.time() - start
    print("\n--- M3.6 E2E Integration Run ---")
    print(f"Total processed: {success + failures}")
    print(f"Success: {success}")
    print(f"Degraded: {degraded}")
    print(f"Failures: {failures}")
    print(f"Runtime: {runtime:.2f}s ({((success + failures)/runtime):.1f} msg/s)")

if __name__ == "__main__":
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        
    asyncio.run(process_batch("dataset/messages.csv", "dataset/output.csv", limit))
