import asyncio
import os
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

# Create Chaos Engines
class ChaosRetrievalEngine(RetrievalEngine):
    def get_evidence(self, msg_id, payload):
        raise Exception("CHAOS INJECTION: Retrieval Failed")

class ChaosRetrievalStage(RetrievalStage):
    @property
    def is_critical(self):
        return False

class ChaosPolicyEngine(PolicyEngine):
    def evaluate_hard(self, payload, context):
        raise Exception("CHAOS INJECTION: Safety Failed")

async def run_chaos():
    output_file = "dataset/chaos_output.csv"
    if os.path.exists(output_file):
        os.remove(output_file)
    
    loader = DataLoader("dataset")
    loader.load_all()
    ctx_builder = ContextBuilder(loader)
    ret_engine = ChaosRetrievalEngine(loader, use_mocks=False)
    dec_engine = DecisionEngine(PolicyEngine()) # Use normal policy for decision to isolate safety fail
    
    orch = DAGOrchestrator()
    orch.register_stage(FeatureExtractionStage(ctx_builder))
    
    orch.register_stage(ChaosRetrievalStage(ret_engine))
    
    # Intelligence fails randomly
    class ChaosIntelStage(NotificationIntelligenceStage):
        @property
        def is_critical(self):
            return False
            
        async def execute(self, ctx):
            raise Exception("CHAOS INJECTION: Intelligence Failed")
            
    orch.register_stage(ChaosIntelStage())
    orch.register_stage(SafetyStage())
    orch.register_stage(DecisionStage(dec_engine))
    orch.register_stage(SerializationStage(output_file))
    
    df = pd.read_csv("dataset/messages.csv")
    success = 0
    crashes = 0
    
    for idx, row in df.iterrows():
        try:
            msg_id = str(row["message_id"])
            payload = {k: v for k, v in row.to_dict().items() if pd.notnull(v)}
            ctx = PipelineContext(message_id=msg_id, payload=payload)
            ctx = await orch.run(ctx)
            success += 1
        except Exception:
            crashes += 1
            
    output_valid = os.path.exists(output_file)
    if output_valid:
        out_df = pd.read_csv(output_file)
        valid_rows = len(out_df) == success
    else:
        valid_rows = False
        
    with open('chaos_report.md', 'w') as f:
        f.write("# Chaos Testing Report\n\n")
        f.write(f"- **Total Evaluated**: {len(df)}\n")
        f.write(f"- **Successful Completions (No Crashes)**: {success}\n")
        f.write(f"- **Crashes**: {crashes}\n")
        f.write(f"- **Valid CSV Output**: {output_valid and valid_rows}\n")
        f.write("\n## Subsystems Injected With Failure\n")
        f.write("- Retrieval Engine\n- Notification Intelligence\n")

if __name__ == '__main__':
    asyncio.run(run_chaos())
