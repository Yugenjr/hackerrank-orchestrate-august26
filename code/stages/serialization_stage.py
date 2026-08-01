import os
import sys
import csv
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.stage import PipelineStage
from orchestrator.context import PipelineContext

class SerializationStage(PipelineStage):
    def __init__(self, output_file: str):
        self.output_file = output_file
        # Ensure file exists with headers
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"])

    @property
    def name(self) -> str:
        return "SerializationStage"
        
    @property
    def dependencies(self):
        return ["DecisionStage"]
        
    @property
    def is_critical(self):
        return True
        
    async def execute(self, context: PipelineContext) -> Any:
        decision_wrapper = context.get_result("DecisionStage")
        if not decision_wrapper:
            raise ValueError("No DecisionStage output found")
            
        decision = decision_wrapper["decision"]
        
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                decision["message_id"],
                decision["action"],
                decision["message_type"],
                decision["reason"],
                decision["confidence"],
                decision["evidence_message_ids"]
            ])
            
        return True
