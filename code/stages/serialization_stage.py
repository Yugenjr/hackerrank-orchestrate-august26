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
        self.write_count = 0
        self.flush_interval = 100
        
        write_header = not os.path.exists(self.output_file)
        self.f = open(self.output_file, 'a', newline='', encoding='utf-8')
        self.writer = csv.writer(self.f)
        
        if write_header:
            self.writer.writerow(["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"])
            self.f.flush()

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
        
        # evidence_message_ids is typically a list; join by semicolon if needed or assume caller formatted it
        ev_ids = decision["evidence_message_ids"]
        if isinstance(ev_ids, list):
            ev_ids = ";".join(ev_ids)
            
        self.writer.writerow([
            decision["message_id"],
            decision["action"],
            decision["message_type"],
            decision["reason"],
            decision["confidence"],
            ev_ids
        ])
        
        self.write_count += 1
        if self.write_count % self.flush_interval == 0:
            self.f.flush()
            
        return True

    def __del__(self) -> None:
        if hasattr(self, 'f') and self.f and not self.f.closed:
            self.f.flush()
            self.f.close()
