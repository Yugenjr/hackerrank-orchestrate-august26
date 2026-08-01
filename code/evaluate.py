import os
import sys
import json
import time
import asyncio
import pandas as pd
from typing import List, Dict, Any
import csv

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
    ret_engine = RetrievalEngine(loader, use_mocks=False) 
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

async def run_evaluation(mode: str):
    print(f"--- Running Evaluation Mode {mode} ---")
    output_file = f"dataset/output_mode_{mode}.csv"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    orch = setup_pipeline(output_file)
    df = pd.read_csv("dataset/messages.csv")
    
    # Check if 'action' column exists, meaning we have ground truth
    has_ground_truth = 'action' in df.columns
    
    results = []
    
    for idx, row in df.iterrows():
        msg_id = str(row["message_id"])
        # We must drop the ground truth columns before passing to the pipeline!
        payload = {k: v for k, v in row.to_dict().items() if pd.notnull(v)}
        ground_truth = {
            "action": payload.pop("action", "digest"),
            "message_type": payload.pop("message_type", "unknown"),
            "evidence_message_ids": str(payload.pop("evidence_message_ids", "none")).split(";")
        }
        
        ctx = PipelineContext(message_id=msg_id, payload=payload)
        ctx.metadata["evaluation_mode"] = mode
        
        ctx = await orch.run(ctx)
        
        decision_raw = ctx.get_result("DecisionStage")
        decision = decision_raw["decision"] if decision_raw else {}
        
        predicted = {
            "action": decision.get("action", "unknown"),
            "message_type": decision.get("message_type", "unknown"),
            "evidence_message_ids": decision.get("evidence_message_ids", ["none"]),
            "confidence": decision.get("confidence", 0.0)
        }
        
        
        # If no ground truth, we can't calculate correctness
        if not has_ground_truth:
            results.append({
                "message_id": msg_id,
                "mode": mode,
                "predicted": predicted,
                "is_correct": False,
                "confidence": predicted["confidence"],
                "oracle_cause": "No Ground Truth"
            })
            continue
            
        # Exact match logic
        predicted_action = predicted["action"]
        gt_action = ground_truth["action"]
        predicted_type = predicted["message_type"]
        gt_type = ground_truth["message_type"]
        
        is_action_correct = predicted_action == gt_action
        is_type_correct = predicted_type == gt_type
        is_correct = is_action_correct and is_type_correct
        
        oracle_cause = "none"
        if not is_correct:
            pred_ev = set(predicted["evidence_message_ids"])
            gt_ev = set(ground_truth["evidence_message_ids"])
            
            # Simple oracle classification
            if pred_ev != gt_ev and gt_ev != {"none"}:
                oracle_cause = "Retrieval"
            elif not is_action_correct and gt_action in ["mute", "notify"]:
                oracle_cause = "Policies"
            else:
                if mode == "A":
                    oracle_cause = "Missing Policy Rule"
                else:
                    oracle_cause = "LLM"
                    
        results.append({
            "message_id": msg_id,
            "mode": mode,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "confidence": predicted["confidence"],
            "oracle_cause": oracle_cause
        })
        
    return results

async def main():
    modes = ["C", "A", "B"]
    all_results = {}
    
    for m in modes:
        all_results[m] = await run_evaluation(m)
        
    # decision_diff_report
    has_gt = "ground_truth" in all_results["C"][0]
    
    with open("decision_diff_report.md", "w", encoding='utf-8') as f:
        f.write("# Decision Diff Report\n\n")
        f.write("| Message ID | Ground Truth Action | Mode C Action | Mode A Action | Mode B Action |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for i in range(len(all_results["C"])):
            msg_id = all_results["C"][i]["message_id"]
            gt = all_results["C"][i].get("ground_truth", {}).get("action", "unknown")
            c = all_results["C"][i]["predicted"]["action"]
            a = all_results["A"][i]["predicted"]["action"]
            b = all_results["B"][i]["predicted"]["action"]
            f.write(f"| {msg_id} | {gt} | {c} | {a} | {b} |\n")
            
    # error_analysis for Mode C
    if has_gt:
        with open("error_analysis.md", "w", encoding='utf-8') as f:
            f.write("# Error Analysis (Mode C)\n\n")
            errors = [r for r in all_results["C"] if not r["is_correct"]]
            for e in errors:
                f.write(f"## {e['message_id']}\n")
                f.write(f"- Ground Truth: {e.get('ground_truth', {}).get('action', 'unknown')} ({e.get('ground_truth', {}).get('message_type', 'unknown')})\n")
                f.write(f"- Predicted: {e['predicted']['action']} ({e['predicted']['message_type']})\n")
                f.write(f"- Oracle Primary Cause: {e['oracle_cause']}\n\n")
            
    # calculate accuracy
    
    if has_gt:
        acc_c = sum(1 for r in all_results["C"] if r["is_correct"]) / len(all_results["C"])
        acc_a = sum(1 for r in all_results["A"] if r["is_correct"]) / len(all_results["A"])
        acc_b = sum(1 for r in all_results["B"] if r["is_correct"]) / len(all_results["B"])
        
        print(f"Accuracy C (Hybrid): {acc_c:.2f}")
        print(f"Accuracy A (Deterministic): {acc_a:.2f}")
        print(f"Accuracy B (LLM): {acc_b:.2f}")
        
        with open("leaderboard_readiness.md", "w", encoding='utf-8') as f:
            f.write("# Leaderboard Readiness Report\n\n")
            f.write(f"- **Final Hybrid Accuracy on Dataset**: {acc_c*100:.1f}%\n")
            f.write(f"- **Status**: READY FOR SUBMISSION\n")
            
        with open("calibration_report.md", "w", encoding='utf-8') as f:
            f.write("# Confidence Calibration\n\n")
            avg_conf = sum(r["confidence"] for r in all_results["C"]) / len(all_results["C"])
            f.write(f"- **Average Output Confidence**: {avg_conf:.3f}\n")
    else:
        print("No ground truth labels found in messages.csv. Output generated successfully.")
        with open("leaderboard_readiness.md", "w", encoding='utf-8') as f:
            f.write("# Leaderboard Readiness Report\n\n")
            f.write("- **Final Hybrid Accuracy on Dataset**: Unknown (Hidden Labels)\n")
            f.write(f"- **Status**: READY FOR SUBMISSION\n")
        
    print("Evaluation complete. Generated reports.")

if __name__ == "__main__":
    asyncio.run(main())
