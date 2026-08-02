import json
import time
import pandas as pd
from decision_engine import DecisionEngine
from policies import PolicyEngine
from data_loader import DataLoader
from feature_engineering import ContextBuilder

def run_ablation():
    df = pd.read_csv('dataset/sample_messages.csv')
    df = df.dropna(subset=['action'])
    
    loader = DataLoader("dataset")
    loader.load_all()
    ctx_builder = ContextBuilder(loader)
    policy_engine = PolicyEngine()
    engine = DecisionEngine(policy_engine)
    
    total = len(df)
    
    # Run Baseline
    baseline_acc = 0
    baseline_lat = 0
    baseline_conf = 0
    
    # Warmup
    for _ in range(3):
        engine.process("warmup", json.dumps({"message_text": "hello", "ocr_text": "", "asr_transcript": ""}), {}, {"evidence": "none", "confidence": 0.8}, mode="C")
        
    start_t = time.time()
    for idx, row in df.iterrows():
        msg_id = row['message_id']
        payload = json.dumps({"message_text": str(row['message_text']), "ocr_text": "", "asr_transcript": ""})
        ctx_data = ctx_builder.build_context(row.to_dict())
        out, _ = engine.process(msg_id, payload, ctx_data, {"evidence": "msg_123", "confidence": 0.8}, mode="C")
        
        if out.action == row['action']:
            baseline_acc += 1
        baseline_conf += out.confidence
    
    baseline_lat = (time.time() - start_t) / total
    baseline_acc /= total
    baseline_conf /= total
    
    results = {}
    
    # Function to test ablation
    def test_ablation(name, modify_ctx_fn, modify_retrieval_fn):
        acc = 0
        conf = 0
        start_t = time.time()
        for idx, row in df.iterrows():
            msg_id = row['message_id']
            payload = json.dumps({"message_text": str(row['message_text']), "ocr_text": "", "asr_transcript": ""})
            ctx_data = ctx_builder.build_context(row.to_dict())
            
            # Apply modifications
            ctx_data = modify_ctx_fn(ctx_data)
            retrieval = modify_retrieval_fn({"evidence": "msg_123", "confidence": 0.8})
            
            out, _ = engine.process(msg_id, payload, ctx_data, retrieval, mode="C")
            
            if out.action == row['action']:
                acc += 1
            conf += out.confidence
            
        lat = (time.time() - start_t) / total
        acc /= total
        conf /= total
        results[name] = {
            "Accuracy Delta": f"{(acc - baseline_acc)*100:+.1f}%",
            "Latency Delta": f"{(lat - baseline_lat)*1000:+.1f}ms",
            "Confidence Delta": f"{(conf - baseline_conf):+.3f}"
        }

    # 1. Without Retrieval
    test_ablation("Without Retrieval", lambda c: c, lambda r: {"evidence": "none", "confidence": 0.0})
    
    # 2. Without Notification Intelligence
    test_ablation("Without Intelligence", lambda c: {}, lambda r: r)
    
    # 3. Without Business Trust
    def no_business(c):
        c['business_trust_score'] = 0.0
        c['business_verified'] = False
        return c
    test_ablation("Without Business Trust", no_business, lambda r: r)
    
    # 4. Without Deadline Detection
    def no_deadline(c):
        c['deadlines'] = []
        c['urgency'] = "low"
        return c
    test_ablation("Without Deadline Detection", no_deadline, lambda r: r)

    with open('ablation_report.md', 'w') as f:
        f.write("# Ablation Study Report\n\n")
        f.write(f"**Baseline Accuracy**: {baseline_acc*100:.1f}%\n")
        f.write(f"**Baseline Average Latency**: {baseline_lat*1000:.1f}ms\n")
        f.write(f"**Baseline Confidence**: {baseline_conf:.3f}\n\n")
        f.write("| Subsystem Disabled | Accuracy Delta | Latency Delta | Confidence Delta |\n")
        f.write("| --- | --- | --- | --- |\n")
        for k, v in results.items():
            f.write(f"| {k} | {v['Accuracy Delta']} | {v['Latency Delta']} | {v['Confidence Delta']} |\n")

if __name__ == '__main__':
    run_ablation()
