import json
import numpy as np
import pandas as pd
from decision_engine import DecisionEngine
from policies import PolicyEngine

def brier_score(y_true, y_prob):
    return np.mean((y_true - y_prob) ** 2)

def expected_calibration_error(y_true, y_prob, num_bins=10):
    bins = np.linspace(0., 1., num_bins + 1)
    binned = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for b in range(num_bins):
        mask = binned == b
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (np.abs(acc - conf) * np.sum(mask)) / len(y_prob)
    return ece

def run_calibration_report():
    df = pd.read_csv('dataset/sample_messages.csv')
    df = df.dropna(subset=['action'])
    
    policy_engine = PolicyEngine()
    engine = DecisionEngine(policy_engine)
    
    y_true = []
    y_prob = []
    
    for idx, row in df.iterrows():
        msg_id = row['message_id']
        original = str(row['message_text'])
        gt_action = row['action']
        
        ctx = {"urgency": "low", "spam_probability": 0.0, "scam_probability": 0.0}
        p_orig = json.dumps({"message_text": original, "ocr_text": "", "asr_transcript": ""})
        out_orig, _ = engine.process(msg_id, p_orig, ctx, {"evidence": "none", "confidence": 0.8}, mode="C")
        
        is_correct = 1.0 if out_orig.action == gt_action else 0.0
        y_true.append(is_correct)
        y_prob.append(out_orig.confidence)
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    brier = brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    
    with open('confidence_report.md', 'w') as f:
        f.write("# Confidence Calibration Report\n\n")
        f.write(f"- **Expected Calibration Error (ECE)**: {ece:.4f} (Lower is better)\n")
        f.write(f"- **Brier Score**: {brier:.4f} (Lower is better)\n")
        f.write(f"- **Mean Confidence**: {np.mean(y_prob):.4f}\n")
        f.write(f"- **Mean Accuracy**: {np.mean(y_true):.4f}\n")
        
if __name__ == '__main__':
    run_calibration_report()
