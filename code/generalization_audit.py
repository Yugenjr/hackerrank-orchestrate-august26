import json
import logging
import pandas as pd
import random
from decision_engine import DecisionEngine
from policies import PolicyEngine
from schemas import FinalDecisionOutput

def perturb_text(text):
    if not isinstance(text, str):
        return text
    
    # 1. Simulate OCR Noise (swap I and l, 0 and O)
    text = text.replace('l', 'I').replace('O', '0')
    
    # 2. Add emojis
    emojis = ['??', '??', '??', '??']
    if random.random() > 0.5:
        text = text + ' ' + random.choice(emojis)
        
    # 3. Spelling variations
    if 'urgent' in text.lower():
        text = text.replace('urgent', 'urgnt')
        
    return text

def run_audit():
    df = pd.read_csv('dataset/sample_messages.csv')
    df = df.dropna(subset=['action', 'message_type'])
    
    policy_engine = PolicyEngine()
    engine = DecisionEngine(policy_engine)
    
    total = len(df)
    stable = 0
    
    for idx, row in df.iterrows():
        msg_id = row['message_id']
        original_text = str(row['message_text'])
        
        # Baseline run
        payload = json.dumps({"message_text": original_text, "ocr_text": "", "asr_transcript": ""})
        context = {"urgency": "low", "spam_probability": 0.0, "scam_probability": 0.0}
        baseline_out, _ = engine.process(msg_id, payload, context, {"evidence": "none", "confidence": 0.8}, mode="C")
        
        # Perturbed run
        perturbed = perturb_text(original_text)
        payload_p = json.dumps({"message_text": perturbed, "ocr_text": "", "asr_transcript": ""})
        pert_out, _ = engine.process(msg_id, payload_p, context, {"evidence": "none", "confidence": 0.8}, mode="C")
        
        if baseline_out.action == pert_out.action:
            stable += 1
            
    stability_score = stable / total
    print(f"==================================================")
    print(f"GENERALIZATION AUDIT: ROBUSTNESS & STABILITY")
    print(f"==================================================")
    print(f"Total evaluated: {total}")
    print(f"Stable predictions (no change across perturbations): {stable}")
    print(f"Prediction Stability Score: {stability_score:.4f}")
    
    with open('generalization_audit_report.md', 'w') as f:
        f.write("# Generalization Audit Report\n\n")
        f.write(f"- **Total Evaluated**: {total}\n")
        f.write(f"- **Stable Predictions**: {stable}\n")
        f.write(f"- **Prediction Stability Score**: {stability_score:.4f}\n")
        f.write("\n## Perturbations Applied\n")
        f.write("- OCR Noise (I/l, 0/O swaps)\n- Random Emoji Insertion\n- Spelling Variations (e.g. urgent -> urgnt)\n")

if __name__ == '__main__':
    run_audit()
