import json
import pandas as pd
import random
from decision_engine import DecisionEngine
from policies import PolicyEngine

def apply_formatting_noise(text):
    text = str(text)
    mutations = [
        lambda t: t.replace('l', 'I').replace('O', '0'),  # OCR
        lambda t: t + ' ????',  # Emoji
        lambda t: t.lower(),  # Casing
        lambda t: t.replace(' ', '  ').replace('.', ''), # Whitespace/Punctuation
        lambda t: t.replace('urgent', 'urgnt').replace('please', 'pls') # Spelling
    ]
    # Apply a random subset of mutations
    for m in random.sample(mutations, k=2):  # nosec
        text = m(text)
    return text

def apply_semantic_flip(text):
    text = str(text).lower()
    flips = {
        "confirmed": "cancelled",
        "pay": "do not pay",
        "approve": "reject",
        "urgent": "not urgent",
        "tomorrow": "next year",
        "issue": "resolved",
        "remind": "forget"
    }
    for k, v in flips.items():
        if k in text:
            return text.replace(k, v)
    return text # No flip applied

def run_generalization_testing():
    df = pd.read_csv('dataset/sample_messages.csv')
    df = df.dropna(subset=['action'])
    
    policy_engine = PolicyEngine()
    engine = DecisionEngine(policy_engine)
    
    total = len(df)
    
    # 1. Stability Testing
    stable = 0
    for idx, row in df.iterrows():
        msg_id = row['message_id']
        original = str(row['message_text'])
        
        noisy = apply_formatting_noise(original)
        
        ctx = {"urgency": "low", "spam_probability": 0.0, "scam_probability": 0.0}
        
        p_orig = json.dumps({"message_text": original, "ocr_text": "", "asr_transcript": ""})
        out_orig, _ = engine.process(msg_id, p_orig, ctx, {"evidence": "none", "confidence": 0.8}, mode="C")
        
        p_noisy = json.dumps({"message_text": noisy, "ocr_text": "", "asr_transcript": ""})
        out_noisy, _ = engine.process(msg_id, p_noisy, ctx, {"evidence": "none", "confidence": 0.8}, mode="C")
        
        if out_orig.action == out_noisy.action:
            stable += 1
            
    # 2. Semantic Sensitivity Testing
    flipped_count = 0
    sensitive = 0
    for idx, row in df.iterrows():
        msg_id = row['message_id']
        original = str(row['message_text'])
        
        flipped = apply_semantic_flip(original)
        if flipped == original.lower():
            continue # Skip if no flip was possible
            
        flipped_count += 1
        
        ctx = {"urgency": "low", "spam_probability": 0.0, "scam_probability": 0.0}
        p_orig = json.dumps({"message_text": original, "ocr_text": "", "asr_transcript": ""})
        out_orig, _ = engine.process(msg_id, p_orig, ctx, {"evidence": "none", "confidence": 0.8}, mode="C")
        
        p_flipped = json.dumps({"message_text": flipped, "ocr_text": "", "asr_transcript": ""})
        out_flipped, _ = engine.process(msg_id, p_flipped, ctx, {"evidence": "none", "confidence": 0.8}, mode="C")
        
        if out_orig.action != out_flipped.action:
            sensitive += 1
            
    stability_score = stable / total
    sensitivity_score = sensitive / flipped_count if flipped_count > 0 else 0
    
    with open('generalization_report.md', 'w') as f:
        f.write("# Generalization Testing Report\n\n")
        f.write(f"- **Prediction Stability Score**: {stability_score:.4f} (Target: > 0.90)\n")
        f.write(f"- **Semantic Sensitivity Score**: {sensitivity_score:.4f} (Target: > 0.50)\n")
        f.write("\n## Details\n")
        f.write(f"- Stable predictions under formatting noise: {stable}/{total}\n")
        f.write(f"- Flipped predictions under semantic inversion: {sensitive}/{flipped_count}\n")

if __name__ == '__main__':
    run_generalization_testing()
