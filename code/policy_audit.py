import pandas as pd
from policies import PolicyEngine

def run_policy_audit():
    df = pd.read_csv('dataset/sample_messages.csv')
    df = df.dropna(subset=['action'])
    
    engine = PolicyEngine("policies.yaml")
    
    # Store stats per rule
    stats = {}
    
    from data_loader import DataLoader
    from feature_engineering import ContextBuilder
    loader = DataLoader("dataset")
    loader.load_all()
    ctx_builder = ContextBuilder(loader)
    
    total = len(df)
    
    for idx, row in df.iterrows():
        gt_action = row['action']
        msg_payload = str(row.get('message_text', ''))
        
        # Build context
        row_dict = row.to_dict()
        ctx_data = ctx_builder.build_context(row_dict)
        
        # Evaluate hard rules
        hard_res = engine.evaluate_hard(msg_payload, ctx_data)
        if hard_res:
            hard_action, _, _, _ = hard_res
            hard_rule = "hard_match" # Need to extract rule name, but wait, evaluate_hard doesn't return rule name in tuple.
            # I will modify policies.py temporarily or parse the reason string.
            reason = hard_res[3]
            if 'Family emergency' in reason:
                hard_rule = 'family_emergency'
            elif 'OTP' in reason:
                hard_rule = 'otp_2fa'
            elif 'payment' in reason:
                hard_rule = 'payment_reminder'
            elif 'muted' in reason:
                hard_rule = 'muted_group'
            
            if hard_rule not in stats:
                stats[hard_rule] = {'triggers': 0, 'correct': 0, 'type': 'hard'}
            stats[hard_rule]['triggers'] += 1
            if hard_action == gt_action:
                stats[hard_rule]['correct'] += 1
                
        # Evaluate soft rules
        soft_recs = engine.evaluate_soft(msg_payload, ctx_data)
        for rec in soft_recs:
            s_rule = rec['rule_name']
            s_action = rec['suggested_action']
            if s_rule not in stats:
                stats[s_rule] = {'triggers': 0, 'correct': 0, 'type': 'soft'}
            stats[s_rule]['triggers'] += 1
            if s_action == gt_action:
                stats[s_rule]['correct'] += 1
                
    with open('policy_audit.md', 'w') as f:
        f.write("# Policy Audit Report\n\n")
        f.write(f"Total messages evaluated: {total}\n\n")
        f.write("| Rule Name | Type | Triggers | Precision | Recommendation |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for rule, data in stats.items():
            precision = data['correct'] / data['triggers'] if data['triggers'] > 0 else 0
            rec = "Keep" if precision >= 0.8 else "Weaken/Remove"
            if data['type'] == 'hard' and precision < 0.95:
                rec = "Convert to Soft"
            f.write(f"| {rule} | {data['type']} | {data['triggers']} | {precision:.2f} | {rec} |\n")

if __name__ == '__main__':
    run_policy_audit()
