import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss

df = pd.read_csv('dataset/sample_messages.csv')
df = df.dropna(subset=['action', 'message_type'])

print('\n' + '='*50)
print('EXPERIMENT 6: CALIBRATION METRICS')
print('='*50)

out_c = pd.read_csv('dataset/output_mode_C.csv')
merged = pd.merge(out_c, df, on='message_id', suffixes=('_pred', '_gt'))

# Binary accuracy mapping
merged['is_correct'] = (merged['action_pred'] == merged['action_gt']).astype(int)

# Brier Score
brier = brier_score_loss(merged['is_correct'], merged['confidence_pred'])
print(f"Brier Score (0 is perfect, 1 is worst): {brier:.4f}")

# ECE Calculation
n_bins = 5
merged['bin'] = pd.cut(merged['confidence_pred'], bins=np.linspace(0, 1, n_bins+1), labels=False, include_lowest=True)  # type: ignore
ece = 0.0
print('\nReliability Diagram Data (Bins):')
print('Bin Range | Mean Conf | Accuracy | Count')
print('-'*40)
for b in range(n_bins):
    bin_data = merged[merged['bin'] == b]
    if len(bin_data) > 0:
        mean_conf = bin_data['confidence_pred'].mean()
        acc = bin_data['is_correct'].mean()
        count = len(bin_data)
        weight = count / len(merged)
        ece += weight * np.abs(mean_conf - acc)
        bin_min = b / n_bins
        bin_max = (b+1) / n_bins
        print(f"{bin_min:.1f}-{bin_max:.1f}   | {mean_conf:.3f}     | {acc:.3f}    | {count}")
    else:
        print(f"{b/n_bins:.1f}-{(b+1)/n_bins:.1f}   | N/A       | N/A      | 0")
        
print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

