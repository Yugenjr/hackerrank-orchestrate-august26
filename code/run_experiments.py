import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

print('='*50)
print('EXPERIMENT 4: FEATURE IMPORTANCE')
print('='*50)

df = pd.read_csv('dataset/sample_messages.csv')
df = df.dropna(subset=['action', 'message_type'])
texts = df['message_text'].fillna("").astype(str)
y = df['action']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1,2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(texts, y)

tfidf = pipeline.named_steps['tfidf']
clf = pipeline.named_steps['clf']
feature_names = tfidf.get_feature_names_out()
importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]
print('Top 15 Features for Action Routing:')
for i in range(15):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

print('\n' + '='*50)
print('EXPERIMENT 7: CROSS-VALIDATION')
print('='*50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, texts, y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy Scores: {scores}")
print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

print('\n' + '='*50)
print('EXPERIMENT 6: CALIBRATION METRICS')
print('='*50)

out_c = pd.read_csv('dataset/output_mode_C.csv')
merged = pd.merge(out_c, df, on='message_id', suffixes=('_pred', '_gt'))

# Binary accuracy mapping
merged['is_correct'] = (merged['action_pred'] == merged['action_gt']).astype(int)

# Brier Score
brier = brier_score_loss(merged['is_correct'], merged['confidence'])
print(f"Brier Score (0 is perfect, 1 is worst): {brier:.4f}")

# ECE Calculation
n_bins = 5
merged['bin'] = pd.cut(merged['confidence'], bins=np.linspace(0, 1, n_bins+1), labels=False, include_lowest=True)
ece = 0.0
print('\nReliability Diagram Data (Bins):')
print('Bin Range | Mean Conf | Accuracy | Count')
print('-'*40)
for b in range(n_bins):
    bin_data = merged[merged['bin'] == b]
    if len(bin_data) > 0:
        mean_conf = bin_data['confidence'].mean()
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

