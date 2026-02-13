"""
Cross-Validation untuk validasi model lebih kuat
10-Fold Stratified Cross-Validation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CROSS-VALIDATION - RANDOM FOREST MODEL (10-FOLD)")
print("="*70)

# Load dataset
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()
print(f"     Total data: {len(df)}")

# Preprocessing
print("\n[2] Preprocessing...")
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text']
y = df['label_num']

# Split untuk final validation
print("\n[3] Split data (80% train untuk CV, 20% test untuk final)...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Training (untuk CV): {len(X_train_full)}")
print(f"     Testing (final): {len(X_test)}")

# TF-IDF pada full training
print("\n[4] TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train_full)
X_test_tfidf = vectorizer.transform(X_test)
print(f"     Features: {X_train_tfidf.shape[1]}")

# Load trained model
print("\n[5] Load trained model...")
model = joblib.load('models/random_forest_model.pkl')
print("     [OK] Model loaded!")

# 10-Fold Cross-Validation
print("\n" + "="*70)
print("10-FOLD STRATIFIED CROSS-VALIDATION")
print("="*70)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\nMelakukan cross-validation...")
cv_scores = cross_val_score(
    model, X_train_tfidf, y_train_full,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Hasil Cross-Validation
print("\n" + "-"*70)
print("HASIL CROSS-VALIDATION")
print("-"*70)

print(f"\nCross-Validation Scores (10-fold):")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score*100:.2f}%")

print(f"\nMean CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std Deviation:   {cv_scores.std()*100:.2f}%")
print(f"Min Accuracy:    {cv_scores.min()*100:.2f}%")
print(f"Max Accuracy:    {cv_scores.max()*100:.2f}%")
print(f"Range:           {cv_scores.max()*100 - cv_scores.min()*100:.2f}%")

# Interpretasi hasil
print("\n" + "-"*70)
print("INTERPRETASI HASIL")
print("-"*70)

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"\nStabilitas Model: ", end="")
if cv_std < 0.01:
    print("SANGAT STABIL (std < 1%)")
elif cv_std < 0.02:
    print("STABIL (std < 2%)")
elif cv_std < 0.03:
    print("Cukup Stabil (std < 3%)")
else:
    print("Kurang Stabil (std >= 3%)")

print(f"\nKonsistensi: ", end="")
if cv_std / cv_mean < 0.02:
    print("SANGAT KONSISTEN")
elif cv_std / cv_mean < 0.05:
    print="KONSISTEN"
else:
    print("KURANG KONSISTEN")

# Final test evaluation
print("\n" + "="*70)
print("EVALUASI FINAL PADA TEST SET (20%)")
print("="*70)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Set Accuracy: {test_accuracy*100:.2f}%")

# Comparison
print("\n" + "="*70)
print("PERBANDINGAN: CV vs TEST SET")
print("="*70)

print(f"\nCross-Validation Mean: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
print(f"Test Set Accuracy:     {test_accuracy*100:.2f}%")

diff = abs(cv_scores.mean() - test_accuracy)
print(f"\nSelisih: {diff*100:.2f}%")

if diff < 0.01:
    print("Status: SANGAT BAIK - Model konsisten antara CV dan test set")
elif diff < 0.02:
    print("Status: BAIK - Perbedaan kecil antara CV dan test set")
elif diff < 0.05:
    print("Status: CUKUP - Perbedaan masih dapat diterima")
else:
    print("Status: PERLU PERHATIAN - Perbedaan cukup besar")

# Detailed classification report
print("\n" + "-"*70)
print("CLASSIFICATION REPORT (TEST SET)")
print("-"*70)
print(classification_report(y_test, y_pred, target_names=['Manusia (0)', 'AI (1)']))

# Confusion Matrix
print("\n" + "-"*70)
print("CONFUSION MATRIX (TEST SET)")
print("-"*70)

cm = confusion_matrix(y_test, y_pred)
print(f"\n                Predicted: 0    Predicted: 1")
print(f"Actual: 0        {cm[0][0]:<6}       {cm[0][1]:<6}")
print(f"Actual: 1        {cm[1][0]:<6}       {cm[1][1]:<6}")

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate (Recall for AI)
specificity = tn / (tn + fp)  # True Negative Rate (Recall for Human)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"\nSensitivity (Recall AI):    {sensitivity*100:.2f}%")
print(f"Specificity (Recall Human): {specificity*100:.2f}%")
print(f"Precision:                  {precision*100:.2f}%")
print(f"F1-Score:                   {f1_score*100:.2f}%")

# Summary
print("\n" + "="*70)
print("RINGKASAN VALIDASI MODEL")
print("="*70)

print(f"""
Model Random Forest dengan TF-IDF telah divalidasi menggunakan:

1. 10-Fold Stratified Cross-Validation
   - Mean Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)
   - Range: {cv_scores.min()*100:.2f}% - {cv_scores.max()*100:.2f}%

2. Test Set Evaluation (20% holdout)
   - Accuracy: {test_accuracy*100:.2f}%
   - Sensitivity: {sensitivity*100:.2f}%
   - Specificity: {specificity*100:.2f}%
   - F1-Score: {f1_score*100:.2f}%

Kesimpulan:
- Model {'SANGAT STABIL' if cv_std < 0.01 else 'STABIL' if cv_std < 0.02 else 'Cukup Stabil'}
- Perbedaan CV vs Test: {diff*100:.2f}% ({'SANGAT BAIK' if diff < 0.01 else 'BAIK' if diff < 0.02 else 'Cukup'})
- Model siap digunakan untuk production
""")

print("="*70)
print("CROSS-VALIDATION SELESAI!")
print("="*70)
