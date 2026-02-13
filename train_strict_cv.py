"""
Strict Cross-Validation dengan Pipeline untuk mencegah Data Leakage
- Menggunakan sklearn.pipeline.Pipeline
- TF-IDF di-fit ulang di setiap fold CV
- Hasil lebih realistis tanpa leakage
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split,
    cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
import json
import os
warnings.filterwarnings('ignore')

print("="*70)
print("STRICT CROSS-VALIDATION - PIPELINE-BASED (NO DATA LEAKAGE)")
print("="*70)

# =====================================================
# LANGKAH 1: LOAD DATA
# =====================================================
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])
print(f"    Total data: {len(df)}")
print(f"    Distribusi label:")
print(f"    - MANUSIA: {(df['label'] == 'MANUSIA').sum()}")
print(f"    - AI: {(df['label'] == 'AI').sum()}")

# Preprocessing
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text'].values
y = df['label_num'].values

# =====================================================
# LANGKAH 2: SPLIT DATA (Train/Test)
# =====================================================
print("\n[2] Split data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Training: {len(X_train)}")
print(f"    Testing: {len(X_test)}")

# =====================================================
# LANGKAH 3: DEFINE PIPELINES
# =====================================================
print("\n[3] Define pipelines (TF-IDF + Classifier)...")

# TF-IDF parameters
tfidf_params = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}

# Define pipelines - TF-IDF akan di-fit ulang di setiap fold
pipelines = {
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        ))
    ]),
    'SVM (RBF Kernel)': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            C=1.0
        ))
    ])
}

# =====================================================
# LANGKAH 4: STRICT CROSS-VALIDATION
# =====================================================
print("\n[4] Running 10-Fold Strict Cross-Validation...")
print("    (TF-IDF di-fit ulang di setiap fold)")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {}

for name, pipeline in pipelines.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    # Cross-validation scores - TF-IDF di-fit ulang per fold
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    # Cross-validated predictions
    cv_predictions = cross_val_predict(
        pipeline, X_train, y_train,
        cv=cv,
        n_jobs=-1
    )

    # Get probabilities for AUC (if supported)
    try:
        cv_proba = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv,
            method='predict_proba',
            n_jobs=-1
        )
        cv_auc = roc_auc_score(y_train, cv_proba[:, 1])
    except:
        cv_auc = None

    # Calculate metrics
    cv_accuracy = cv_scores.mean()
    cv_std = cv_scores.std()
    cv_precision = precision_score(y_train, cv_predictions)
    cv_recall = recall_score(y_train, cv_predictions)
    cv_f1 = f1_score(y_train, cv_predictions)

    # Print fold-by-fold results
    print(f"\nFold-by-Fold Accuracy:")
    for i, score in enumerate(cv_scores, 1):
        bar = '#' * int(score * 50)
        print(f"  Fold {i:2d}: {score*100:5.2f}% {bar}")

    print(f"\nCV Results (NO LEAKAGE):")
    print(f"  Mean Accuracy: {cv_accuracy*100:.2f}% (±{cv_std*100:.2f}%)")
    print(f"  Min - Max:     {cv_scores.min()*100:.2f}% - {cv_scores.max()*100:.2f}%")
    print(f"  Precision:     {cv_precision*100:.2f}%")
    print(f"  Recall:        {cv_recall*100:.2f}%")
    print(f"  F1-Score:      {cv_f1*100:.2f}%")
    if cv_auc:
        print(f"  AUC-ROC:       {cv_auc*100:.2f}%")

    # Store results
    results[name] = {
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_accuracy,
        'cv_std': cv_std,
        'cv_precision': cv_precision,
        'cv_recall': cv_recall,
        'cv_f1': cv_f1,
        'cv_auc': cv_auc
    }

# =====================================================
# LANGKAH 5: FINAL TRAINING & TEST EVALUATION
# =====================================================
print("\n" + "="*70)
print("[5] FINAL TRAINING ON FULL TRAIN SET & TEST EVALUATION")
print("="*70)

final_results = {}

for name, pipeline in pipelines.items():
    print(f"\n--- {name} ---")

    # Fit on full training data
    pipeline.fit(X_train, y_train)

    # Predict on test data
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Get probabilities
    try:
        y_proba = pipeline.predict_proba(X_test)
        test_auc = roc_auc_score(y_test, y_proba[:, 1])
    except:
        test_auc = None

    print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Test Precision: {test_precision*100:.2f}%")
    print(f"  Test Recall:    {test_recall*100:.2f}%")
    print(f"  Test F1-Score:  {test_f1*100:.2f}%")
    if test_auc:
        print(f"  Test AUC-ROC:   {test_auc*100:.2f}%")

    # CV vs Test comparison
    cv_mean = results[name]['cv_mean']
    diff = abs(cv_mean - test_accuracy)
    print(f"\n  CV vs Test Difference: {diff*100:.2f}%")
    if diff < 0.02:
        print(f"  Status: EXCELLENT - Consistent performance")
    elif diff < 0.05:
        print(f"  Status: GOOD - Acceptable difference")
    else:
        print(f"  Status: WARNING - Significant difference")

    final_results[name] = {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'cv_test_diff': diff
    }

# =====================================================
# LANGKAH 6: COMPARISON SUMMARY
# =====================================================
print("\n" + "="*70)
print("RINGKASAN PERBANDINGAN MODEL (STRICT CV)")
print("="*70)

print(f"\n{'Model':<25} {'CV Mean':<15} {'CV Std':<12} {'Test Acc':<12} {'Diff':<10}")
print("-"*74)

for name in pipelines.keys():
    cv_mean = results[name]['cv_mean']
    cv_std = results[name]['cv_std']
    test_acc = final_results[name]['test_accuracy']
    diff = final_results[name]['cv_test_diff']

    print(f"{name:<25} {cv_mean*100:>6.2f}%        ±{cv_std*100:>4.2f}%     {test_acc*100:>6.2f}%      {diff*100:>5.2f}%")

# Find best model
best_model = max(final_results.keys(), key=lambda x: final_results[x]['test_accuracy'])
print(f"\nBEST MODEL (by Test Accuracy): {best_model}")
print(f"  Test Accuracy: {final_results[best_model]['test_accuracy']*100:.2f}%")
print(f"  CV Mean: {results[best_model]['cv_mean']*100:.2f}% (±{results[best_model]['cv_std']*100:.2f}%)")

# =====================================================
# LANGKAH 7: SAVE BEST PIPELINE
# =====================================================
print("\n" + "="*70)
print("[6] SAVING BEST PIPELINE")
print("="*70)

import joblib
os.makedirs('models_strict', exist_ok=True)

# Save best pipeline
best_pipeline = pipelines[best_model]
best_pipeline.fit(X_train, y_train)  # Refit on full training data
joblib.dump(best_pipeline, f'models_strict/best_pipeline_{best_model.lower().replace(" ", "_").replace("(", "").replace(")", "")}.pkl')
print(f"  Saved: models_strict/best_pipeline_{best_model.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

# Save results
all_results = {
    'cv_results': results,
    'test_results': final_results,
    'best_model': best_model,
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test)
}

with open('models_strict/strict_cv_results.json', 'w') as f:
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    json.dump(convert(all_results), f, indent=2)

print(f"  Saved: models_strict/strict_cv_results.json")

# =====================================================
# CONFUSION MATRIX FOR BEST MODEL
# =====================================================
print("\n" + "="*70)
print(f"CONFUSION MATRIX - {best_model}")
print("="*70)

y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

print(f"\n                Predicted: MANUSIA    Predicted: AI")
print(f"Actual: MANUSIA     {cm[0][0]:>8}          {cm[0][1]:>8}")
print(f"Actual: AI          {cm[1][0]:>8}          {cm[1][1]:>8}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn} (MANUSIA correctly identified)")
print(f"False Positives: {fp} (MANUSIA wrongly classified as AI)")
print(f"False Negatives: {fn} (AI wrongly classified as MANUSIA) <-- CRITICAL!")
print(f"True Positives:  {tp} (AI correctly identified)")

# False Negative Analysis
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"\nFalse Negative Rate: {fnr*100:.2f}%")
print(f"(AI yang lolos sebagai MANUSIA - semakin rendah semakin baik)")

print("\n" + "="*70)
print("STRICT CROSS-VALIDATION SELESAI!")
print("="*70)
print("""
KESIMPULAN:
- Cross-validation menggunakan Pipeline mencegah data leakage
- TF-IDF di-fit ulang di setiap fold CV
- Hasil lebih realistis dan dapat diandalkan
- Model siap untuk tahap selanjutnya (threshold tuning)
""")
