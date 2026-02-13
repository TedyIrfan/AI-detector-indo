"""
Group-based Cross-Validation untuk mencegah data leakage antar sumber
- Memastikan data dari sumber yang sama tidak bocor antara train/test
- Menggunakan GroupKFold dari sklearn
- Berguna jika ada sumber data yang berpotensi memiliki pattern serupa
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, train_test_split, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
import json
import os
warnings.filterwarnings('ignore')

print("="*70)
print("GROUP-BASED CROSS-VALIDATION")
print("(Mencegah leakage dari sumber data yang sama)")
print("="*70)

# =====================================================
# LANGKAH 1: LOAD DATA
# =====================================================
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])
print(f"    Total data: {len(df)}")

# =====================================================
# LANGKAH 2: DETECT/CREATE SOURCE GROUPS
# =====================================================
print("\n[2] Detecting source groups...")

# Try to detect source from text characteristics
# If no explicit source column, we'll create synthetic groups based on text patterns

def detect_source(text):
    """
    Heuristik untuk mendeteksi sumber teks:
    - IndoSum: biasanya berita dengan format "KOTA, MEDIA -"
    - Twitter: ada @mentions, hashtags, singkatan
    - Reddit: informal, bahasa campuran
    - AI: pola formal, struktur teratur
    """
    text = str(text).lower()

    # Check for news format (IndoSum pattern)
    if any(pattern in text[:100] for pattern in ['cnn indonesia', 'kompas', 'antara', 'republika', 'tribun']):
        return 'indosum_news'

    # Check for Twitter patterns
    if '@' in text or text.count('#') >= 2:
        return 'twitter'

    # Check for formal AI patterns
    formal_indicators = [
        'implementasi', 'optimalisasi', 'transformasi', 'peningkatan',
        'pengembangan', 'berkelanjutan', 'komprehensif', 'strategis'
    ]
    if sum(1 for ind in formal_indicators if ind in text) >= 3:
        # Could be AI - check label
        return 'ai_formal'

    # Default groups based on content length and structure
    if len(text) > 500:
        return 'long_form'
    elif len(text) > 200:
        return 'medium_form'
    else:
        return 'short_form'

# Apply source detection
df['source_group'] = df['text'].apply(detect_source)

# Also add label-based grouping
# This ensures AI and human texts are properly separated in groups
df['source_group'] = df.apply(
    lambda row: f"{row['label'].lower()}_{row['source_group']}",
    axis=1
)

print(f"\nDistribusi Source Groups:")
for group, count in df['source_group'].value_counts().items():
    print(f"  {group}: {count}")

# =====================================================
# LANGKAH 3: PREPARE DATA
# =====================================================
print("\n[3] Preparing data...")

label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text'].values
y = df['label_num'].values
groups = df['source_group'].values

# =====================================================
# LANGKAH 4: GROUP-BASED CV
# =====================================================
print("\n" + "="*70)
print("[4] GROUP-BASED CROSS-VALIDATION")
print("="*70)
print("\nNote: Data dari group yang sama akan SELALU di train ATAU test,")
print("      tidak pernah di keduanya sekaligus.")

# Define pipelines
tfidf_params = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}

pipelines = {
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ]),
    'SVM': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42, C=1.0))
    ])
}

# Number of groups
n_groups = len(np.unique(groups))
n_splits = min(5, n_groups)  # Can't have more splits than groups

print(f"\nNumber of unique groups: {n_groups}")
print(f"Number of splits: {n_splits}")

# GroupKFold
gkf = GroupKFold(n_splits=n_splits)

results = {}

for name, pipeline in pipelines.items():
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    fold_scores = []

    print(f"\nFold-by-Fold Results:")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Train and test groups (for display)
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        # Fit pipeline
        pipeline.fit(X_train_fold, y_train_fold)

        # Predict
        y_pred = pipeline.predict(X_test_fold)
        acc = accuracy_score(y_test_fold, y_pred)

        fold_scores.append(acc)

        print(f"  Fold {fold}: {acc*100:.2f}%")
        print(f"    Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
        print(f"    Train size: {len(X_train_fold)}, Test size: {len(X_test_fold)}")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\nGroup CV Results:")
    print(f"  Mean Accuracy: {mean_score*100:.2f}% (±{std_score*100:.2f}%)")
    print(f"  Min - Max:     {min(fold_scores)*100:.2f}% - {max(fold_scores)*100:.2f}%")

    results[name] = {
        'fold_scores': fold_scores,
        'mean': mean_score,
        'std': std_score
    }

# =====================================================
# LANGKAH 5: COMPARISON WITH STANDARD CV
# =====================================================
print("\n" + "="*70)
print("[5] COMPARISON: GROUP CV vs STANDARD CV")
print("="*70)

from sklearn.model_selection import StratifiedKFold

standard_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<25} {'Group CV':<20} {'Standard CV':<20} {'Diff':<15}")
print("-"*80)

for name, pipeline in pipelines.items():
    # Standard CV scores
    std_scores = cross_val_score(pipeline, X, y, cv=standard_cv, scoring='accuracy', n_jobs=-1)
    std_mean = std_scores.mean()

    # Group CV scores
    group_mean = results[name]['mean']

    diff = group_mean - std_mean

    print(f"{name:<25} {group_mean*100:>6.2f}% (±{results[name]['std']*100:>4.2f}%)  {std_mean*100:>6.2f}% (±{std_scores.std()*100:>4.2f}%)  {diff*100:>+6.2f}%")

print("""
\nINTERPRETASI:
- Jika Group CV < Standard CV: Ada kemungkinan leakage dari sumber yang sama
- Jika Group CV ~ Standard CV: Model generalisasi dengan baik antar sumber
- Jika Group CV > Standard CV: Jarang terjadi, mungkin kebetulan distribusi group
""")

# =====================================================
# LANGKAH 6: FINAL TRAINING
# =====================================================
print("\n" + "="*70)
print("[6] FINAL TRAINING ON FULL DATASET")
print("="*70)

# Split by groups for final evaluation
unique_groups = np.unique(groups)
np.random.seed(42)
np.random.shuffle(unique_groups)

# Use 80% of groups for training, 20% for testing
n_train_groups = int(len(unique_groups) * 0.8)
train_groups_set = set(unique_groups[:n_train_groups])

train_mask = np.array([g in train_groups_set for g in groups])
test_mask = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
groups_train, groups_test = groups[train_mask], groups[test_mask]

print(f"\nFinal split by groups:")
print(f"  Training: {len(X_train)} samples from {len(set(groups_train))} groups")
print(f"  Testing:  {len(X_test)} samples from {len(set(groups_test))} groups")

# Train best model
best_model_name = max(results.keys(), key=lambda k: results[k]['mean'])
best_pipeline = pipelines[best_model_name]

print(f"\nTraining best model ({best_model_name})...")
best_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = best_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\nTest Results (groups not seen during training):")
print(f"  Accuracy:  {test_acc*100:.2f}%")
print(f"  Precision: {test_precision*100:.2f}%")
print(f"  Recall:    {test_recall*100:.2f}%")
print(f"  F1-Score:  {test_f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted: MANUSIA    Predicted: AI")
print(f"Actual: MANUSIA     {cm[0][0]:>8}          {cm[0][1]:>8}")
print(f"Actual: AI          {cm[1][0]:>8}          {cm[1][1]:>8}")

tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"\nFalse Negative Rate: {fnr*100:.2f}%")

# =====================================================
# LANGKAH 7: SAVE RESULTS
# =====================================================
print("\n" + "="*70)
print("[7] SAVING RESULTS")
print("="*70)

os.makedirs('models_strict', exist_ok=True)

# Save results
group_cv_results = {
    'group_cv_results': {
        name: {
            'fold_scores': [float(s) for s in r['fold_scores']],
            'mean': float(r['mean']),
            'std': float(r['std'])
        } for name, r in results.items()
    },
    'test_results': {
        'accuracy': float(test_acc),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1': float(test_f1),
        'fnr': float(fnr)
    },
    'n_groups': int(n_groups),
    'train_groups': len(train_groups_set),
    'test_groups': len(set(groups_test))
}

with open('models_strict/group_cv_results.json', 'w') as f:
    json.dump(group_cv_results, f, indent=2)

print("    Saved: models_strict/group_cv_results.json")

print("\n" + "="*70)
print("GROUP-BASED CV COMPLETED!")
print("="*70)
print("""
KESIMPULAN:
- Group-based CV memastikan tidak ada data leakage antar sumber
- Hasil lebih konservatif tapi lebih realistis
- Cocok untuk mengukur generalisasi model ke sumber data baru

Jika perlu menambahkan kolom 'source' secara eksplisit ke dataset,
modifikasi file CSV dan tambahkan kolom source_group sebelum training.
""")
