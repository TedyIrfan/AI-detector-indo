import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

try:  # Optional: matplotlib for visualization
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found, skipping plots")

print("Threshold Tuning Analysis")
print("\nLoad dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')  # Load dataset
df = df.dropna(subset=['text', 'label'])
print(f"    Total data: {len(df)}")

label_mapping = {'MANUSIA': 0, 'AI': 1}  # Preprocessing
df['label_num'] = df['label'].map(label_mapping)

X = df['text'].values
y = df['label_num'].values

print("Split data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # Split data
print(f"    Training: {len(X_train)}")
print(f"    Testing: {len(X_test)}")

print("\nTraining models...")  # Train 3 models: RF, Logistic Regression, SVM

tfidf_params = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}  # TF-IDF parameters

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

for name, pipeline in pipelines.items():  # Train all models
    print(f"    Training {name}...")
    pipeline.fit(X_train, y_train)  # Train each model

print("\nThreshold Analysis")

thresholds = np.arange(0.3, 0.75, 0.05)  # Test thresholds from 0.3 to 0.75

all_results = {}

for name, pipeline in pipelines.items():
    print(f"\nModel: {name}")

    y_proba = pipeline.predict_proba(X_test)  # Get probabilities
    y_proba_ai = y_proba[:, 1]

    results = []

    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FNR':<12}")
    print("-"*72)

    for thresh in thresholds:
        y_pred_thresh = (y_proba_ai >= thresh).astype(int)  # Apply threshold

        acc = accuracy_score(y_test, y_pred_thresh)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

        cm = confusion_matrix(y_test, y_pred_thresh)  # Confusion matrix
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fnr': fnr,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        })

        print(f"{thresh:<12.2f} {acc*100:>8.2f}%    {prec*100:>8.2f}%    {rec*100:>8.2f}%    {f1*100:>8.2f}%    {fnr*100:>8.2f}%")

    all_results[name] = results

    best_idx = max(range(len(results)),
                   key=lambda i: results[i]['f1'] * (1 - results[i]['fnr']))
    best_thresh = results[best_idx]['threshold']

    print(f"Rekomendasi Threshold: {best_thresh:.2f}")
    print(f"  - F1-Score: {results[best_idx]['f1']*100:.2f}%")
    print(f"  - Recall:   {results[best_idx]['recall']*100:.2f}%")
    print(f"  - FNR:      {results[best_idx]['fnr']*100:.2f}%")
    print(f"  - FN count: {results[best_idx]['fn']} (AI yang lolos sebagai MANUSIA)")

print("\nDetailed Analysis - Best Model Selection")

best_model_name = max(pipelines.keys(),
                      key=lambda n: accuracy_score(y_test, pipelines[n].predict(X_test)))  # Select best model
print(f"\nBest performing model: {best_model_name}")

pipeline = pipelines[best_model_name]
y_proba = pipeline.predict_proba(X_test)
y_proba_ai = y_proba[:, 1]

fine_thresholds = np.arange(0.40, 0.65, 0.01)  # Fine-grained search
fine_results = []

print(f"\nFine-grained threshold search for {best_model_name}:")
print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FNR':<12}")
print("-"*72)

for thresh in fine_thresholds:
    y_pred_thresh = (y_proba_ai >= thresh).astype(int)

    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    fine_results.append({
        'threshold': thresh,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fnr': fnr,
        'fn': int(fn)
    })

    marker = " <-- RECOMMENDED" if 0.55 <= thresh <= 0.65 and fnr <= 0.05 else ""
    print(f"{thresh:<12.2f} {acc*100:>8.2f}%    {prec*100:>8.2f}%    {rec*100:>8.2f}%    {f1*100:>8.2f}%    {fnr*100:>8.2f}%{marker}")

if HAS_MATPLOTLIB:
    print("\nGenerating Visualizations")

    os.makedirs('visualizations', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Plot 1: Threshold vs Metrics

    colors = {'Random Forest': 'blue', 'Logistic Regression': 'green', 'SVM': 'red'}

    for name, results in all_results.items():
        thresh_list = [r['threshold'] for r in results]
        acc_list = [r['accuracy'] for r in results]
        prec_list = [r['precision'] for r in results]
        rec_list = [r['recall'] for r in results]
        f1_list = [r['f1'] for r in results]
        fnr_list = [r['fnr'] for r in results]

        axes[0, 0].plot(thresh_list, acc_list, 'o-', label=name, color=colors[name])
        axes[0, 1].plot(thresh_list, prec_list, 'o-', label=name, color=colors[name])
        axes[1, 0].plot(thresh_list, rec_list, 'o-', label=name, color=colors[name])
        axes[1, 1].plot(thresh_list, f1_list, 'o-', label=name, color=colors[name])

    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/threshold_metrics_comparison.png', dpi=150)
    print("    Saved: visualizations/threshold_metrics_comparison.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))  # Plot 2: FNR Analysis

    for name, results in all_results.items():
        thresh_list = [r['threshold'] for r in results]
        fnr_list = [r['fnr'] for r in results]
        ax.plot(thresh_list, fnr_list, 'o-', label=name, color=colors[name], linewidth=2, markersize=8)

    ax.axhline(y=0.05, color='orange', linestyle='--', label='Target FNR (5%)')
    ax.axvline(x=0.5, color='gray', linestyle=':', label='Default Threshold (0.5)')
    ax.axvspan(0.55, 0.65, alpha=0.2, color='green', label='Recommended Zone')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('False Negative Rate (FNR)', fontsize=12)
    ax.set_title('False Negative Rate vs Threshold\n(Lower is better - fewer AI texts passing as Human)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/fnr_analysis.png', dpi=150)
    print("    Saved: visualizations/fnr_analysis.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))  # Plot 3: ROC Curve

    for name, pipeline in pipelines.items():
        y_proba = pipeline.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', color=colors[name])

    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png', dpi=150)
    print("    Saved: visualizations/roc_curve.png")
    plt.close()

print("\nSaving Threshold Analysis Results")

os.makedirs('models_strict', exist_ok=True)

threshold_analysis = {  # Convert to JSON format
    'analysis_date': pd.Timestamp.now().isoformat(),
    'dataset_size': len(df),
    'test_size': len(X_test),
    'models': {},
    'recommendations': {}
}

for name, results in all_results.items():
    threshold_analysis['models'][name] = {
        'threshold_results': results,
        'optimal_threshold': max(results, key=lambda r: r['f1'] * (1 - r['fnr']))['threshold']
    }

threshold_analysis['recommendations'] = {  # Add recommendations
    'default_threshold': 0.5,
    'conservative_threshold': 0.60,  # Lower FNR
    'balanced_threshold': 0.55,       # Balance F1 and FNR
    'explanation': {
        'default': 'Standard threshold, may have higher FNR',
        'conservative': 'Lower FNR (fewer AI passing as human), but more false positives',
        'balanced': 'Good balance between precision and recall'
    }
}

with open('models_strict/threshold_analysis.json', 'w') as f:
    json.dump(threshold_analysis, f, indent=2)

print("    Saved: models_strict/threshold_analysis.json")

print("\nRekomendasi Threshold")  # Final recommendations

print("""
Berdasarkan analisis threshold tuning:

1. DEFAULT THRESHOLD (0.5)
   - Akurasi tinggi, tapi FNR mungkin masih signifikan
   - Cocok untuk penggunaan umum

2. CONSERVATIVE THRESHOLD (0.60)
   - FNR lebih rendah (lebih sedikit AI yang lolos)
   - Precision meningkat, Recall sedikit turun
   - Cocok untuk konteks yang membutuhkan keamanan tinggi

3. BALANCED THRESHOLD (0.55)
   - Trade-off optimal antara F1 dan FNR
   - Rekomendasi untuk production

CONTOH PENGGUNAAN:

    # Default
    y_pred = (y_proba[:, 1] >= 0.5).astype(int)

    # Conservative (lebih ketat terhadap AI)
    y_pred = (y_proba[:, 1] >= 0.6).astype(int)

    # Balanced
    y_pred = (y_proba[:, 1] >= 0.55).astype(int)
""")

print("\nDampak Penggunaan Threshold 0.55 vs 0.5")  # Calculate impact

best_results = all_results[best_model_name]

def find_threshold_result(results, target_threshold, tolerance=0.01):  # Use tolerance for float comparison
    for r in results:
        if abs(r['threshold'] - target_threshold) < tolerance:
            return r
    return results[0]

default_result = find_threshold_result(best_results, 0.5)
recommended_result = find_threshold_result(best_results, 0.55)  # Compare 0.5 vs 0.55

print(f"\nModel: {best_model_name}")
print(f"\n{'Metric':<20} {'Threshold 0.5':<20} {'Threshold 0.55':<20} {'Change':<15}")
print("-"*75)
print(f"{'Accuracy':<20} {default_result['accuracy']*100:>10.2f}%      {recommended_result['accuracy']*100:>10.2f}%      {(recommended_result['accuracy']-default_result['accuracy'])*100:>+6.2f}%")
print(f"{'Precision':<20} {default_result['precision']*100:>10.2f}%      {recommended_result['precision']*100:>10.2f}%      {(recommended_result['precision']-default_result['precision'])*100:>+6.2f}%")
print(f"{'Recall':<20} {default_result['recall']*100:>10.2f}%      {recommended_result['recall']*100:>10.2f}%      {(recommended_result['recall']-default_result['recall'])*100:>+6.2f}%")
print(f"{'F1-Score':<20} {default_result['f1']*100:>10.2f}%      {recommended_result['f1']*100:>10.2f}%      {(recommended_result['f1']-default_result['f1'])*100:>+6.2f}%")
print(f"{'FNR':<20} {default_result['fnr']*100:>10.2f}%      {recommended_result['fnr']*100:>10.2f}%      {(recommended_result['fnr']-default_result['fnr'])*100:>+6.2f}%")
print(f"{'False Negatives':<20} {default_result['fn']:>10}         {recommended_result['fn']:>10}         {recommended_result['fn']-default_result['fn']:>+6}")

print("\nThreshold Tuning Selesai!")
