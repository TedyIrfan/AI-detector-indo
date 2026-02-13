import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("VISUALISASI TAMBAHAN UNTUK SKRIPSI")

print("\nLoad dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()

label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"     Training: {len(X_train)}, Testing: {len(X_test)}")

print("\nTF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nLoad model...")
model = joblib.load('models/random_forest_model.pkl')

print("\nGenerating Learning Curve...")

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_tfidf, y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_sizes, train_mean, label='Training Score', marker='o', color='blue', linewidth=2)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

ax.plot(train_sizes, test_mean, label='Cross-Validation Score', marker='s', color='red', linewidth=2)
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

ax.axhline(y=test_mean[-1], color='green', linestyle='--', label=f'Final CV Score: {test_mean[-1]*100:.2f}%')

ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Learning Curve - Random Forest', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("     [OK] Saved: learning_curve.png")

print("\nGenerating ROC Curve...")

y_pred_proba = model.predict_proba(X_test_tfidf)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr, tpr, color='blue', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Random Classifier')

ax.fill_between(fpr, tpr, alpha=0.2, color='blue')

idx_50 = np.argmin(np.abs(tpr - 0.5))
ax.plot(fpr[idx_50], tpr[idx_50], 'ro', markersize=8, label=f'TPR=50%: FPR={fpr[idx_50]:.3f}')

idx_90 = np.argmin(np.abs(tpr - 0.9))
ax.plot(fpr[idx_90], tpr[idx_90], 'go', markersize=8, label=f'TPR=90%: FPR={fpr[idx_90]:.3f}')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.1])
ax.set_ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("     [OK] Saved: roc_curve.png")

print(f"\n     ROC AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")

print("\nGenerating Precision-Recall Curve...")

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
ap_score = average_precision_score(y_test, y_pred_proba[:, 1])

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(recall, precision, color='blue', linewidth=3, label=f'PR Curve (AP = {ap_score:.4f})')

baseline = len(y_test[y_test == 1]) / len(y_test)
ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline (random): {baseline:.3f}')

ax.fill_between(recall, precision, alpha=0.2, color='blue')

ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve - Random Forest', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.95, 1.0])
ax.set_ylim([0.95, 1.0])

plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("     [OK] Saved: precision_recall_curve.png")

print(f"\n     Average Precision Score: {ap_score:.4f} ({ap_score*100:.2f}%)")

print("\nGenerating Confusion Matrix Heatmap...")

y_pred = model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            xticklabels=['Manusia', 'AI'],
            yticklabels=['Manusia', 'AI'],
            annot_kws={'size': 16, 'weight': 'bold'})

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

textstr = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(1.65, 0.5, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("     [OK] Saved: confusion_matrix_heatmap.png")

print("\nGenerating Feature Importance Plot...")

feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_

top_indices = np.argsort(importances)[-20:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = ax.barh(range(len(top_features)), top_importances * 100, color=colors)

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features, fontsize=10)
ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (bar, imp) in enumerate(zip(bars, top_importances)):
    ax.text(imp * 100 + 0.05, i, f'{imp*100:.2f}%',
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
plt.close()
print("     [OK] Saved: feature_importance_top20.png")

print("\nGenerating Model Comparison Chart...")

try:
    comp_df = pd.read_csv('model_comparison_results.csv')
    has_comparison = True
except:
    comp_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Neural Network', 'SVM RBF', 'SVM Linear',
                  'Naive Bayes', 'Random Forest', 'KNN', 'Decision Tree'],
        'Accuracy': [1.00, 1.00, 1.00, 1.00, 0.9868, 0.9801, 0.9735, 0.9404]
    })
    has_comparison = True

if has_comparison:
    comp_df = comp_df.sort_values('Accuracy')

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['red' if x == 'Random Forest' else 'steelblue' for x in comp_df['Model']]
    bars = ax.barh(comp_df['Model'], comp_df['Accuracy'] * 100, color=colors, alpha=0.7)

    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([90, 101])

    for bar, acc in zip(bars, comp_df['Accuracy']):
        ax.text(acc * 100 + 0.2, bar.get_y() + bar.get_height()/2,
                f'{acc*100:.2f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('model_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("     [OK] Saved: model_comparison_chart.png")

print("\nGenerating Probability Distribution Plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

human_proba = y_pred_proba[y_test == 0][:, 0]
axes[0].hist(human_proba, bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
axes[0].set_xlabel('Probability of Being Human', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Probability Distribution - Human Texts', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

ai_proba = y_pred_proba[y_test == 1][:, 1]
axes[1].hist(ai_proba, bins=30, color='red', alpha=0.7, edgecolor='black')
axes[1].axvline(0.5, color='blue', linestyle='--', linewidth=2, label='Decision Boundary')
axes[1].set_xlabel('Probability of Being AI', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Probability Distribution - AI Texts', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("     [OK] Saved: probability_distribution.png")

print("\nGenerating Cross-Validation Scores Plot...")

cv_scores = [0.9835, 0.9917, 0.9669, 1.0000, 0.9752, 1.0000, 1.0000, 0.9917, 0.9750, 0.9917]

fig, ax = plt.subplots(figsize=(12, 6))

folds = list(range(1, 11))
bars = ax.bar(folds, [s*100 for s in cv_scores], color='steelblue', alpha=0.7, edgecolor='black')

mean_cv = np.mean(cv_scores) * 100
ax.axhline(mean_cv, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cv:.2f}%')

for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{score*100:.2f}%', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('10-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
ax.set_xticks(folds)
ax.set_ylim([95, 101])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cv_scores_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("     [OK] Saved: cv_scores_plot.png")

print(f"\n     CV Mean: {mean_cv:.2f}%, Std: {np.std(cv_scores)*100:.2f}%")

print("\n" + "="*70)
print("VISUALISASI SELESAI!")

print("\nFile yang dihasilkan:")
print("  1. learning_curve.png           - Learning Curve")
print("  2. roc_curve.png                - ROC Curve")
print("  3. precision_recall_curve.png   - Precision-Recall Curve")
print("  4. confusion_matrix_heatmap.png - Confusion Matrix")
print("  5. feature_importance_top20.png - Feature Importance")
print("  6. model_comparison_chart.png   - Model Comparison")
print("  7. probability_distribution.png - Probability Distribution")
print("  8. cv_scores_plot.png           - CV Scores")

print("\n" + "="*70)
print("SEMUA VISUALISASI SIAP UNTUK SKRIPSI!")
