"""
Perbandingan Model untuk Bab Skripsi
Membandingkan berbagai algoritma ML untuk deteksi AI vs Manusia
"""

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PERBANDINGAN MODEL - BAB SKRIPSI")
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

# Split
print("\n[3] Split data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Training: {len(X_train)}, Testing: {len(X_test)}")

# TF-IDF
print("\n[4] TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"     Features: {X_train_tfidf.shape[1]}")

# Definisi model
print("\n" + "="*70)
print("TRAINING BERBAGAI MODEL")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM Linear': SVC(kernel='linear', probability=True, random_state=42),
    'SVM RBF': SVC(kernel='rbf', probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'XGBoost': None,  # Akan diinisialisasi di loop
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

# Coba import XGBoost
try:
    from xgboost import XGBClassifier
    models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
    use_xgb = True
except ImportError:
    print("     [!] XGBoost tidak tersedia, di-skip")
    models.pop('XGBoost')
    use_xgb = False

results = []
detailed_results = []

for name, model in models.items():
    print(f"\n[{len(results)+1}/{len(models)}] Training {name}...")

    if model is None:
        continue

    try:
        # Training
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_time

        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test_tfidf)
        pred_time = time.time() - start_time

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Sensitivity & Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall AI
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall Human

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        detailed_results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Sensitivity (AI)': sensitivity,
            'Specificity (Human)': specificity,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Train Time (s)': train_time,
            'Predict Time (s)': pred_time
        })

        print(f"     Accuracy: {accuracy*100:.2f}%")
        print(f"     F1-Score: {f1*100:.2f}%")
        print(f"     Train Time: {train_time:.2f}s")

    except Exception as e:
        print(f"     [X] Error: {e}")
        continue

# Hasil perbandingan
print("\n" + "="*70)
print("HASIL PERBANDINGAN MODEL")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\nPeringkat Model berdasarkan Accuracy:")
print("-"*70)
for i, row in results_df.iterrows():
    print(f"{row['Accuracy']*100:5.2f}%  -  {row['Model']}")

# Tabel lengkap
print("\n" + "-"*70)
print("TABEL LENGKAP PERBANDINGAN MODEL")
print("-"*70)

comparison_df = pd.DataFrame(detailed_results)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

# Format untuk display
display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
                'Sensitivity (AI)', 'Specificity (Human)']
display_df = comparison_df[display_cols].copy()

for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sensitivity (AI)', 'Specificity (Human)']:
    display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

print(display_df.to_string(index=False))

# Visualisasi comparison
print("\n" + "="*70)
print("STATISTIK PERBANDINGAN")
print("="*70)

print(f"\nTotal Model: {len(results_df)}")
print(f"Accuracy Tertinggi: {results_df['Accuracy'].max()*100:.2f}% ({results_df.iloc[0]['Model']})")
print(f"Accuracy Terendah: {results_df['Accuracy'].min()*100:.2f}% ({results_df.iloc[-1]['Model']})")
print(f"Rata-rata Accuracy: {results_df['Accuracy'].mean()*100:.2f}%")
print(f"Std Deviation: {results_df['Accuracy'].std()*100:.2f}%")

# Top 3 models
print("\n" + "-"*70)
print("TOP 3 MODEL TERBAIK")
print("-"*70)

top3 = comparison_df.head(3)
for i, (idx, row) in enumerate(top3.iterrows(), 1):
    print(f"\n#{i} {row['Model']}")
    print(f"   Accuracy:    {row['Accuracy']*100:.2f}%")
    print(f"   F1-Score:    {row['F1-Score']*100:.2f}%")
    print(f"   Sensitivity: {row['Sensitivity (AI)']*100:.2f}% (AI Detection)")
    print(f"   Specificity: {row['Specificity (Human)']*100:.2f}% (Human Detection)")
    print(f"   TP: {int(row['TP'])}, TN: {int(row['TN'])}, FP: {int(row['FP'])}, FN: {int(row['FN'])}")

# Perbandingan training time
print("\n" + "-"*70)
print("PERBANDINGAN WAKTU TRAINING & PREDICTION")
print("-"*70)

time_df = comparison_df[['Model', 'Train Time (s)', 'Predict Time (s)']].copy()
time_df = time_df.sort_values('Train Time (s)')
print(time_df.to_string(index=False))

# Export results
print("\n" + "="*70)
print("EXPORT HASIL PERBANDINGAN")
print("="*70)

# Full results
comparison_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8')
print("     [OK] Full results: model_comparison_results.csv")

# Summary for thesis
summary_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
summary_df = summary_df.sort_values('Accuracy', ascending=False)
summary_df.to_csv('model_comparison_summary.csv', index=False, encoding='utf-8')
print("     [OK] Summary: model_comparison_summary.csv")

# Rekomendasi
print("\n" + "="*70)
print("REKOMENDASI MODEL")
print("="*70)

best_model = comparison_df.iloc[0]
print(f"\nModel Terbaik: {best_model['Model']}")
print(f"  - Accuracy: {best_model['Accuracy']*100:.2f}%")
print(f"  - F1-Score: {best_model['F1-Score']*100:.2f}%")
print(f"  - Sensitivity (AI Detection): {best_model['Sensitivity (AI)']*100:.2f}%")
print(f"  - Waktu Training: {best_model['Train Time (s)']:.2f} detik")

if best_model['Sensitivity (AI)'] >= 0.99:
    print("\n  [+] SANGAT BAIK untuk mendeteksi AI (>99%)")
if best_model['Specificity (Human)'] >= 0.95:
    print("  [+] SANGAT BAIK untuk mendeteksi Manusia (>95%)")
if best_model['Train Time (s)'] < 5:
    print("  [+] Cepat dalam training")

print("\n" + "="*70)
print("PERBANDINGAN MODEL SELESAI!")
print("="*70)

# Print untuk skripsi
print("\n" + "="*70)
print("FORMAT UNTUK SKRIPSI")
print("="*70)

print("\nTabel Perbandingan Model:")
print("-"*70)
print("| No | Model | Accuracy | Precision | Recall | F1-Score |")
print("|----|-------|----------|-----------|--------|----------|")

for i, (idx, row) in enumerate(comparison_df.iterrows(), 1):
    print(f"| {i} | {row['Model']:<20} | {row['Accuracy']*100:>6.2f}% | {row['Precision']*100:>9.2f}% | {row['Recall']*100:>6.2f}% | {row['F1-Score']*100:>8.2f}% |")

print("-"*70)
