"""
Ablation Study - Analisis Importance Fitur
Melihat pengaruh penghapusan fitur terhadap performa model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ABLATION STUDY - FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Load dataset
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()

label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text']
y = df['label_num']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"     Training: {len(X_train)}, Testing: {len(X_test)}")

# Load baseline model
print("\n[2] Load baseline model...")
baseline_model = joblib.load('models/random_forest_model.pkl')
baseline_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Baseline accuracy
X_test_tfidf = baseline_vectorizer.transform(X_test)
baseline_pred = baseline_model.predict(X_test_tfidf)
baseline_acc = accuracy_score(y_test, baseline_pred)

print(f"     Baseline Accuracy: {baseline_acc*100:.2f}%")

# Get feature names and importance
print("\n[3] Analisis Feature Importance...")
feature_names = baseline_vectorizer.get_feature_names_out()
feature_importance = baseline_model.feature_importances_

# Top 20 important features
top_indices = np.argsort(feature_importance)[-20:][::-1]

print("\n     Top 20 Most Important Features (Words):")
print("-"*70)
print(f"{'Rank':<5} {'Feature':<20} {'Importance':<12} {'Cumulative':<12}")
print("-"*70)

cumulative = 0
for rank, idx in enumerate(top_indices, 1):
    imp = feature_importance[idx]
    feat = feature_names[idx]
    cumulative += imp
    print(f"{rank:<5} {feat:<20} {imp*100:>11.4f}% {cumulative*100:>11.4f}%")

# Ablation Study Configurations
print("\n" + "="*70)
print("ABLATION STUDY - PENGHAPUSAN FITUR")
print("="*70)

ablation_results = []

# Configuration 1: Reduce max_features
print("\n[1] Mengurangi jumlah fitur (max_features)...")
for max_feat in [1000, 2000, 3000, 4000, 5000]:
    vectorizer_temp = TfidfVectorizer(
        max_features=max_feat,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    X_train_temp = vectorizer_temp.fit_transform(X_train)
    X_test_temp = vectorizer_temp.transform(X_test)

    model_temp = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model_temp.fit(X_train_temp, y_train)
    train_time = time.time() - start_time

    y_pred_temp = model_temp.predict(X_test_temp)
    acc_temp = accuracy_score(y_test, y_pred_temp)

    diff = baseline_acc - acc_temp

    ablation_results.append({
        'Configuration': f'max_features={max_feat}',
        'Description': f'{max_feat} fitur',
        'Accuracy': acc_temp,
        'Difference': diff,
        'Features': max_feat,
        'Train Time': train_time
    })

    print(f"     max_features={max_feat}: {acc_temp*100:.2f}% (diff: {diff*100:+.2f}%)")

# Configuration 2: Remove n-grams (use unigram only)
print("\n[2] Menghapus bigram (unigram only)...")
vectorizer_uni = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 1)  # Unigram only
)

X_train_uni = vectorizer_uni.fit_transform(X_train)
X_test_uni = vectorizer_uni.transform(X_test)

model_uni = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model_uni.fit(X_train_uni, y_train)
y_pred_uni = model_uni.predict(X_test_uni)
acc_uni = accuracy_score(y_test, y_pred_uni)
diff_uni = baseline_acc - acc_uni

ablation_results.append({
    'Configuration': 'unigram_only',
    'Description': 'Hanya unigram (tanpa bigram)',
    'Accuracy': acc_uni,
    'Difference': diff_uni,
    'Features': X_train_uni.shape[1],
    'Train Time': 0
})

print(f"     Unigram only: {acc_uni*100:.2f}% (diff: {diff_uni*100:+.2f}%)")

# Configuration 3: Different min_df
print("\n[3] Mengubah min_df (minimum document frequency)...")
for min_df in [1, 2, 3, 5]:
    vectorizer_temp = TfidfVectorizer(
        max_features=5000,
        min_df=min_df,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    X_train_temp = vectorizer_temp.fit_transform(X_train)
    X_test_temp = vectorizer_temp.transform(X_test)

    if X_train_temp.shape[1] == 0:
        print(f"     min_df={min_df}: Tidak ada fitur (skip)")
        continue

    model_temp = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_temp.fit(X_train_temp, y_train)
    y_pred_temp = model_temp.predict(X_test_temp)
    acc_temp = accuracy_score(y_test, y_pred_temp)
    diff = baseline_acc - acc_temp

    ablation_results.append({
        'Configuration': f'min_df={min_df}',
        'Description': f'Dokumen minimal: {min_df}',
        'Accuracy': acc_temp,
        'Difference': diff,
        'Features': X_train_temp.shape[1],
        'Train Time': 0
    })

    print(f"     min_df={min_df}: {acc_temp*100:.2f}% (diff: {diff*100:+.2f}%)")

# Configuration 4: Different max_df
print("\n[4] Mengubah max_df (maximum document frequency)...")
for max_df in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    vectorizer_temp = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=max_df,
        ngram_range=(1, 2)
    )

    X_train_temp = vectorizer_temp.fit_transform(X_train)
    X_test_temp = vectorizer_temp.transform(X_test)

    model_temp = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_temp.fit(X_train_temp, y_train)
    y_pred_temp = model_temp.predict(X_test_temp)
    acc_temp = accuracy_score(y_test, y_pred_temp)
    diff = baseline_acc - acc_temp

    ablation_results.append({
        'Configuration': f'max_df={max_df}',
        'Description': f'Max document freq: {max_df}',
        'Accuracy': acc_temp,
        'Difference': diff,
        'Features': X_train_temp.shape[1],
        'Train Time': 0
    })

    print(f"     max_df={max_df}: {acc_temp*100:.2f}% (diff: {diff*100:+.2f}%)")

# Configuration 5: Remove top N important features
print("\n[5] Menghapus top N fitur penting...")

# Get original full TF-IDF
vectorizer_full = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_full = vectorizer_full.fit_transform(X_train)
feature_names_full = vectorizer_full.get_feature_names_out()

# Train model to get importance
model_full = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model_full.fit(X_train_full, y_train)
importance_full = model_full.feature_importances_

# Try removing top 10, 20, 50 important features
for remove_top in [10, 20, 50, 100]:
    # Get indices of features to keep
    top_indices = np.argsort(importance_full)[-remove_top:]
    keep_indices = [i for i in range(len(feature_names_full)) if i not in top_indices]

    # Create new vectorizer with reduced features
    X_train_reduced = X_train_full[:, keep_indices]
    X_test_reduced = vectorizer_full.transform(X_test)[:, keep_indices]

    model_reduced = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_reduced.fit(X_train_reduced, y_train)
    y_pred_reduced = model_reduced.predict(X_test_reduced)
    acc_reduced = accuracy_score(y_test, y_pred_reduced)
    diff = baseline_acc - acc_reduced

    ablation_results.append({
        'Configuration': f'remove_top_{remove_top}',
        'Description': f'Hapus top {remove_top} fitur penting',
        'Accuracy': acc_reduced,
        'Difference': diff,
        'Features': X_train_reduced.shape[1],
        'Train Time': 0
    })

    print(f"     Remove top {remove_top}: {acc_reduced*100:.2f}% (diff: {diff*100:+.2f}%)")

# Results Summary
print("\n" + "="*70)
print("RINGKASAN HASIL ABLATION STUDY")
print("="*70)

ablation_df = pd.DataFrame(ablation_results)
ablation_df = ablation_df.sort_values('Accuracy', ascending=False)

print("\n" + "-"*70)
print(f"{'Configuration':<20} {'Accuracy':<12} {'Difference':<12} {'Features':<10}")
print("-"*70)

for _, row in ablation_df.iterrows():
    print(f"{row['Configuration']:<20} {row['Accuracy']*100:>11.2f}% {row['Difference']*100:>+11.2f}% {row['Features']:>10}")

print("-"*70)
print(f"{'BASELINE':<20} {baseline_acc*100:>11.2f}% {'-':>11} {5000:>10}")
print("-"*70)

# Analysis
print("\n" + "="*70)
print("ANALISIS")
print("="*70)

print("\n1. Konfigurasi dengan akurasi TERTINGGI:")
best = ablation_df.iloc[0]
print(f"   {best['Configuration']}: {best['Accuracy']*100:.2f}%")
print(f"   {best['Description']}")

print("\n2. Konfigurasi dengan penurunan TERBESAR:")
worst = ablation_df.loc[ablation_df['Difference'].idxmax()]
print(f"   {worst['Configuration']}: {worst['Accuracy']*100:.2f}% (turun {worst['Difference']*100:.2f}%)")
print(f"   {worst['Description']}")

print("\n3. Sensitivitas fitur:")
max_diff = ablation_df['Difference'].max()
if max_diff < 0.01:
    print(f"   Model SANGAT STABIL - perubahan fitur minim pengaruh")
elif max_diff < 0.03:
    print(f"   Model STABIL - perubahan fitur pengaruh kecil")
elif max_diff < 0.05:
    print(f"   Model CUKUP STABIL - perubahan fitur pengaruh sedang")
else:
    print(f"   Model KURANG STABIL - perubahan fitur pengaruh besar")

# Feature importance analysis
print("\n" + "="*70)
print("ANALISIS FITUR PENTING")
print("="*70)

# Word categories
ai_keywords = ['implementasi', 'penting', 'perlu', 'diharapkan', 'upaya',
               'program', 'melalui', 'dalam', 'untuk', 'comprehensif',
               'mengimplementasikan', 'berbagai', 'berdasarkan', 'mengenai',
               'telah', 'dapat', 'karena', 'dengan', 'yang', 'dan']

formal_words = ['telah', 'tersebut', 'melalui', 'dalam', 'untuk', 'bagai',
                'apabila', 'yakni', 'ialah', 'merupakan', 'diharapkan',
                'implementasi', 'pentingnya', 'perlunya']

informal_words = ['gue', 'elo', 'lu', 'gw', 'gak', 'nggak', 'ga', 'udah',
                  'nih', 'deh', 'dong', 'yuk', 'ayo', 'sih', 'loh', 'kok',
                  'kenapa', 'gimana', 'sampe', 'dah']

# Check if these words are in top features
print("\nApakah kata-kata kunci ada di top 20 fitur?")
print("-"*70)

for category, words in [
    ('Formal', formal_words),
    ('Informal', informal_words),
    ('AI-like', ai_keywords)
]:
    found = []
    for word in words:
        if word in feature_names:
            idx = list(feature_names).index(word)
            imp = feature_importance[idx]
            if imp > 0:
                found.append((word, imp))

    if found:
        found.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{category} words in features:")
        for word, imp in found[:5]:
            print(f"  - {word}: {imp*100:.4f}%")

# Export
print("\n" + "="*70)
print("EXPORT HASIL")
print("="*70)

ablation_df.to_csv('ablation_study_results.csv', index=False, encoding='utf-8')
print("\n[OK] Export: ablation_study_results.csv")

# Export feature importance
fi_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
})
fi_df = fi_df.sort_values('importance', ascending=False)
fi_df.to_csv('feature_importance.csv', index=False, encoding='utf-8')
print("[OK] Export: feature_importance.csv")

print("\n" + "="*70)
print("ABLATION STUDY SELESAI!")
print("="*70)
