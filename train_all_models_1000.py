# =====================================================
# TRAIN SEMUA MODEL DENGAN DATASET 1000 (500 vs 500)
# Random Forest, Logistic Regression, SVM
# =====================================================

import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from pathlib import Path

# =====================================================
# LANGKAH 1: PENGGABUNGAN DATA (MERGE)
# =====================================================
print("=" * 70)
print("LANGKAH 1: PENGGABUNGAN DATA (MERGE)")
print("=" * 70)

# Load data AI
print("\n[1/4] Load data AI...")
df_ai = pd.read_csv('dataset_ai_500.csv')
df_ai = df_ai[['text', 'label']]
print(f"  - AI data: {len(df_ai)}")

# Load data Manusia
print("[2/4] Load data Manusia...")
df_human = pd.read_csv('data_manusia_full_dev.csv')
# Ambil 500 data random
df_human = df_human.sample(n=500, random_state=42)[['text', 'label']]
print(f"  - Human data: {len(df_human)}")

# Gabungkan
print("[3/4] Gabungkan data...")
df_all = pd.concat([df_ai, df_human], ignore_index=True)
print(f"  - Total sebelum shuffle: {len(df_all)}")

# PENTING: Acak urutannya (Shuffle)
print("[4/4] Shuffle data...")
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  - Total setelah shuffle: {len(df_all)}")

# Print distribusi label
print(f"\nDistribusi Label (500 vs 500):")
print(df_all['label'].value_counts())

# Cek shuffle - print 10 data pertama
print(f"\n10 data pertama (cek shuffle):")
for idx, row in df_all.head(10).iterrows():
    print(f"  {idx+1}. {row['label']:8} - {row['text'][:50]}...")

# Simpan hasil gabungan
output_file = 'dataset_final_1000.csv'
df_all.to_csv(output_file, index=False)
print(f"\n[SAVE] {output_file} saved!")

# =====================================================
# LANGKAH 2: PELATIHAN ULANG (RETRAIN MODEL)
# =====================================================
print("\n" + "=" * 70)
print("LANGKAH 2: PELATIHAN ULANG (RETRAIN 4 MODEL)")
print("=" * 70)

# Preprocessing
print("\n[1/6] Preprocessing...")
X = df_all['text']
y = df_all['label'].map({'AI': 1, 'MANUSIA': 0})
print(f"  - Features: {X.shape}")
print(f"  - Target: {y.shape}")

# Split data 80/20
print("\n[2/6] Split data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  - Train: {len(X_train)}")
print(f"  - Test: {len(X_test)}")

# Vectorization (TF-IDF)
print("\n[3/6] Vectorization (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"  - Feature shape: {X_train_vec.shape}")

# Save vectorizer
os.makedirs('models_1000', exist_ok=True)
joblib.dump(vectorizer, 'models_1000/vectorizer.pkl')
print(f"  - Vectorizer saved!")

# =====================================================
# TRAIN 4 MODEL
# =====================================================
models_config = {
    'Random Forest (100 trees, max_depth=20)': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Random Forest (200 trees, max_depth=30)': RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    ),
    'SVM (RBF Kernel)': SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        C=1.0
    )
}

results = {}
model_idx = 0

for model_name, model in models_config.items():
    model_idx += 1
    print(f"\n[4/6] Training Model {model_idx}/4: {model_name}...")
    print(f"  - Fitting model...")

    model.fit(X_train_vec, y_train)
    print(f"  - Training completed!")

    # Predict
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    filename = model_name.split('(')[0].strip().lower().replace(' ', '_')
    joblib.dump(model, f'models_1000/{filename}.pkl')
    print(f"  - Model saved: {filename}.pkl")

    # Store results
    results[model_name] = {
        'model': model,
        'accuracy': acc,
        'y_pred': y_pred,
        'filename': filename
    }

    print(f"  - Accuracy: {acc*100:.2f}%")

# =====================================================
# HASIL PERBANDINGAN
# =====================================================
print("\n" + "=" * 70)
print("HASIL PERBANDINGAN SEMUA MODEL")
print("=" * 70)

print(f"\n{'Model':<40} {'Akurasi':<15} {'File'}")
print("-" * 70)

for model_name, result in results.items():
    print(f"{model_name:<40} {result['accuracy']*100:>6.2f}%        {result['filename']}.pkl")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_acc = results[best_model_name]['accuracy']
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_acc*100:.2f}%")

# =====================================================
# DETAIL REPORT UNTUK BEST MODEL
# =====================================================
print("\n" + "=" * 70)
print(f"CLASSIFICATION REPORT - {best_model_name}")
print("=" * 70)

best_result = results[best_model_name]
print(f"\nAccuracy: {best_result['accuracy']*100:.2f}%\n")
print(classification_report(y_test, best_result['y_pred'], target_names=['MANUSIA', 'AI']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_result['y_pred'])
print(f"                Prediksi MANUSIA    Prediksi AI")
print(f"Aktual MANUSIA      {cm[0,0]:>6}              {cm[0,1]:>6}")
print(f"Aktual AI            {cm[1,0]:>6}              {cm[1,1]:>6}")

print("\n" + "=" * 70)
print("SEMUA MODEL TERSIMPAN DI FOLDER: models_1000/")
print("=" * 70)
print("Files:")
print("  - vectorizer.pkl")
print("  - random_forest.pkl (100 trees)")
print("  - random_forest_200_trees.pkl (200 trees)")
print("  - logistic_regression.pkl")
print("  - svm_rbf_kernel.pkl")
print("=" * 70)
