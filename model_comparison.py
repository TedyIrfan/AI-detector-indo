# PERBANDINGAN MODEL MACHINE LEARNING
# Random Forest vs Logistic Regression vs SVM

# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import joblib
from pathlib import Path
import time

print("PERBANDINGAN MODEL: Random Forest vs Logistic Regression vs SVM")

# Load dataset dulu
print("\nMemuat dataset...")
input_file = "dataset_skripsi_final.csv"

try:
    df = pd.read_csv(input_file)
    print(f"[OK] Dataset dimuat: {len(df)} baris")
    print(f"     Distribusi: MANUSIA={sum(df['label']=='MANUSIA')}, AI={sum(df['label']=='AI')}")
except FileNotFoundError:
    print(f"[ERROR] File {input_file} tidak ditemukan!")
    exit(1)

# Preprocessing data
print("\nPreprocessing data...")

# Mapping label biar jadi angka
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_encoded'] = df['label'].map(label_mapping)

X = df['teks'].values
y = df['label_encoded'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Data di-split: Train={len(X_train)}, Test={len(X_test)}")

# Vectorization dengan TF-IDF
print("\nVectorization dengan TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"[OK] Vectorization selesai: {len(vectorizer.vocabulary_)} fitur")

# Siapin 3 model
print("\nMenyiapkan 3 model...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0
    ),
    'SVM (RBF Kernel)': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    )
}

for name in models.keys():
    print(f"  - {name}")

# Training semua model
print("\nTraining semua model...")
print("-"*70)

results = {}
training_times = {}

for model_name, model in models.items():
    print(f"\n[TRAINING] {model_name}...")

    # Hitung waktu training
    start_time = time.time()

    model.fit(X_train_tfidf, y_train)

    training_time = time.time() - start_time
    training_times[model_name] = training_time

    # Prediksi
    y_pred = model.predict(X_test_tfidf)

    # Ambil probabilitas kalau ada
    y_proba = model.predict_proba(X_test_tfidf) if hasattr(model, 'predict_proba') else None

    # Hitung metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Simpen hasil
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'training_time': training_time,
        'predictions': y_pred,
        'probabilities': y_proba
    }

    print(f"   [OK] Selesai! ({training_time:.2f} detik)")
    print(f"   Akurasi: {accuracy*100:.2f}%")

# Tampilkan hasil perbandingan
print("\n" + "="*70)
print("HASIL PERBANDINGAN MODEL")

comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'Akurasi (%)': f"{result['accuracy']*100:.2f}",
        'Precision': f"{result['precision']:.4f}",
        'Recall': f"{result['recall']:.4f}",
        'F1-Score': f"{result['f1_score']:.4f}",
        'Training Time (s)': f"{result['training_time']:.2f}"
    })

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# Pilih model terbaik
best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print(f"\n[BEST MODEL] {best_model_name}")
print(f"   Akurasi: {best_accuracy*100:.2f}%")

# Tampilkan confusion matrix detail
print("\n" + "="*70)
print("CONFUSION MATRIX DETAIL")

for model_name, result in results.items():
    cm = result['confusion_matrix']
    print(f"\n{model_name}:")
    print(f"{'':15}{'Prediksi':>20}")
    print(f"{'':15}{'MANUSIA':>10}{'AI':>10}")
    print(f"{'Aktual MANUSIA':>15}{cm[0][0]:>10}{cm[0][1]:>10}")
    print(f"{'Aktual AI':>15}{cm[1][0]:>10}{cm[1][1]:>10}")

# Tampilkan classification report detail
print("\n" + "="*70)
print("CLASSIFICATION REPORT (DETAIL)")

for model_name, result in results.items():
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print('='*70)
    print(classification_report(
        y_test, result['predictions'],
        target_names=['MANUSIA', 'AI'],
        digits=4
    ))

# Simpen semua model
print("\n" + "="*70)
print("MENYIMPAN SEMUA MODEL")

models_dir = Path("models_comparison")
models_dir.mkdir(exist_ok=True)

# Simpen vectorizer
vectorizer_file = models_dir / "vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_file)
print(f"[OK] Vectorizer disimpan: {vectorizer_file}")

# Simpen tiap model
for model_name, result in results.items():
    safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    model_file = models_dir / f"{safe_name}.pkl"

    joblib.dump(result['model'], model_file)
    print(f"[OK] {model_name} disimpan: {model_file}")

# Simpen hasil perbandingan
df_comparison.to_csv(models_dir / "comparison_results.csv", index=False)
print(f"[OK] Hasil perbandingan disimpan: {models_dir / 'comparison_results.csv'}")

# Simpen label mapping
mapping_file = models_dir / "label_mapping.txt"
with open(mapping_file, 'w') as f:
    f.write("Label Mapping:\n")
    f.write("MANUSIA -> 0\n")
    f.write("AI -> 1\n")
print(f"[OK] Label mapping disimpan: {mapping_file}")

# Tampilkan ringkasan
print("\n" + "="*70)
print("SUMMARY")

print(f"\n{'Model':<25} {'Akurasi':<12} {'F1-Score':<12} {'Time (s)':<12}")
print("-"*70)

sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

for model_name, result in sorted_results:
    print(f"{model_name:<25} {result['accuracy']*100:>10.2f}%  "
          f"{result['f1_score']:>10.4f}  {result['training_time']:>10.2f}")

print("\n" + "="*70)
print(f"REKOMENDASI: Gunakan {best_model_name} untuk produksi")
print(f"Alasan: Akurasi tertinggi ({best_accuracy*100:.2f}%)")
