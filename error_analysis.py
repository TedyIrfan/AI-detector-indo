"""
Analisis Error - Identifikasi teks yang salah prediksi
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ANALISIS ERROR - RANDOM FOREST MODEL")
print("="*60)

# Load dataset
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()
print(f"     Total data: {len(df)}")

# Preprocessing
print("\n[2] Preprocessing...")
label_mapping = {'MANUSIA': 0, 'AI': 1}
reverse_label_mapping = {0: 'MANUSIA', 1: 'AI'}
df['label_num'] = df['label'].map(label_mapping)

X = df['text'].tolist()
y = df['label_num'].tolist()

# Split (sama seperti saat training)
print("\n[3] Split data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Training: {len(X_train)}, Testing: {len(X_test)}")

# Load model dan vectorizer
print("\n[4] Load model dan vectorizer...")
model = joblib.load('models/random_forest_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
print("     [OK] Model loaded!")

# Transform dan predict
print("\n[5] Predict test data...")
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)

# Hitung akurasi
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"     Test Accuracy: {accuracy*100:.2f}%")

# Identifikasi error
print("\n" + "="*60)
print("ANALISIS TEKS YANG SALAH PREDIKSI")
print("="*60)

# Buat DataFrame hasil
results_df = pd.DataFrame({
    'text': X_test,
    'true_label': y_test,
    'predicted_label': y_pred,
    'proba_manusia': y_proba[:, 0],
    'proba_ai': y_proba[:, 1]
})

# Konversi ke label string
results_df['true_label_str'] = results_df['true_label'].map(reverse_label_mapping)
results_df['predicted_label_str'] = results_df['predicted_label'].map(reverse_label_mapping)

# Filter error
errors = results_df[results_df['true_label'] != results_df['predicted_label']].copy()
print(f"\nTotal Error: {len(errors)} dari {len(results_df)} data")
print(f"Error Rate: {len(errors)/len(results_df)*100:.2f}%")

# Analisis tipe error
print("\n" + "-"*60)
print("TIPE ERROR:")
print("-"*60)

# False Positive (Manusia diprediksi AI)
false_positives = errors[errors['true_label'] == 0]
print(f"\nFalse Positive (Manusia -> AI): {len(false_positives)}")

# False Negative (AI diprediksi Manusia)
false_negatives = errors[errors['true_label'] == 1]
print(f"False Negative (AI -> Manusia): {len(false_negatives)}")

# Tampilkan contoh error
print("\n" + "="*60)
print("CONTEH TEKS YANG SALAH PREDIKSI")
print("="*60)

# False Positives (Manusia yang kedeteksi AI)
if len(false_positives) > 0:
    print("\n" + "-"*60)
    print("FALSE POSITIVES (MANUSIA -> AI)")
    print("-"*60)
    for i, (idx, row) in enumerate(false_positives.head(5).iterrows(), 1):
        print(f"\n[{i}] Manusia terdeteksi sebagai AI")
        print(f"Confidence: {row['proba_ai']*100:.1f}%")
        print(f"Text: {row['text'][:300]}...")
        if len(row['text']) > 300:
            print(f"      (total: {len(row['text'])} karakter)")

# False Negatives (AI yang kedeteksi Manusia)
if len(false_negatives) > 0:
    print("\n" + "-"*60)
    print("FALSE NEGATIVES (AI -> MANUSIA)")
    print("-"*60)
    for i, (idx, row) in enumerate(false_negatives.head(5).iterrows(), 1):
        print(f"\n[{i}] AI terdeteksi sebagai Manusia")
        print(f"Confidence: {row['proba_manusia']*100:.1f}%")
        print(f"Text: {row['text'][:300]}...")
        if len(row['text']) > 300:
            print(f"      (total: {len(row['text'])} karakter)")

# Analisis panjang teks error
print("\n" + "="*60)
print("ANALISIS PANJANG TEKS ERROR")
print("="*60)

results_df['text_length'] = results_df['text'].str.len()
results_df['is_error'] = results_df['true_label'] != results_df['predicted_label']

print("\nStatistik Panjang Teks:")
print(f"  Rata-rata (semua): {results_df['text_length'].mean():.1f} karakter")
print(f"  Rata-rata (error): {results_df[results_df['is_error']]['text_length'].mean():.1f} karakter")
print(f"  Rata-rata (benar): {results_df[~results_df['is_error']]['text_length'].mean():.1f} karakter")

# Analisis confidence error
print("\n" + "="*60)
print("ANALISIS CONFIDENCE ERROR")
print("="*60)

errors['confidence'] = errors[['proba_manusia', 'proba_ai']].max(axis=1)
print(f"\nConfidence rata-rata pada error: {errors['confidence'].mean()*100:.1f}%")
print(f"Confidence tertinggi pada error: {errors['confidence'].max()*100:.1f}%")
print(f"Confidence terendah pada error: {errors['confidence'].min()*100:.1f}%")

# Error dengan confidence tinggi (model sangat yakin tapi salah)
high_conf_errors = errors[errors['confidence'] > 0.8]
print(f"\nError dengan confidence > 80%: {len(high_conf_errors)}")

if len(high_conf_errors) > 0:
    print("\nContoh error dengan confidence tinggi (model sangat yakin tapi salah):")
    for i, (idx, row) in enumerate(high_conf_errors.head(3).iterrows(), 1):
        print(f"\n[{i}] {row['true_label_str']} -> {row['predicted_label_str']}")
        print(f"    Confidence: {row['confidence']*100:.1f}%")
        print(f"    Text: {row['text'][:200]}...")

# Export error data
print("\n" + "="*60)
print("EXPORT ERROR DATA")
print("="*60)

errors_export = errors[['text', 'true_label_str', 'predicted_label_str',
                        'proba_manusia', 'proba_ai']].copy()
errors_export.columns = ['text', 'true_label', 'predicted_label',
                        'proba_manusia', 'proba_ai']
errors_export.to_csv('error_analysis_results.csv', index=False, encoding='utf-8')
print("\nError data exported to: error_analysis_results.csv")

print("\n" + "="*60)
print("ANALISIS ERROR SELESAI!")
print("="*60)
