"""
Analisis Per Kategori/Model AI
Melihat performa deteksi berdasarkan sumber teks AI
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ANALISIS PER KATEGORI/MODEL AI")
print("="*70)

# Load dataset utama
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()
print(f"     Total data: {len(df)}")

# Cek apakah ada kolom ai_model atau source
if 'ai_model' in df.columns or 'source' in df.columns:
    print("     [OK] Dataset memiliki kolom kategori")
else:
    print("     [!] Dataset tidak memiliki kolom ai_model/source")
    print("     Mencoba merge dengan file AI data asli...")

# Cari file AI data yang mungkin memiliki info model
import glob
import os

ai_files = glob.glob("dataset_ai_*.csv")
print(f"\n[2] Mencari file AI data...")
print(f"     Ditemukan {len(ai_files)} file AI:")

file_info = {}
for f in ai_files:
    try:
        temp_df = pd.read_csv(f, encoding='utf-8')
        print(f"     - {f}: {len(temp_df)} baris")
        if 'ai_model' in temp_df.columns:
            file_info[f] = temp_df
            print(f"       [OK] Memiliki kolom ai_model")
        elif 'model' in temp_df.columns:
            temp_df = temp_df.rename(columns={'model': 'ai_model'})
            file_info[f] = temp_df
            print(f"       [OK] Memiliki kolom model (di-rename)")
    except Exception as e:
        print(f"     - {f}: Error - {e}")

# Analisis distribusi data
print("\n" + "="*70)
print("DISTRIBUSI DATA")
print("="*70)

print("\nDistribusi Label:")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(df) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

# Load model untuk prediksi
print("\n" + "="*70)
print("LOAD MODEL UNTUK PREDIKSI")
print("="*70)

print("\n[3] Load model dan vectorizer...")
model = joblib.load('models/random_forest_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Preprocessing untuk prediksi
label_mapping = {'MANUSIA': 0, 'AI': 1}
reverse_label_mapping = {0: 'MANUSIA', 1: 'AI'}
df['label_num'] = df['label'].map(label_mapping)

X = df['text']
y = df['label_num']

# TF-IDF
X_tfidf = vectorizer.transform(X)

# Predict semua data
y_pred = model.predict(X_tfidf)
y_proba = model.predict_proba(X_tfidf)

df['predicted_label'] = y_pred
df['predicted_label_str'] = df['predicted_label'].map(reverse_label_mapping)
df['proba_ai'] = y_proba[:, 1]
df['proba_manusia'] = y_proba[:, 0]

# Analisis per label
print("\n" + "="*70)
print("ANALISIS PER LABEL")
print("="*70)

for label_val, label_name in [(0, 'MANUSIA'), (1, 'AI')]:
    subset = df[df['label_num'] == label_val]
    correct = subset[subset['predicted_label'] == label_val]
    wrong = subset[subset['predicted_label'] != label_val]

    accuracy = len(correct) / len(subset) * 100 if len(subset) > 0 else 0

    print(f"\n{label_name}:")
    print(f"  Total: {len(subset)}")
    print(f"  Benar: {len(correct)}")
    print(f"  Salah: {len(wrong)}")
    print(f"  Accuracy: {accuracy:.2f}%")

    if len(wrong) > 0:
        print(f"\n  Contoh yang salah prediksi:")
        for i, (idx, row) in enumerate(wrong.head(3).iterrows(), 1):
            print(f"\n  [{i}] {label_name} -> {row['predicted_label_str']}")
            print(f"      Confidence: {row['proba_ai' if row['predicted_label'] == 1 else 'proba_manusia']*100:.1f}%")
            print(f"      Text: {row['text'][:150]}...")

# Analisis kategori data manusia
print("\n" + "="*70)
print("ANALISIS SUMBER DATA MANUSIA")
print("="*70)

# Cek file manusia
human_files = glob.glob("dataset_manusia*.csv") + glob.glob("*manusia*.csv")

print(f"\n[4] Mencari file data manusia...")
for f in human_files:
    try:
        temp_df = pd.read_csv(f, encoding='utf-8')
        source_col = None
        for col in ['source', 'category', 'kategori', 'sumber', 'type']:
            if col in temp_df.columns:
                source_col = col
                break

        if source_col:
            print(f"\n  File: {f}")
            print(f"  Kolom sumber: {source_col}")
            print(f"  Kategori: {temp_df[source_col].unique().tolist()}")
    except:
        pass

# Analisis panjang teks per label
print("\n" + "="*70)
print("ANALISIS PANJANG TEKS PER LABEL")
print("="*70)

df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

for label_val, label_name in [(0, 'MANUSIA'), (1, 'AI')]:
    subset = df[df['label_num'] == label_val]
    print(f"\n{label_name}:")
    print(f"  Rata-rata panjang: {subset['text_length'].mean():.1f} karakter")
    print(f"  Median: {subset['text_length'].median():.1f}")
    print(f"  Min: {subset['text_length'].min()}, Max: {subset['text_length'].max()}")
    print(f"  Rata-rata kata: {subset['word_count'].mean():.1f}")

# Analisis confidence per label
print("\n" + "="*70)
print("ANALISIS CONFIDENCE PREDIKSI PER LABEL")
print("="*70)

for label_val, label_name in [(0, 'MANUSIA'), (1, 'AI')]:
    subset = df[df['label_num'] == label_val]
    print(f"\n{label_name}:")
    print(f"  Rata-rata confidence (prediksi benar): ", end="")
    correct = subset[subset['predicted_label'] == label_val]
    if len(correct) > 0:
        conf_col = 'proba_manusia' if label_val == 0 else 'proba_ai'
        print(f"{correct[conf_col].mean()*100:.2f}%")

    print(f"  Rata-rata confidence (prediksi salah): ", end="")
    wrong = subset[subset['predicted_label'] != label_val]
    if len(wrong) > 0:
        conf_col = 'proba_ai' if wrong.iloc[0]['predicted_label'] == 1 else 'proba_manusia'
        print(f"{wrong[conf_col].mean()*100:.2f}%")

# Statistik confidence
print("\n" + "-"*70)
print("Distribusi Confidence:")
print("-"*70)

print("\nManusia yang benar diprediksi:")
human_correct = df[(df['label_num'] == 0) & (df['predicted_label'] == 0)]
print(f"  Mean confidence: {human_correct['proba_manusia'].mean()*100:.2f}%")
print(f"  Min confidence: {human_correct['proba_manusia'].min()*100:.2f}%")

print("\nAI yang benar diprediksi:")
ai_correct = df[(df['label_num'] == 1) & (df['predicted_label'] == 1)]
print(f"  Mean confidence: {ai_correct['proba_ai'].mean()*100:.2f}%")
print(f"  Min confidence: {ai_correct['proba_ai'].min()*100:.2f}%")

# Export hasil
print("\n" + "="*70)
print("EXPORT HASIL ANALISIS")
print("="*70)

# Export dengan prediksi
export_cols = ['text', 'label', 'predicted_label_str', 'proba_ai', 'proba_manusia', 'text_length', 'word_count']
df_export = df[export_cols].copy()
df_export.columns = ['text', 'true_label', 'predicted_label', 'proba_ai', 'proba_manusia', 'text_length', 'word_count']
df_export['is_correct'] = df_export['true_label'] == df_export['predicted_label']

df_export.to_csv('analysis_with_predictions.csv', index=False, encoding='utf-8')
print("\n[OK] Export: analysis_with_predictions.csv")

# Export error
errors = df_export[df_export['is_correct'] == False]
errors.to_csv('analysis_errors.csv', index=False, encoding='utf-8')
print(f"[OK] Export: analysis_errors.csv ({len(errors)} errors)")

print("\n" + "="*70)
print("ANALISIS PER KATEGORI SELESAI!")
print("="*70)

# Summary
print("\n" + "="*70)
print("RINGKASAN")
print("="*70)

total_correct = len(df[df['predicted_label'] == df['label_num']])
total_accuracy = total_correct / len(df) * 100

print(f"""
Total Data: {len(df)}
Total Benar: {total_correct}
Total Salah: {len(df) - total_correct}
Overall Accuracy: {total_accuracy:.2f}%

Per Label:
- MANUSIA: {len(df[df['label_num'] == 0])} data
- AI: {len(df[df['label_num'] == 1])} data

Confidence Rata-rata:
- Manusia (benar): {human_correct['proba_manusia'].mean()*100:.2f}%
- AI (benar): {ai_correct['proba_ai'].mean()*100:.2f}%
""")

print("="*70)
