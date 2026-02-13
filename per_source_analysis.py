"""
Analisis Detail Per Sumber Data Manusia
Melihat performa deteksi per kategori sumber (IndoSum, Reddit, Twitter, dll)
"""

import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ANALISIS DETAIL PER SUMBER DATA MANUSIA")
print("="*70)

# Load semua file manusia
print("\n[1] Load file data manusia...")

human_files = [
    "data_manusia_indosum_250.csv",
    "data_manusia_kaggle_marketplace.csv",
    "data_manusia_kaggle_reddit.csv",
    "data_manusia_kaggle_terrorism.csv",
    "data_manusia_kaggle_twitter.csv"
]

all_human_data = []

for f in human_files:
    try:
        df = pd.read_csv(f, encoding='utf-8')
        if 'source' in df.columns:
            source = df['source'].iloc[0] if len(df) > 0 else f
            print(f"  [OK] {f}: {len(df)} data (sumber: {source})")
            all_human_data.append(df)
        else:
            print(f"  [!] {f}: Tidak ada kolom source")
    except Exception as e:
        print(f"  [X] {f}: Error - {e}")

if not all_human_data:
    print("  Tidak ada data manusia yang bisa dimuat")
    exit(1)

# Gabungkan semua data manusia
print(f"\n[2] Menggabungkan {len(all_human_data)} file...")
human_df = pd.concat(all_human_data, ignore_index=True)
print(f"  Total data manusia: {len(human_df)}")

# Load data AI
print("\n[3] Load data AI...")
ai_df = pd.read_csv("data_ai_all_clean.csv", encoding='utf-8')
print(f"  Total data AI: {len(ai_df)}")

# Load model dan vectorizer
print("\n[4] Load model dan vectorizer...")
model = joblib.load('models/random_forest_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Gabungkan untuk prediksi
print("\n[5] Gabungkan data...")
human_df['true_label'] = 'MANUSIA'
ai_df['true_label'] = 'AI'

combined_df = pd.concat([
    human_df[['text', 'true_label', 'source']],
    ai_df[['text', 'true_label']].assign(source='AI')
], ignore_index=True)

combined_df = combined_df.dropna(subset=['text'])
print(f"  Total combined: {len(combined_df)}")

# TF-IDF transform
print("\n[6] Transform dan predict...")
X_tfidf = vectorizer.transform(combined_df['text'])

label_mapping = {'MANUSIA': 0, 'AI': 1}
reverse_label_mapping = {0: 'MANUSIA', 1: 'AI'}

y_true = combined_df['true_label'].map(label_mapping).values
y_pred = model.predict(X_tfidf)
y_proba = model.predict_proba(X_tfidf)

combined_df['predicted_label'] = [reverse_label_mapping[p] for p in y_pred]
combined_df['proba_manusia'] = y_proba[:, 0]
combined_df['proba_ai'] = y_proba[:, 1]
combined_df['is_correct'] = combined_df['true_label'] == combined_df['predicted_label']

# Analisis per sumber
print("\n" + "="*70)
print("ANALISIS PER SUMBER DATA")
print("="*70)

sources = combined_df['source'].unique()
print(f"\nDitemukan {len(sources)} sumber:")

results_summary = []

for source in sorted(sources):
    subset = combined_df[combined_df['source'] == source]
    total = len(subset)
    correct = subset['is_correct'].sum()
    wrong = total - correct
    accuracy = (correct / total * 100) if total > 0 else 0

    # Confidence rata-rata
    correct_subset = subset[subset['is_correct']]
    wrong_subset = subset[~subset['is_correct']]

    if len(correct_subset) > 0:
        if source == 'AI':
            avg_conf = correct_subset['proba_ai'].mean()
        else:
            avg_conf = correct_subset['proba_manusia'].mean()
    else:
        avg_conf = 0

    results_summary.append({
        'Source': source,
        'Total': total,
        'Correct': correct,
        'Wrong': wrong,
        'Accuracy': accuracy,
        'Avg Confidence': avg_conf * 100
    })

print("\n" + "-"*70)
print(f"{'Source':<25} {'Total':>6} {'Benar':>6} {'Salah':>6} {'Accuracy':>10} {'Conf':>8}")
print("-"*70)

for r in sorted(results_summary, key=lambda x: x['Accuracy'], reverse=True):
    print(f"{r['Source']:<25} {r['Total']:>6} {r['Correct']:>6} {r['Wrong']:>6} {r['Accuracy']:>9.2f}% {r['Avg Confidence']:>7.1f}%")

# Detail per sumber manusia
print("\n" + "="*70)
print("DETAIL ANALISIS SUMBER MANUSIA")
print("="*70)

human_sources = [s for s in sources if s != 'AI']

for source in sorted(human_sources):
    subset = combined_df[combined_df['source'] == source]
    correct = subset[subset['is_correct']]
    wrong = subset[~subset['is_correct']]

    print(f"\n{'='*70}")
    print(f"SUMBER: {source}")
    print(f"{'='*70}")
    print(f"Total Data: {len(subset)}")
    print(f"Benar: {len(correct)} ({len(correct)/len(subset)*100:.2f}%)")
    print(f"Salah: {len(wrong)} ({len(wrong)/len(subset)*100:.2f}%)")

    if len(correct) > 0:
        print(f"\nConfidence (yang benar):")
        print(f"  Mean: {correct['proba_manusia'].mean()*100:.2f}%")
        print(f"  Min: {correct['proba_manusia'].min()*100:.2f}%")
        print(f"  Max: {correct['proba_manusia'].max()*100:.2f}%")

    if len(wrong) > 0:
        print(f"\nContoh yang salah diprediksi:")
        for i, (idx, row) in enumerate(wrong.head(3).iterrows(), 1):
            print(f"\n  [{i}] Manusia -> AI")
            print(f"      Confidence AI: {row['proba_ai']*100:.1f}%")
            print(f"      Panjang: {len(row['text'])} karakter")
            print(f"      Text: {row['text'][:150]}...")

# Analisis per panjang teks
print("\n" + "="*70)
print("ANALISIS BERDASARKAN PANJANG TEKS")
print("="*70)

combined_df['text_length'] = combined_df['text'].str.len()

length_bins = [
    (0, 500, 'Sangat Pendek (<500)'),
    (500, 1000, 'Pendek (500-1000)'),
    (1000, 1500, 'Sedang (1000-1500)'),
    (1500, 2000, 'Panjang (1500-2000)'),
    (2000, 10000, 'Sangat Panjang (>2000)')
]

print("\n" + "-"*70)
print(f"{'Kategori':<20} {'Total':>6} {'Salah':>6} {'Accuracy':>10}")
print("-"*70)

for min_len, max_len, label in length_bins:
    subset = combined_df[
        (combined_df['text_length'] >= min_len) &
        (combined_df['text_length'] < max_len)
    ]
    if len(subset) > 0:
        wrong = (~subset['is_correct']).sum()
        accuracy = (subset['is_correct'].sum() / len(subset) * 100)
        print(f"{label:<20} {len(subset):>6} {wrong:>6} {accuracy:>9.2f}%")

# Distribusi panjang per sumber
print("\n" + "="*70)
print("STATISTIK PANJANG TEKS PER SUMBER")
print("="*70)

print(f"\n{'Source':<25} {'Mean':>8} {'Median':>8} {'Min':>6} {'Max':>6}")
print("-"*70)

for source in sorted(sources):
    subset = combined_df[combined_df['source'] == source]
    if len(subset) > 0:
        lengths = subset['text_length']
        print(f"{source:<25} {lengths.mean():>8.0f} {lengths.median():>8.0f} {lengths.min():>6.0f} {lengths.max():>6.0f}")

# Export hasil
print("\n" + "="*70)
print("EXPORT HASIL")
print("="*70)

combined_df.to_csv('per_source_analysis_results.csv', index=False, encoding='utf-8')
print("\n[OK] Export: per_source_analysis_results.csv")

# Export summary per source
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv('per_source_summary.csv', index=False, encoding='utf-8')
print("[OK] Export: per_source_summary.csv")

print("\n" + "="*70)
print("ANALISIS PER SUMBER SELESAI!")
print("="*70)

# Ringkasan
print("\n" + "="*70)
print("RINGKASAN UTAMA")
print("="*70)

best_source = max(results_summary, key=lambda x: x['Accuracy'])
worst_source = min(results_summary, key=lambda x: x['Accuracy'])

print(f"""
Sumber dengan Accuracy TERTINGGI:
  {best_source['Source']}: {best_source['Accuracy']:.2f}% ({best_source['Correct']}/{best_source['Total']})

Sumber dengan Accuracy TERENDAH:
  {worst_source['Source']}: {worst_source['Accuracy']:.2f}% ({worst_source['Correct']}/{worst_source['Total']})

Total Error: {(combined_df['is_correct'] == False).sum()} dari {len(combined_df)} data
Overall Accuracy: {(combined_df['is_correct'].sum() / len(combined_df) * 100):.2f}%
""")

print("="*70)
