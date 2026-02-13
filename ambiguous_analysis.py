# Import library yang diperlukan
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("ANALISIS TEKS AMBIGU (PROBABILITAS 40-60%)")

# Load dataset dulu
print("\nLoad dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()

# Mapping label
label_mapping = {'MANUSIA': 0, 'AI': 1}
reverse_label_mapping = {0: 'MANUSIA', 1: 'AI'}
df['label_num'] = df['label'].map(label_mapping)

# Siapin data
X = df['text'].tolist()
y = df['label_num'].tolist()
original_labels = df['label'].tolist()

print(f"     Total data: {len(df)}")

# Load model dan vectorizer
print("\nLoad model dan vectorizer...")
model = joblib.load('models/random_forest_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Transform dan prediksi
print("\nTransform dan predict...")
X_tfidf = vectorizer.transform(X)

# Prediksi dan ambil probabilitas
y_pred = model.predict(X_tfidf)
y_proba = model.predict_proba(X_tfidf)

# Ambil probabilitas AI
print("\n" + "="*70)
print("DISTRIBUSI PROBABILITAS PREDIKSI")

proba_ai = y_proba[:, 1]

# Tampilkan statistik probabilitas
print(f"\nStatistik Probabilitas AI:")
print(f"  Mean: {proba_ai.mean()*100:.2f}%")
print(f"  Median: {np.median(proba_ai)*100:.2f}%")
print(f"  Std Dev: {proba_ai.std()*100:.2f}%")
print(f"  Min: {proba_ai.min()*100:.2f}%")
print(f"  Max: {proba_ai.max()*100:.2f}%")

# Distribusi probabilitas dalam bins
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

print(f"\nDistribusi Probabilitas:")
hist, _ = np.histogram(proba_ai, bins=bins)
for i, (count, label) in enumerate(zip(hist, bin_labels)):
    pct = count / len(proba_ai) * 100
    bar = '#' * int(pct / 2)
    print(f"  {label:<10} {count:>4} ({pct:>5.1f}%) {bar}")

# Analisis teks ambigu
print("\n" + "="*70)
print("ANALISIS TEKS AMBIGU")

# Definisikan threshold buat teks ambigu
ambiguous_lower = 0.45
ambiguous_upper = 0.55

very_ambiguous_lower = 0.48
very_ambiguous_upper = 0.52

# Cek teks ambigu
ambiguous_mask = (proba_ai >= ambiguous_lower) & (proba_ai <= ambiguous_upper)
very_ambiguous_mask = (proba_ai >= very_ambiguous_lower) & (proba_ai <= very_ambiguous_upper)

ambiguous_indices = np.where(ambiguous_mask)[0]
very_ambiguous_indices = np.where(very_ambiguous_mask)[0]

print(f"\nTeks ambigu (probabilitas {ambiguous_lower*100:.0f}%-{ambiguous_upper*100:.0f}%): {len(ambiguous_indices)}")
print(f"Teks sangat ambigu (probabilitas {very_ambiguous_lower*100:.0f}%-{very_ambiguous_upper*100:.0f}%): {len(very_ambiguous_indices)}")

# Tampilkan contoh teks ambigu
if len(ambiguous_indices) > 0:
    print("\n" + "-"*70)
    print("CONTOH TEKS AMBIGU")
    print("-"*70)

    # Sort berdasarkan kedekatan ke 0.5
    ambiguous_with_dist = [(idx, abs(proba_ai[idx] - 0.5)) for idx in ambiguous_indices]
    ambiguous_with_dist.sort(key=lambda x: x[1])

    for i, (idx, dist) in enumerate(ambiguous_with_dist[:10], 1):
        true_label = original_labels[idx]
        pred_label = reverse_label_mapping[y_pred[idx]]
        prob_ai = proba_ai[idx]
        prob_manusia = 1 - prob_ai

        print(f"\n[{i}] True: {true_label}, Pred: {pred_label}")
        print(f"    Prob: AI={prob_ai*100:.1f}%, Manusia={prob_manusia*100:.1f}%")
        print(f"    Panjang: {len(X[idx])} karakter")
        print(f"    Text: {X[idx][:200]}...")

# Cek label sebenarnya dari teks ambigu
print("\n" + "-"*70)
print("LABEL SEBENARNYA DARI TEKS AMBIGU")
print("-"*70)

if len(ambiguous_indices) > 0:
    ambiguous_true_labels = [original_labels[idx] for idx in ambiguous_indices]
    ambiguous_true_label_counts = pd.Series(ambiguous_true_labels).value_counts()

    print("\nDistribusi label sebenarnya pada teks ambigu:")
    for label, count in ambiguous_true_label_counts.items():
        pct = count / len(ambiguous_indices) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

# Analisis teks yang yakin tapi salah
print("\n" + "="*70)
print("ANALISIS TEKS YANG 'YAKIN TAPI SALAH'")

wrong_mask = y_pred != y
wrong_indices = np.where(wrong_mask)[0]

if len(wrong_indices) > 0:
    # Cari teks yang yakin (>70%) tapi salah prediksinya
    high_conf_wrong_indices = []
    for idx in wrong_indices:
        prob = proba_ai[idx] if y_pred[idx] == 1 else (1 - proba_ai[idx])
        if prob > 0.7:
            high_conf_wrong_indices.append((idx, prob))

    # Sort berdasarkan confidence
    high_conf_wrong_indices.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTeks yang yakin (>70%) tapi salah: {len(high_conf_wrong_indices)}")

    if len(high_conf_wrong_indices) > 0:
        print("\n" + "-"*70)
        print("CONTOH YAKIN TAPI SALAH:")
        print("-"*70)

        for i, (idx, conf) in enumerate(high_conf_wrong_indices[:5], 1):
            true_label = original_labels[idx]
            pred_label = reverse_label_mapping[y_pred[idx]]
            prob_ai = proba_ai[idx]

            print(f"\n[{i}] True: {true_label}, Pred: {pred_label}")
            print(f"    Confidence (salah): {conf*100:.1f}%")
            print(f"    Prob AI: {prob_ai*100:.1f}%")
            print(f"    Text: {X[idx][:200]}...")

# Analisis teks borderline (zone of uncertainty)
print("\n" + "="*70)
print("ANALISIS TEKS BORDERLINE (ZONE OF UNCERTAINTY)")

correct_mask = y_pred == y
correct_indices = np.where(correct_mask)[0]

# Cari teks yang benar tapi confidence rendah
low_conf_correct = []
for idx in correct_indices:
    prob = proba_ai[idx] if y_pred[idx] == 1 else (1 - proba_ai[idx])
    if prob < 0.6:
        low_conf_correct.append((idx, prob))

low_conf_correct.sort(key=lambda x: x[1])

print(f"\nTeks yang benar tapi confidence rendah (<60%): {len(low_conf_correct)}")

if len(low_conf_correct) > 0:
    print("\n" + "-"*70)
    print("CONTOH BENAR TAPI RAGU:")
    print("-"*70)

    for i, (idx, conf) in enumerate(low_conf_correct[:5], 1):
        true_label = original_labels[idx]
        prob_ai = proba_ai[idx]

        print(f"\n[{i}] Label: {true_label}")
        print(f"    Confidence (benar): {conf*100:.1f}%")
        print(f"    Prob AI: {prob_ai*100:.1f}%")
        print(f"    Text: {X[idx][:200]}...")

# Analisis karakteristik teks ambigu
print("\n" + "="*70)
print("KARAKTERISTIK TEKS AMBIGU")

if len(ambiguous_indices) > 0:
    # Hitung panjang teks ambigu vs non-ambigu
    ambiguous_lengths = [len(X[idx]) for idx in ambiguous_indices]
    non_ambiguous_indices = np.where(~ambiguous_mask)[0]
    non_ambiguous_lengths = [len(X[idx]) for idx in non_ambiguous_indices]

    print(f"\nStatistik Panjang Teks:")
    print(f"  Ambigu:      Mean={np.mean(ambiguous_lengths):.0f}, Median={np.median(ambiguous_lengths):.0f}")
    print(f"  Non-Ambigu:  Mean={np.mean(non_ambiguous_lengths):.0f}, Median={np.median(non_ambiguous_lengths):.0f}")

    # Hitung jumlah kata
    ambiguous_words = [len(X[idx].split()) for idx in ambiguous_indices]
    non_ambiguous_words = [len(X[idx].split()) for idx in non_ambiguous_indices[:len(ambiguous_indices)]]

    print(f"\nStatistik Jumlah Kata:")
    print(f"  Ambigu:      Mean={np.mean(ambiguous_words):.0f}, Median={np.median(ambiguous_words):.0f}")
    print(f"  Non-Ambigu:  Mean={np.mean(non_ambiguous_words):.0f}, Median={np.median(non_ambiguous_words):.0f}")

# Export hasil analisis
print("\n" + "="*70)
print("EXPORT HASIL ANALISIS")

# Buat dataframe dengan hasil lengkap
results_df = pd.DataFrame({
    'text': X,
    'true_label': original_labels,
    'predicted_label': [reverse_label_mapping[p] for p in y_pred],
    'proba_ai': proba_ai,
    'proba_manusia': 1 - proba_ai,
    'is_correct': y_pred == y,
    'is_ambiguous': ambiguous_mask,
    'confidence': [max(proba_ai[idx], 1 - proba_ai[idx]) for idx in range(len(X))]
})

# Simpen ke CSV
results_df.to_csv('ambiguous_analysis_results.csv', index=False, encoding='utf-8')
print("\n[OK] Export: ambiguous_analysis_results.csv")

# Export teks ambigu aja
ambiguous_df = results_df[results_df['is_ambiguous']].copy()
ambiguous_df = ambiguous_df.sort_values('proba_ai', key=lambda x: abs(x - 0.5))
ambiguous_df.to_csv('ambiguous_texts_only.csv', index=False, encoding='utf-8')
print(f"[OK] Export: ambiguous_texts_only.csv ({len(ambiguous_df)} teks)")

# Export teks yang benar tapi rendah
low_conf_df = results_df[(results_df['is_correct']) & (results_df['confidence'] < 0.6)].copy()
low_conf_df = low_conf_df.sort_values('confidence')
low_conf_df.to_csv('low_confidence_correct.csv', index=False, encoding='utf-8')
print(f"[OK] Export: low_confidence_correct.csv ({len(low_conf_df)} teks)")

print("\n" + "="*70)
print("ANALISIS TEKS AMBIGU SELESAI!")

# Tampilkan ringkasan
print("\n" + "="*70)
print("RINGKASAN")

print(f"""
Total Data: {len(df)}
Teks Ambigu (45-55%): {len(ambiguous_indices)} ({len(ambiguous_indices)/len(df)*100:.1f}%)
Teks Sangat Ambigu (48-52%): {len(very_ambiguous_indices)} ({len(very_ambiguous_indices)/len(df)*100:.1f}%)
Teks Benar tapi Ragu: {len(low_conf_correct)} ({len(low_conf_correct)/len(df)*100:.1f}%)

Rata-rata Probabilitas AI: {proba_ai.mean()*100:.1f}%
Std Deviation: {proba_ai.std()*100:.1f}%

Interpretasi:
- Semakin kecil std dev = semakin yakin model
- Semakin banyak teks ambigu = semakin sulit membedakan
""")
