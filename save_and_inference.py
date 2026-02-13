"""
Step 9: Simpan Model
Step 10: Inference/Prediksi Baru
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("STEP 9: SIMPAN MODEL")
print("="*60)

# Load dan preprocess data
print("\n[1] Load dan preprocess data...")
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

# TF-IDF
print("\n[2] TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training model (baseline - yang 98.68%)
print("\n[3] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)

# Evaluasi sebentar
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"     Model accuracy: {accuracy*100:.2f}%")

# Buat folder models
os.makedirs('models', exist_ok=True)

# Simpan model, vectorizer, dan label encoder
print("\n[4] Simpan Model dan Artifacts...")
joblib.dump(model, 'models/random_forest_model.pkl')
print("     [OK] Model saved: models/random_forest_model.pkl")

joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("     [OK] Vectorizer saved: models/tfidf_vectorizer.pkl")

joblib.dump(label_mapping, 'models/label_mapping.pkl')
print("     [OK] Label mapping saved: models/label_mapping.pkl")

# Simpan metadata
metadata = {
    'model_name': 'Random Forest Classifier',
    'accuracy': accuracy,
    'n_features': X_train_tfidf.shape[1],
    'train_size': len(X_train),
    'test_size': len(X_test),
    'label_mapping': label_mapping
}

joblib.dump(metadata, 'models/metadata.pkl')
print("     [OK] Metadata saved: models/metadata.pkl")

print("\n" + "="*60)
print("STEP 10: INFERENCE/PREDIKSI BARU")
print("="*60)

# Load model
print("\n[1] Load model...")
loaded_model = joblib.load('models/random_forest_model.pkl')
loaded_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
loaded_label_mapping = joblib.load('models/label_mapping.pkl')
reverse_label_mapping = {v: k for k, v in loaded_label_mapping.items()}

print("     [OK] Model loaded!")

# Contoh teks baru
print("\n[2] Test Prediksi dengan Teks Baru...")

sample_texts = [
    "Liverpool baru saja memenangkan pertandingan melawan Manchester City dengan skor 2-1 di Anfield. Mohamed Salah mencetak dua gol penting.",
    "Harga cabe di pasar lagi naik. Sekarang udah Rp 100 ribu per kilo. Padahal kemarin masih Rp 50 ribu. Bingung banget sih.",
    "Teknologi artificial intelligence semakin canggih. Banyak perusahaan yang mulai mengimplementasikan AI untuk otomatisasi pekerjaan.",
    "guys, udah pada nonton film terbaru Marvel belum? gila banget visual effects-nya!",
    "Pemerintah mengumumkan kebijakan baru untuk pengembangan ekonomi digital di Indonesia."
]

for i, text in enumerate(sample_texts, 1):
    # Preprocess
    text_tfidf = loaded_vectorizer.transform([text])

    # Predict
    pred_num = loaded_model.predict(text_tfidf)[0]
    pred_proba = loaded_model.predict_proba(text_tfidf)[0]

    # Get label
    pred_label = reverse_label_mapping[pred_num]

    print(f"\n  Teks {i}:")
    print(f"  Content: {text[:100]}...")
    print(f"  Prediksi: {pred_label}")
    print(f"  Confidence: {pred_proba[pred_num]*100:.1f}%")

# Interactive inference
print("\n" + "="*60)
print("INFERENCE MODE")
print("="*60)
print("\nKetik teks untuk prediksi (ketik 'exit' untuk keluar):")

while True:
    user_input = input("\nMasukkan teks: ")

    if user_input.lower() == 'exit':
        print("Keluar dari inference mode.")
        break

    if not user_input.strip():
        print("Teks tidak boleh kosong!")
        continue

    # Predict
    text_tfidf = loaded_vectorizer.transform([user_input])
    pred_num = loaded_model.predict(text_tfidf)[0]
    pred_proba = loaded_model.predict_proba(text_tfidf)[0]
    pred_label = reverse_label_mapping[pred_num]

    print(f"\n  Hasil Prediksi: {pred_label}")
    print(f"  Confidence: {pred_proba[pred_num]*100:.1f}%")

    if pred_proba[pred_num] > 0.8:
        print(f"  Status: TINGGI (Yakin)")
    elif pred_proba[pred_num] > 0.5:
        print(f"  Status: SEDANG")
    else:
        print(f"  Status: RENDAH (Kurang yakin)")

print("\n" + "="*60)
print("SELESAI!")
print("="*60)
print("\nModel dan artifacts tersimpan di folder 'models/'")
