"""
Script Inference - Prediksi Teks Baru
Run langsung di terminal (bukan background)
"""

import joblib
import sys

print("="*60)
print("INFERENCE - PREDIKSI TEKS AI vs MANUSIA")
print("="*60)

# Load model
print("\n[1] Load model...")
try:
    model = joblib.load('models/random_forest_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    label_mapping = joblib.load('models/label_mapping.pkl')
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    print("     [OK] Model loaded!")
except Exception as e:
    print(f"     [X] Error loading model: {e}")
    sys.exit(1)

# Interactive inference
print("\n" + "="*60)
print("MODE INTERAKTIF")
print("="*60)
print("\nKetik teks untuk prediksi")
print("Ketik 'exit' untuk keluar\n")

prediction_count = 0

while True:
    try:
        user_input = input(">>> ")

        if user_input.lower().strip() == 'exit':
            print("\nKeluar dari program. Terima kasih!")
            break

        if not user_input.strip():
            print("[!] Teks tidak boleh kosong!")
            continue

        # Predict
        text_tfidf = vectorizer.transform([user_input])
        pred_num = model.predict(text_tfidf)[0]
        pred_proba = model.predict_proba(text_tfidf)[0]
        pred_label = reverse_label_mapping[pred_num]

        prediction_count += 1

        print(f"\n  Hasil Prediksi: {pred_label}")
        print(f"  Confidence: {pred_proba[pred_num]*100:.1f}%")

        # Status confidence
        if pred_proba[pred_num] > 0.8:
            print(f"  Status: TINGGI (Yakin)")
        elif pred_proba[pred_num] > 0.5:
            print(f"  Status: SEDANG")
        else:
            print(f"  Status: RENDAH (Kurang yakin)")

        # Detail probabilitas
        print(f"  Detail: P(AI)={pred_proba[1]*100:.1f}%, P(Manusia)={pred_proba[0]*100:.1f}%")
        print(f"  Total predictions: {prediction_count}")

    except KeyboardInterrupt:
        print("\n\nProgram dihentukan user.")
        break
    except Exception as e:
        print(f"\n[X] Error: {e}")

print("\n" + "="*60)
