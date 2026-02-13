# 🤖 Deteksi Tulisan AI vs Manusia (Bahasa Indonesia)

Project Skripsi - Sistem Klasifikasi Machine Learning untuk membedakan tulisan yang dihasilkan oleh AI dan manusia dalam Bahasa Indonesia.

## 📊 Dataset

### 📥 Download Dataset & Model
Dataset dan model besar dapat diunduh dari Google Drive:

**[Google Drive Folder](https://drive.google.com/drive/folders/15vydyLv3M7Ap4lruqLBWxV80PvIg4hHA?usp=sharing)**

Isi folder:
- `dataset_final_1500.csv` - Dataset utama (~1.5 MB)
- `models_indobert/` - IndoBERT model (~475 MB)

**Cara pakai:**
1. Clone repository ini
2. Download dataset dari Google Drive di atas
3. Letakkan `dataset_final_1500.csv` di folder project
4. (Opsional) Download & letakkan folder `models_indobert/` untuk pakai IndoBERT

### Info Dataset
- **Total Data:** 1,510 teks
- **MANUSIA:** 750 teks (49.7%)
- **AI:** 760 teks (50.3%)

### Sumber Data Manusia
- IndoSum (Berita)
- Reddit
- Twitter
- Kaggle Terrorism
- Kaggle Marketplace

### Sumber Data AI
- OpenRouter (3 model)
- Groq (1 model)
- HuggingFace (2 model)

## 🎯 Hasil Training

| Model | Test Accuracy | Precision | Recall | F1-Score |
|--------|---------------|-----------|--------|-----------|
| Logistic Regression | **100.00%** | 100% | 100% | 100% |
| SVM (RBF Kernel) | **100.00%** | 100% | 100% | 100% |
| Random Forest | 98.68% | 97.4% | 100% | 98.7% |
| IndoBERT | **100.00%** | 100% | 100% | 100% |

## 📁 Struktur Project

```
.
├── dataset_final_1500.csv          # Dataset utama
├── train_strict_cv.py              # Training dengan pipeline (no leakage)
├── train_group_cv.py               # Training dengan group-based CV
├── threshold_tuning.py             # Analisis threshold
├── train_indobert.py               # Training IndoBERT
├── eda_analysis.py                # Exploratory Data Analysis
├── linguistic_analysis.py          # Analisis linguistik
├── error_analysis.py               # Analisis error
├── ablation_study.py               # Feature importance
├── per_source_analysis.py          # Analisis per sumber
├── app.py                         # Streamlit web app
├── models_strict/                  # Model & hasil training
│   ├── best_pipeline_logistic_regression.pkl
│   ├── strict_cv_results.json
│   ├── threshold_analysis.json
│   └── group_cv_results.json
├── models_indobert/                # Model IndoBERT
│   ├── indobert_results.json
│   └── final_model/
└── visualizations/                 # Visualisasi
    ├── roc_curve.png
    ├── fnr_analysis.png
    ├── threshold_metrics_comparison.png
    └── ...
```

## 🚀 Cara Penggunaan

### 1. Training Model

```bash
# Training dengan strict cross-validation (no leakage)
python train_strict_cv.py

# Training dengan group-based CV (validasi sumber)
python train_group_cv.py

# Threshold tuning
python threshold_tuning.py
```

### 2. Run Web App (Streamlit)

```bash
# Install streamlit jika belum
pip install streamlit

# Jalankan aplikasi
streamlit run app.py
```

### 3. Prediksi dengan Model

```python
import joblib

# Load model
pipeline = joblib.load('models_strict/best_pipeline_logistic_regression.pkl')

# Prediksi
text = "Masukkan teks di sini..."
label = pipeline.predict([text])[0]
label_name = "MANUSIA" if label == 0 else "AI"
proba = pipeline.predict_proba([text])[0]

print(f"Prediksi: {label_name}")
print(f"Confidence: {max(proba)*100:.2f}%")
```

## 📦 Requirements

```
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
wordcloud
streamlit
torch
transformers
```

Install:
```bash
pip install -r requirements.txt
```

## 📈 Hasil Analisis

### Cross-Validation Results (Strict CV - No Leakage)
- Logistic Regression: **99.67% ± 0.41%**
- SVM (RBF): **99.67% ± 0.41%**
- Random Forest: **98.59% ± 0.91%**

### Group-Based CV (Validasi Sumber)
- SVM: **96.72%** (gap -2.95% dari standard CV)
- Logistic Regression: **87.66%** (gap -12.01%)
- Random Forest: **82.92%** (gap -15.75%)

### Threshold Recommendations
- **Default (0.5):** Untuk penggunaan umum
- **Balanced (0.55):** Rekomendasi untuk production
- **Conservative (0.60):** Untuk keamanan tinggi (FNR lebih rendah)

## 👤 Author

Dibuat untuk Skripsi Sarjana Teknik Informatika

## 📄 Lisensi

Project ini hanya untuk keperluan edukasi dan penelitian skripsi.
