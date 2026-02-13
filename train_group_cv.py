# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupKFold, train_test_split, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
import json
import os
warnings.filterwarnings('ignore')

print("Validasi Silang Berbasis Group")

# Load dataset dulu
print("\nLoad dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])
print(f"    Total data: {len(df)}")

# Deteksi sumber teks buat grouping
print("\nDetecting source groups...")

def detect_source(text):
    """Deteksi sumber teks buat grouping"""
    text = str(text).lower()

    # Cek IndoSum news dari pola byline
    if any(pattern in text[:100] for pattern in ['cnn indonesia', 'kompas', 'antara', 'republika', 'tribun']):
        return 'indosum_news'

    # Twitter ada @ dan hashtag
    if '@' in text or text.count('#') >= 2:
        return 'twitter'

    # AI text biasanya pake kata-kata formal
    formal_indicators = [
        'implementasi', 'optimalisasi', 'transformasi', 'peningkatan',
        'pengembangan', 'berkelanjutan', 'komprehensif', 'strategis'
    ]
    if sum(1 for ind in formal_indicators if ind in text) >= 3:
        return 'ai_formal'

    # Klasifikasi berdasarkan panjang teks
    if len(text) > 500:
        return 'long_form'
    elif len(text) > 200:
        return 'medium_form'
    else:
        return 'short_form'

# Apply deteksi source ke dataset
df['source_group'] = df['text'].apply(detect_source)

# Gabungin label sama source group biar lebih spesifik
df['source_group'] = df.apply(
    lambda row: f"{row['label'].lower()}_{row['source_group']}",
    axis=1
)

print(f"\nDistribusi Source Groups:")
for group, count in df['source_group'].value_counts().items():
    print(f"  {group}: {count}")

# Preprocessing label
print("\nPreparing data...")

# Mapping label biar jadi angka
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text'].values
y = df['label_num'].values
groups = df['source_group'].values

# Pake GroupKFold biar gak ada leakage dari sumber yang sama
print("\nMelatih model dengan GroupKFold untuk mencegah leakage dari sumber yang sama")

# Setup TF-IDF
tfidf_params = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.8
}

# Define 3 model: RF, Logistic Regression, SVM
pipelines = {
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ]),
    'SVM': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42, C=1.0))
    ])
}

# Jumlah fold gak boleh lebih dari jumlah group
n_groups = len(np.unique(groups))
n_splits = min(5, n_splits)

print(f"\nNumber of unique groups: {n_groups}")
print(f"Number of splits: {n_splits}")

# Setup GroupKFold
gkf = GroupKFold(n_splits=n_splits)

# Simpen hasil tiap model
results = {}

# Latih tiap model dengan Group CV
for name, pipeline in pipelines.items():
    print(f"\nModel: {name}")

    fold_scores = []

    print(f"\nFold-by-Fold Results:")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        # Split data pake index
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Cek group yang dipake
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        # Fit model ke fold ini
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict(X_test_fold)
        acc = accuracy_score(y_test_fold, y_pred)

        fold_scores.append(acc)

        print(f"  Fold {fold}: {acc*100:.2f}%")
        print(f"    Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
        print(f"    Train size: {len(X_train_fold)}, Test size: {len(X_test_fold)}")

    # Hitung rata-rata dan standar deviasi
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\nGroup CV Results:")
    print(f"  Mean Accuracy: {mean_score*100:.2f}% (±{std_score*100:.2f}%)")
    print(f"  Min - Max:     {min(fold_scores)*100:.2f}% - {max(fold_scores)*100:.2f}%")

    results[name] = {
        'fold_scores': fold_scores,
        'mean': mean_score,
        'std': std_score
    }

# Bandingin dengan standard CV
print("\nPerbandingan: Group CV vs Standard CV")

from sklearn.model_selection import StratifiedKFold

# Setup standard CV ( Stratified KFold )
standard_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Tampilin tabel perbandingan
print(f"\n{'Model':<25} {'Group CV':<20} {'Standard CV':<20} {'Diff':<15}")
print("-"*80)

for name, pipeline in pipelines.items():
    # Hitung standard CV score
    std_scores = cross_val_score(pipeline, X, y, cv=standard_cv, scoring='accuracy', n_jobs=-1)
    std_mean = std_scores.mean()

    # Ambil Group CV score
    group_mean = results[name]['mean']

    # Hitung selisih
    diff = group_mean - std_mean

    print(f"{name:<25} {group_mean*100:>6.2f}% (±{results[name]['std']*100:>4.2f}%)  {std_mean*100:>6.2f}% (±{std_scores.std()*100:>4.2f}%)  {diff*100:>+6.2f}%")

print("""
Interpretasi:
- Group CV < Standard CV: Ada kemungkinan leakage dari sumber yang sama
- Group CV ~ Standard CV: Model generalisasi dengan baik antar sumber
- Group CV > Standard CV: Jarang terjadi, mungkin kebetulan distribusi group
""")
