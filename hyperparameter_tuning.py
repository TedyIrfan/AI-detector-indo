"""
Hyperparameter Tuning untuk Random Forest
Mencari parameter terbaik dengan GridSearchCV
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*60)

# Load data
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()
print(f"     Total data: {len(df)}")

# Preprocessing
print("\n[2] Preprocessing...")
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

X = df['text']
y = df['label_num']

# Split
print("\n[3] Split data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Training: {len(X_train)}, Testing: {len(X_test)}")

# TF-IDF
print("\n[4] TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"     Features: {X_train_tfidf.shape[1]}")

# Parameter Grid
print("\n[5] Definisikan Parameter Grid...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("     Parameter Grid:")
print("     n_estimators: [50, 100, 200]")
print("     max_depth: [10, 20, None]")
print("     min_samples_split: [2, 5, 10]")
print("     min_samples_leaf: [1, 2, 4]")

# Grid Search
print("\n[6] Grid Search CV...")
print("     Ini akan memakan waktu beberapa menit...")

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    rf_base,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_tfidf, y_train)

print("\n     Grid Search selesai!")

# Best Parameters
print("\n" + "="*60)
print("HASIL GRID SEARCH")
print("="*60)

print(f"\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# Evaluasi dengan model terbaik
print("\n[7] Evaluasi Model Terbaik...")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Manusia (0)', 'AI (1)']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predict: 0    Predict: 1")
print(f"Actual: 0        {cm[0][0]:<6}       {cm[0][1]:<6}")
print(f"Actual: 1        {cm[1][0]:<6}       {cm[1][1]:<6}")

# Compare dengan baseline
print("\n" + "="*60)
print("COMPARISON: BASELINE vs TUNED")
print("="*60)

print(f"\nBaseline (sebelum tuning):")
print(f"  - Akurasi: 98.68%")
print(f"  - Params: n_estimators=100, max_depth=20, dll (default)")

print(f"\nTuned (setelah GridSearch):")
print(f"  - Akurasi: {accuracy*100:.2f}%")
print(f"  - Params: {grid_search.best_params_}")

diff = accuracy - 0.9868
if diff > 0:
    print(f"\n  [+] Improvement: +{diff*100:.2f}%")
else:
    print(f"\n  [-] Change: {diff*100:.2f}%")

print("\n" + "="*60)
print("HYPERPARAMETER TUNING SELESAI!")
print("="*60)
