# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("Training Random Forest - TF-IDF")

# Load dataset dulu
print("\nLoad dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
print(f"     Total data: {len(df)}")

# Cek missing values
print("\nCek missing values...")
print(f"     Missing: {df.isnull().sum().sum()}")
df = df.dropna()
print(f"     After drop NA: {len(df)}")

# Preprocessing Label
print("\nPreprocessing Label...")
print(f"     Label mapping: MANUSIA -> 0, AI -> 1")
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)
print(f"     Label distribution:")
print(df['label_num'].value_counts().sort_index())

# Siapin X dan y
X = df['text']
y = df['label_num']

# Split data (80% Train, 20% Test)
print("\nSplit data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"     Testing:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# TF-IDF Vectorization
print("\nTF-IDF Vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words=None
)

# Fit ke train data terus transform test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"     Feature shape: {X_train_tfidf.shape}")
print(f"     Vocabulary size: {len(vectorizer.vocabulary_)}")

# Training Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Latih model
rf_model.fit(X_train_tfidf, y_train)
print("     Training completed!")

# Prediksi test set
print("\nPrediction...")
y_pred = rf_model.predict(X_test_tfidf)

# Tampilin hasil evaluasi
print("\nHasil Evaluasi")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Manusia (0)', 'AI (1)']))

# Tampilin confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predict: 0    Predict: 1")
print(f"Actual: 0        {cm[0][0]:<6}       {cm[0][1]:<6}")
print(f"Actual: 1        {cm[1][0]:<6}       {cm[1][1]:<6}")

# Tampilin top 10 fitur penting
print("\nTop 10 Important Features")

feature_names = vectorizer.get_feature_names_out()
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

for i, idx in enumerate(indices, 1):
    print(f"{i}. {feature_names[idx]} ({importances[idx]:.4f})")

# Summary
print("\nTraining Selesai!")
print(f"\nModel: Random Forest Classifier")
print(f"Vectorization: TF-IDF")
print(f"Features: {len(feature_names)}")
print(f"Akurasi: {accuracy*100:.2f}%")
