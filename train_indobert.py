# Import library yang diperlukan
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Cek apakah PyTorch dan Transformers tersedia
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    HAS_TRANSFORMERS = True
except ImportError as e:
    HAS_TRANSFORMERS = False
    missing = str(e)
    print(f"Error: Missing dependencies - {missing}")
    print("\nPlease install required packages:")
    print("  pip install torch transformers scikit-learn")

print("Model Transformer IndoBERT - Deteksi AI vs Manusia")

# Kalau dependencies gak ada, stop di sini
if not HAS_TRANSFORMERS:
    print("\nCannot proceed without required dependencies.")
    exit(1)

# Setup device (GPU kalau ada)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load dataset dulu
print("\nLoad dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna(subset=['text', 'label'])
print(f"    Total data: {len(df)}")
print(f"    - MANUSIA: {(df['label'] == 'MANUSIA').sum()}")
print(f"    - AI: {(df['label'] == 'AI').sum()}")

# Mapping label biar jadi angka
label_mapping = {'MANUSIA': 0, 'AI': 1}
df['label_num'] = df['label'].map(label_mapping)

# Split data jadi train, test, terus validation
print("\nSplit data (80% train, 20% test)...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label_num'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label_num']
)

# Split train jadi train dan validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts,
    train_labels,
    test_size=0.1,
    random_state=42,
    stratify=train_labels
)

print(f"    Training: {len(train_texts)}")
print(f"    Validation: {len(val_texts)}")
print(f"    Testing: {len(test_texts)}")

# Load tokenizer IndoBERT
print("\nLoading IndoBERT tokenizer...")

MODEL_NAME = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"    Model: {MODEL_NAME}")

# Max length buat tokenization
MAX_LENGTH = 512

# Tokenize semua text
print("\nTokenizing texts...")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

print(f"    Max length: {MAX_LENGTH}")
print(f"    Tokenized samples: train={len(train_encodings['input_ids'])}, val={len(val_encodings['input_ids'])}, test={len(test_encodings['input_ids'])}")

# Dataset class buat PyTorch
class IndoBERTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Buat dataset
train_dataset = IndoBERTDataset(train_encodings, train_labels)
val_dataset = IndoBERTDataset(val_encodings, val_labels)
test_dataset = IndoBERTDataset(test_encodings, test_labels)

# Load model IndoBERT
print("\nLoading IndoBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(device)

print(f"    Model loaded successfully!")
print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

def compute_metrics(pred):
    """Hitung metrics buat evaluasi"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Setup training arguments
print("\nSetting up training arguments...")

OUTPUT_DIR = "./models_indobert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Konfigurasi training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=42,
    fp16=device.type == 'cuda',
    gradient_accumulation_steps=2,
)

print(f"    Epochs: {training_args.num_train_epochs}")
print(f"    Batch size: {training_args.per_device_train_batch_size}")
print(f"    Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    FP16: {training_args.fp16}")

# Inisialisasi Trainer
print("\nInitializing Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Mulai training
print("\nMelatih model IndoBERT")
print("Estimasi waktu: 30-60 menit (CPU), 5-10 menit (GPU)")

trainer.train()

# Evaluasi di test set
print("\nEvaluasi pada test set")

predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# Hitung metrics
accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

print(f"\nTest Results:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")

print(f"\nClassification Report:")
print(classification_report(labels, preds, target_names=['MANUSIA', 'AI']))

# Hitung confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, preds)

print(f"\nConfusion Matrix:")
print(f"                Predicted: MANUSIA    Predicted: AI")
print(f"Actual: MANUSIA     {cm[0][0]:>8}          {cm[0][1]:>8}")
print(f"Actual: AI          {cm[1][0]:>8}          {cm[1][1]:>8}")

# Hitung False Negative Rate
tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
print(f"\nFalse Negative Rate: {fnr*100:.2f}%")
print(f"({fn} AI texts classified as MANUSIA)")

# Simpen model
print("\nMenyimpan model")

model_path = f"{OUTPUT_DIR}/final_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"    Model saved to: {model_path}")

# Kumpulin hasil ke dictionary
results = {
    'model_name': MODEL_NAME,
    'test_accuracy': accuracy,
    'test_precision': precision,
    'test_recall': recall,
    'test_f1': f1,
    'false_negative_rate': fnr,
    'confusion_matrix': {
        'tn': int(tn), 'fp': int(fp),
        'fn': int(fn), 'tp': int(tp)
    },
    'training_args': {
        'epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'max_length': MAX_LENGTH
    }
}

# Simpen ke JSON
with open(f"{OUTPUT_DIR}/indobert_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"    Results saved to: {OUTPUT_DIR}/indobert_results.json")

# Contoh fungsi inferensi
print("\nContoh fungsi inferensi")

def predict_text(text, model_path=f"{OUTPUT_DIR}/final_model"):
    """Prediksi teks pake model IndoBERT"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred_label = "AI" if probs[0][1] > 0.5 else "MANUSIA"
    confidence = probs[0][1].item() if pred_label == "AI" else probs[0][0].item()

    return {
        'label': pred_label,
        'confidence': confidence,
        'probabilities': {
            'MANUSIA': probs[0][0].item(),
            'AI': probs[0][1].item()
        }
    }

print("\nTesting inference function:")

sample_texts = [
    "Jakarta, CNN Indonesia - Pemerintah memastikan akan terus meningkatkan pembangunan infrastruktur di seluruh Indonesia.",
    "Implementasi teknologi AI dalam kehidupan sehari-hari membawa perubahan signifikan dalam berbagai aspek kehidupan manusia modern."
]

for text in sample_texts:
    result = predict_text(text)
    print(f"\nText: {text[:60]}...")
    print(f"  Prediction: {result['label']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")

print("\nTraining IndoBERT selesai!")
print(f"""
Hasil:
- Test Accuracy: {accuracy*100:.2f}%
- Test F1-Score: {f1*100:.2f}%
- False Negative Rate: {fnr*100:.2f}%

Files tersimpan di {OUTPUT_DIR}/:
- final_model/          (model dan tokenizer)
- indobert_results.json (hasil evaluasi)

Untuk menggunakan model:
    from train_indobert import predict_text
    result = predict_text("Teks yang ingin diuji")
""")
