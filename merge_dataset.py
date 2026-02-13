"""
Merge AI dan Human Data jadi Dataset Final
"""

import pandas as pd
import glob
from sklearn.utils import shuffle

print("="*60)
print("MERGE DATA AI + MANUSIA")
print("="*60)

# Load AI data
print("\n[1] Load AI data...")
df_ai = pd.read_csv("data_ai_all_clean.csv", encoding='utf-8')
print(f"     AI data: {len(df_ai)} rows")

# Load Human data
print("\n[2] Load Human data...")
human_files = glob.glob("data_manusia*.csv")
all_human = []

for f in sorted(human_files):
    df = pd.read_csv(f, encoding='utf-8')
    all_human.append(df)

df_human = pd.concat(all_human, ignore_index=True)
print(f"     Human data: {len(df_human)} rows")

# Tambah label
df_ai_final = df_ai[['text', 'label']].copy()
df_human_final = df_human[['text']].copy()
df_human_final['label'] = 'MANUSIA'

print(f"\n[3] Label distribution:")
print(f"     AI: {len(df_ai_final)}")
print(f"     MANUSIA: {len(df_human_final)}")

# Merge
print("\n[4] Merge data...")
df_merged = pd.concat([df_ai_final, df_human_final], ignore_index=True)
print(f"     Total: {len(df_merged)} rows")

# Shuffle
print("\n[5] Shuffle data...")
df_shuffled = shuffle(df_merged, random_state=42)
print(f"     Shuffled!")

# Reset index
df_shuffled = df_shuffled.reset_index(drop=True)

# Save
filename = "dataset_final_1500.csv"
df_shuffled.to_csv(filename, index=False, encoding='utf-8')

print(f"\n[6] Saved to: {filename}")

# Final summary
print("\n" + "="*60)
print("FINAL DATASET SUMMARY")
print("="*60)
print(f"\nFile: {filename}")
print(f"Total rows: {len(df_shuffled)}")
print(f"\nLabel distribution:")
print(df_shuffled['label'].value_counts())

print(f"\nFirst 5 rows:")
print(df_shuffled.head())

print(f"\nLast 5 rows:")
print(df_shuffled.tail())

print(f"\n" + "="*60)
print("DATASET FINAL SELESAI!")
print("="*60)
