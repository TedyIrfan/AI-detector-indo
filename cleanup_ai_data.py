"""
Cek dan Clean Up semua AI Data
"""

import pandas as pd
import os
from glob import glob

# Cari semua file AI data
print("="*60)
print("CEK SEMUA AI DATA FILES")
print("="*60)

ai_files = glob("data_ai_*.csv")
print(f"\nDitemukan {len(ai_files)} file AI:")

all_data = []
file_summary = []

for file in sorted(ai_files):
    try:
        df = pd.read_csv(file, encoding='utf-8')
        print(f"\n{file}:")
        print(f"  - Jumlah baris: {len(df)}")

        # Cek kolom
        print(f"  - Kolom: {list(df.columns)}")

        # Cek nilai unik per kolom penting
        if 'model' in df.columns:
            print(f"  - Model: {df['model'].unique()}")
        if 'style' in df.columns:
            print(f"  - Style: {df['style'].value_counts().to_dict()}")
        if 'topic' in df.columns:
            print(f"  - Topics: {df['topic'].value_counts().to_dict()}")

        all_data.append(df)
        file_summary.append({
            'file': file,
            'rows': len(df),
            'model': df['model'].unique()[0] if 'model' in df.columns else 'Unknown'
        })

    except Exception as e:
        print(f"\n{file}: ERROR - {e}")

if all_data:
    # Gabungkan semua
    print("\n" + "="*60)
    print("MENGGABUNGKAN SEMUA DATA")
    print("="*60)

    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal sebelum clean up: {len(df_combined)} data")

    # Clean up
    print("\n" + "="*60)
    print("CLEAN UP PROCESS")
    print("="*60)

    # 1. Hapus baris dengan text kosong atau terlalu pendek
    before_len = len(df_combined)
    df_combined = df_combined[df_combined['text'].str.len() > 100]
    print(f"\n1. Hapus text < 100 karakter: {before_len} -> {len(df_combined)} (dihapus {before_len - len(df_combined)})")

    # 2. Hapus text yang jelas kotor (sisa prompt) - lebih spesifik
    before_len = len(df_combined)
    dirty_patterns = [
        r'^\[nama topik\]', r'^\[isi teks\]', r'^TOPIK:\s*\[nama topik\]',
        r'^\.\.\. dan seterusnya sampai', r'^Buatkan 10 teks berbeda',
    ]
    for pattern in dirty_patterns:
        df_combined = df_combined[~df_combined['text'].str.contains(pattern, case=False, regex=True, na=False)]
    print(f"2. Hapus text kotor (sisa prompt): {before_len} -> {len(df_combined)} (dihapus {before_len - len(df_combined)})")

    # 3. Hapus duplikat
    before_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['text'], keep='first')
    print(f"3. Hapus duplikat: {before_len} -> {len(df_combined)} (dihapus {before_len - len(df_combined)})")

    # 4. Reset index
    df_combined = df_combined.reset_index(drop=True)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY SETELAH CLEAN UP")
    print("="*60)
    print(f"\nTotal final: {len(df_combined)} data AI")
    print(f"\nPer model:")
    if 'model' in df_combined.columns:
        print(df_combined['model'].value_counts().to_string())
    print(f"\nPer style:")
    if 'style' in df_combined.columns:
        print(df_combined['style'].value_counts().to_string())
    print(f"\nPer topic:")
    if 'topic' in df_combined.columns:
        print(df_combined['topic'].value_counts().to_string())

    # Save combined file
    output_file = "data_ai_all_clean.csv"
    df_combined.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nSaved to: {output_file}")

    # Progress ke target 750
    print("\n" + "="*60)
    print("PROGRESS KE TARGET 750 AI DATA")
    print("="*60)
    current = len(df_combined)
    target = 750
    remaining = target - current
    percentage = (current / target) * 100

    print(f"\nCurrent: {current}")
    print(f"Target:  {target}")
    print(f"Remaining: {remaining}")
    print(f"Progress: {percentage:.1f}%")

    bar_length = 40
    filled = int(bar_length * current / target)
    bar = '#' * filled + '-' * (bar_length - filled)
    print(f"\n[{bar}] {percentage:.1f}%")

else:
    print("\nTidak ada data yang ditemukan!")
