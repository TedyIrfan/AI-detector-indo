"""
Bersihkan file yang tidak terpakai
Pindahkan file lama ke folder archive
"""

import os
import shutil
from datetime import datetime

# File yang PENTING - jangan dihapus
KEEP_FILES = [
    'merge_dataset.py',
    'cleanup_ai_data.py',
    'train_all_models_1000.py',
    'compare_models.py',
    'app.py',
]

# File yang akan di-arsipkan (dipindah ke folder archive)
ARCHIVE_PY = [
    # Test/Debug scripts
    'debug_openrouter.py',
    'debug_qwen.py',
    'test_groq.py',
    'test_groq_gpt_oss.py',
    'test_groq_quality.py',
    'test_hf_direct.py',
    'test_hf_speed.py',
    'test_huggingface.py',
    'test_huggingface_correct.py',
    'test_huggingface_v2.py',
    'test_kimi.py',
    'test_openrouter.py',
    'test_qwen.py',
    'test_qwen3_hf.py',
    'test_qwen3_indo.py',
    # Generate scripts (lama)
    'generate_ai_openrouter.py',
    'generate_groq.py',
    'generate_groq_gpt_oss.py',
    'generate_hf_qwen.py',
    'generate_ai_groq_100_long.py',
    # Filter scripts
    'filter_ai_groq_100.py',
    'filter_indosum_or.py',
    'filter_kaggle_twitter_or.py',
    'filter_reddit_auto.py',
    'filter_reddit_max250.py',
    'filter_terrorism_max82.py',
    # Add AI scripts
    'add_ai_gpt_manual.py',
    'add_gemini_gpt_batch.py',
    'add_gpt_batch2.py',
    'add_gpt_batch4.py',
    'data_ai_gpt_10.py',
]

# File data parsial yang akan di-arsipkan (karena sudah digabung)
ARCHIVE_CSV = [
    'data_ai_openrouter_72.csv',
    'data_ai_openrouter_96.csv',
    'data_ai_openrouter_99.csv',
    'data_ai_openrouter_100.csv',
    'data_ai_hf_qwen_30.csv',
    'data_ai_hf_qwen_100.csv',
    'data_ai_groq_100.csv',
    'data_ai_groq_gptoss_80.csv',
]

print("="*60)
print("BERSIH-BERSIH FILE LAMA")
print("="*60)

# Buat folder archive
archive_folder = "archive_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(archive_folder, exist_ok=True)
print(f"\nFolder archive: {archive_folder}")

# Arsipkan Python files
print("\n[1] Arsip Python files...")
moved_py = 0
for f in ARCHIVE_PY:
    if os.path.exists(f):
        shutil.move(f, f"{archive_folder}/{f}")
        print(f"     Moved: {f}")
        moved_py += 1

print(f"     Total Python files moved: {moved_py}")

# Arsipkan CSV files
print("\n[2] Arsip CSV files...")
moved_csv = 0
for f in ARCHIVE_CSV:
    if os.path.exists(f):
        shutil.move(f, f"{archive_folder}/{f}")
        print(f"     Moved: {f}")
        moved_csv += 1

print(f"     Total CSV files moved: {moved_csv}")

# Sisa file penting
print("\n[3] File penting yang DIPERTAHANKAN:")
for f in KEEP_FILES:
    if os.path.exists(f):
        print(f"     [KEEP] {f}")

# Summary
print("\n" + "="*60)
print("SELESAI!")
print("="*60)
print(f"\nTotal files moved to archive: {moved_py + moved_csv}")
print(f"Archive folder: {archive_folder}/")
print(f"\nFile FINAL yang penting:")
print(f"  - dataset_final_1500.csv (DATASET UTAMA)")
print(f"  - data_ai_all_clean.csv (backup AI)")
print(f"  - data_manusia_*.csv (backup manusia)")
