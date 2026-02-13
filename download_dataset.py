"""
Download Dataset & Model dari Google Drive
Jalankan script ini untuk mengunduh file yang dibutuhkan
"""

import os
import webbrowser

print("="*60)
print("DOWNLOAD DATASET & MODEL DARI GOOGLE DRIVE")
print("="*60)

# Link Google Drive Folder
GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/15vydyLv3M7Ap4lruqLBWxV80PvIg4hHA?usp=sharing"

print("\n📂 Google Drive Folder akan dibuka di browser...")
print("\nFile yang perlu diunduh:")
print("  1. dataset_final_1500.csv (WAJIB)")
print("  2. models_indobert/ folder (OPTIONAL - untuk IndoBERT)")

print("\nLangkah-langkah:")
print("  1. Folder Google Drive akan terbuka di browser")
print("  2. Download file yang dibutuhkan")
print("  3. Letakkan di folder project ini")
print("  4. Jalankan training atau aplikasi")

print("\n" + "-"*60)
input("Tekan ENTER untuk membuka Google Drive...")

# Buka browser
webbrowser.open(GOOGLE_DRIVE_FOLDER)

print("\n✅ Google Drive terbuka di browser!")
print("   Silakan download file yang dibutuhkan.")
print("   Pastikan letakkan di folder: " + os.getcwd())

print("\n" + "="*60)
