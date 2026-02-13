# Import library yang diperlukan
import pandas as pd
import numpy as np
import joblib
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("ANALISIS LINGUISTIK - PERBEDAAN AI vs MANUSIA")

# Load dataset dulu
print("\nLoad dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()

# Pisah teks AI dan Manusia
ai_texts = df[df['label'] == 'AI']['text'].tolist()
human_texts = df[df['label'] == 'MANUSIA']['text'].tolist()

print(f"     AI texts: {len(ai_texts)}")
print(f"     Human texts: {len(human_texts)}")

# Analisis 1: Panjang kalimat rata-rata
print("\n" + "="*70)
print("1. ANALISIS PANJANG KALIMAT")

def avg_sentence_length(texts):
    """Hitung rata-rata panjang kalimat dalam kata"""
    total_words = 0
    total_sentences = 0

    for text in texts:
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        total_words += len(words)
        total_sentences += len(sentences)

    return total_words / max(total_sentences, 1)

# Hitung rata-rata panjang kalimat
ai_avg_sent_len = avg_sentence_length(ai_texts)
human_avg_sent_len = avg_sentence_length(human_texts)

print(f"\nRata-rata panjang kalimat (dalam kata):")
print(f"  AI:      {ai_avg_sent_len:.2f} kata/kalimat")
print(f"  Manusia: {human_avg_sent_len:.2f} kata/kalimat")
print(f"  Selisih: {abs(ai_avg_sent_len - human_avg_sent_len):.2f} kata")

# Analisis 2: Keragaman leksikal (Lexical Diversity)
print("\n" + "="*70)
print("2. KERAGAMAN LEKSIKAL (LEXICAL DIVERSITY)")

def lexical_diversity(texts):
    """Hitung TTR (Type-Token Ratio)"""
    all_words = []

    for text in texts:
        words = text.lower().split()
        all_words.extend(words)

    unique_words = set(all_words)
    ttr = len(unique_words) / len(all_words) if len(all_words) > 0 else 0

    return {
        'total_words': len(all_words),
        'unique_words': len(unique_words),
        'ttr': ttr
    }

# Hitung keragaman leksikal
ai_lex = lexical_diversity(ai_texts)
human_lex = lexical_diversity(human_texts)

print(f"\nStatistik Leksikal:")
print(f"  {'':<15} {'Total Kata':<12} {'Kata Unik':<12} {'TTR':<10}")
print(f"  {'-'*50}")
print(f"  {'AI':<15} {ai_lex['total_words']:<12} {ai_lex['unique_words']:<12} {ai_lex['ttr']:<10.4f}")
print(f"  {'Manusia':<15} {human_lex['total_words']:<12} {human_lex['unique_words']:<12} {human_lex['ttr']:<10.4f}")

print(f"\nInterpretasi TTR:")
print(f"  AI:      {ai_lex['ttr']*100:.2f}% (semakin tinggi = semakin beragam)")
print(f"  Manusia: {human_lex['ttr']*100:.2f}%")

# Analisis 3: Panjang kata rata-rata
print("\n" + "="*70)
print("3. PANJANG KATA RATA-RATA")

def avg_word_length(texts):
    """Hitung rata-rata panjang kata"""
    total_chars = 0
    total_words = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        total_chars += sum(len(word) for word in words)

    return total_chars / max(total_words, 1)

# Hitung rata-rata panjang kata
ai_avg_word_len = avg_word_length(ai_texts)
human_avg_word_len = avg_word_length(human_texts)

print(f"\nRata-rata panjang kata (karakter):")
print(f"  AI:      {ai_avg_word_len:.2f} karakter")
print(f"  Manusia: {human_avg_word_len:.2f} karakter")
print(f"  Selisih: {abs(ai_avg_word_len - human_avg_word_len):.2f} karakter")

# Analisis 4: Frekuensi tanda baca
print("\n" + "="*70)
print("4. FREKUENSI TANDA BACA")

def punctuation_analysis(texts):
    """Hitung frekuensi tanda baca"""
    punct_count = Counter()

    for text in texts:
        punct_count['.'] += text.count('.')
        punct_count[','] += text.count(',')
        punct_count['!'] += text.count('!')
        punct_count['?'] += text.count('?')
        punct_count[';'] += text.count(';')
        punct_count[':'] += text.count(':')

    return punct_count

# Hitung frekuensi tanda baca
ai_punct = punctuation_analysis(ai_texts)
human_punct = punctuation_analysis(human_texts)

print(f"\nFrekuensi tanda baca per 1000 kata:")
print(f"  {'Tanda':<10} {'AI':<12} {'Manusia':<12}")
print(f"  {'-'*35}")

for punct in ['.', ',', '!', '?', ';', ':']:
    ai_rate = (ai_punct[punct] / ai_lex['total_words']) * 1000
    human_rate = (human_punct[punct] / human_lex['total_words']) * 1000
    print(f"  {punct:<10} {ai_rate:<12.2f} {human_rate:<12.2f}")

# Ringkasan perbedaan
print("\n" + "="*70)
print("RINGKASAN PERBEDAAN LINGUISTIK")

# Buat ringkasan
summary = f"""
Perbedaan Utama Teks AI vs Manusia:

1. Panjang Kalimat:
   - AI:      {ai_avg_sent_len:.2f} kata/kalimat
   - Manusia: {human_avg_sent_len:.2f} kata/kalimat
   - {"AI lebih panjang" if ai_avg_sent_len > human_avg_sent_len else "Manusia lebih panjang"}

2. Keragaman Leksikal (TTR):
   - AI:      {ai_lex['ttr']*100:.2f}%
   - Manusia: {human_lex['ttr']*100:.2f}%
   - {"AI lebih beragam" if ai_lex['ttr'] > human_lex['ttr'] else "Manusia lebih beragam"}

3. Panjang Kata:
   - AI:      {ai_avg_word_len:.2f} karakter
   - Manusia: {human_avg_word_len:.2f} karakter
"""

print(summary)
