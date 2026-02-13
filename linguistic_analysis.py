"""
Analisis Linguistik Perbedaan AI vs Manusia
Studi perbandingan karakteristik linguistik teks AI dan manusia
"""

import pandas as pd
import numpy as np
import joblib
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ANALISIS LINGUISTIK - PERBEDAAN AI vs MANUSIA")
print("="*70)

# Load dataset
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
df = df.dropna()

# Split texts
ai_texts = df[df['label'] == 'AI']['text'].tolist()
human_texts = df[df['label'] == 'MANUSIA']['text'].tolist()

print(f"     AI texts: {len(ai_texts)}")
print(f"     Human texts: {len(human_texts)}")

# =====================================================
# 1. PANJANG KALIMAT RATA-RATA
# =====================================================
print("\n" + "="*70)
print("1. ANALISIS PANJANG KALIMAT")
print("="*70)

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

ai_avg_sent_len = avg_sentence_length(ai_texts)
human_avg_sent_len = avg_sentence_length(human_texts)

print(f"\nRata-rata panjang kalimat (dalam kata):")
print(f"  AI:      {ai_avg_sent_len:.2f} kata/kalimat")
print(f"  Manusia: {human_avg_sent_len:.2f} kata/kalimat")
print(f"  Selisih: {abs(ai_avg_sent_len - human_avg_sent_len):.2f} kata")

# =====================================================
# 2. KERAGAMAN LEKSIKAL (VOCABULARY RICHNESS)
# =====================================================
print("\n" + "="*70)
print("2. KERAGAMAN LEKSIKAL (LEXICAL DIVERSITY)")
print("="*70)

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

# =====================================================
# 3. PANJANG KATA RATA-RATA
# =====================================================
print("\n" + "="*70)
print("3. PANJANG KATA RATA-RATA")
print("="*70)

def avg_word_length(texts):
    total_chars = 0
    total_words = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        total_chars += sum(len(word) for word in words)

    return total_chars / max(total_words, 1)

ai_avg_word_len = avg_word_length(ai_texts)
human_avg_word_len = avg_word_length(human_texts)

print(f"\nRata-rata panjang kata (karakter):")
print(f"  AI:      {ai_avg_word_len:.2f} karakter")
print(f"  Manusia: {human_avg_word_len:.2f} karakter")
print(f"  Selisih: {abs(ai_avg_word_len - human_avg_word_len):.2f} karakter")

# =====================================================
# 4. FREKUENSI TANDA BACA
# =====================================================
print("\n" + "="*70)
print("4. ANALISIS TANDA BACA")
print("="*70)

def punctuation_analysis(texts):
    punct_count = Counter()

    for text in texts:
        punct_count['.'] += text.count('.')
        punct_count[','] += text.count(',')
        punct_count['!'] += text.count('!')
        punct_count['?'] += text.count('?')
        punct_count[';'] += text.count(';')
        punct_count[':'] += text.count(':')

    return punct_count

ai_punct = punctuation_analysis(ai_texts)
human_punct = punctuation_analysis(human_texts)

print(f"\nFrekuensi tanda baca per 1000 kata:")
print(f"  {'Tanda':<10} {'AI':<12} {'Manusia':<12}")
print(f"  {'-'*35}")

for punct in ['.', ',', '!', '?', ';', ':']:
    ai_rate = (ai_punct[punct] / ai_lex['total_words']) * 1000
    human_rate = (human_punct[punct] / human_lex['total_words']) * 1000
    print(f"  {punct:<10} {ai_rate:<12.2f} {human_rate:<12.2f}")

# =====================================================
# 5. ANALISIS KATA GANTI (PRONOUNS)
# =====================================================
print("\n" + "="*70)
print("5. ANALISIS KATA GANTI (PRONOUNS)")
print("="*70)

first_person = ['saya', 'aku', 'gue', 'gua', 'beta', 'kami', 'kita']
second_person = ['kamu', 'anda', 'ente', 'kalian', 'engkau']
third_person = ['dia', 'beliau', 'mereka', 'mereka semua']

def count_pronouns(texts, pronoun_list):
    count = 0
    total_words = 0

    for text in texts:
        words = text.lower().split()
        total_words += len(words)
        count += sum(1 for word in words if word in pronoun_list)

    return (count / total_words * 100) if total_words > 0 else 0

print(f"\nFrekuensi kata ganti (per 100 kata):")
print(f"  {'Kategori':<20} {'AI':<12} {'Manusia':<12}")
print(f"  {'-'*45}")

ai_first = count_pronouns(ai_texts, first_person)
human_first = count_pronouns(human_texts, first_person)
print(f"  {'Orang Pertama':<20} {ai_first:<12.2f} {human_first:<12.2f}")

ai_second = count_pronouns(ai_texts, second_person)
human_second = count_pronouns(human_texts, second_person)
print(f"  {'Orang Kedua':<20} {ai_second:<12.2f} {human_second:<12.2f}")

ai_third = count_pronouns(ai_texts, third_person)
human_third = count_pronouns(human_texts, third_person)
print(f"  {'Orang Ketiga':<20} {ai_third:<12.2f} {human_third:<12.2f}")

# =====================================================
# 6. ANALISIS KATA FORMAL vs INFORMAL
# =====================================================
print("\n" + "="*70)
print("6. ANALISIS FORMAL vs INFORMAL")
print("="*70)

formal_words = ['telah', 'tersebut', 'melalui', 'dalam', 'untuk', 'bagai',
                'apabila', 'yakni', 'ialah', 'merupakan', 'diharapkan',
                'implementasi', 'pentingnya', 'perlunya', 'comprehensif']

informal_words = ['gue', 'elo', 'lu', 'gw', 'gak', 'nggak', 'ga', 'udah',
                  'nih', 'deh', 'dong', 'yuk', 'ayo', 'sih', 'loh', 'kok',
                  'kenapa', 'gimana', 'sampe', 'dah', 'banget']

def count_word_category(texts, word_list):
    count = 0
    total_words = 0

    for text in texts:
        words = text.lower().split()
        total_words += len(words)
        count += sum(1 for word in words if word in word_list)

    return (count / total_words * 100) if total_words > 0 else 0

ai_formal = count_word_category(ai_texts, formal_words)
human_formal = count_word_category(human_texts, formal_words)

ai_informal = count_word_category(ai_texts, informal_words)
human_informal = count_word_category(human_texts, informal_words)

print(f"\nFrekuensi kata formal/informal (per 100 kata):")
print(f"  {'Kategori':<15} {'AI':<12} {'Manusia':<12}")
print(f"  {'-'*40}")
print(f"  {'Formal':<15} {ai_formal:<12.2f} {human_formal:<12.2f}")
print(f"  {'Informal':<15} {ai_informal:<12.2f} {human_informal:<12.2f}")

# =====================================================
# 7. ANALISIS N-GRAM (KATA BERDAMPINGAN)
# =====================================================
print("\n" + "="*70)
print("7. ANALISIS BIGRAM (PASANGAN KATA)")
print("="*70)

def get_top_bigrams(texts, n=10):
    all_bigrams = []

    for text in texts:
        words = text.lower().split()
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        all_bigrams.extend(bigrams)

    return Counter(all_bigrams).most_common(n)

print(f"\nTop 10 Bigram - AI:")
ai_bigrams = get_top_bigrams(ai_texts)
for bigram, count in ai_bigrams:
    print(f"  - {bigram}: {count}")

print(f"\nTop 10 Bigram - Manusia:")
human_bigrams = get_top_bigrams(human_texts)
for bigram, count in human_bigrams:
    print(f"  - {bigram}: {count}")

# =====================================================
# 8. RATA-RATA PANJANG TEKS
# =====================================================
print("\n" + "="*70)
print("8. STATISTIK PANJANG TEKS")
print("="*70)

ai_lengths = [len(t) for t in ai_texts]
human_lengths = [len(t) for t in human_texts]

print(f"\nStatistik Panjang Teks (karakter):")
print(f"  {'Metrik':<20} {'AI':<15} {'Manusia':<15}")
print(f"  {'-'*50}")
print(f"  {'Mean':<20} {np.mean(ai_lengths):<15.1f} {np.mean(human_lengths):<15.1f}")
print(f"  {'Median':<20} {np.median(ai_lengths):<15.1f} {np.median(human_lengths):<15.1f}")
print(f"  {'Std Dev':<20} {np.std(ai_lengths):<15.1f} {np.std(human_lengths):<15.1f}")
print(f"  {'Min':<20} {np.min(ai_lengths):<15.0f} {np.min(human_lengths):<15.0f}")
print(f"  {'Max':<20} {np.max(ai_lengths):<15.0f} {np.max(human_lengths):<15.0f}")

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "="*70)
print("RINGKASAN PERBEDAAN LINGUISTIK")
print("="*70)

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

4. Panjang Teks:
   - AI:      rata-rata {np.mean(ai_lengths):.0f} karakter
   - Manusia: rata-rata {np.mean(human_lengths):.0f} karakter

5. Formalitas:
   - Kata formal:  AI {ai_formal:.2f}%, Manusia {human_formal:.2f}%
   - Kata informal: AI {ai_informal:.2f}%, Manusia {human_informal:.2f}%
"""

print(summary)

# Export hasil
print("\n" + "="*70)
print("EXPORT HASIL ANALISIS")
print("="*70)

results = {
    'metric': ['Avg Sentence Length', 'Lexical Diversity (TTR)', 'Avg Word Length',
               'Avg Text Length', 'Formal Words %', 'Informal Words %',
               'First Person Pronoun %', 'Second Person Pronoun %'],
    'AI': [ai_avg_sent_len, ai_lex['ttr'], ai_avg_word_len, np.mean(ai_lengths),
           ai_formal, ai_informal, ai_first, ai_second],
    'Human': [human_avg_sent_len, human_lex['ttr'], human_avg_word_len, np.mean(human_lengths),
              human_formal, human_informal, human_first, human_second]
}

results_df = pd.DataFrame(results)
results_df.to_csv('linguistic_analysis_results.csv', index=False, encoding='utf-8')
print("\n[OK] Export: linguistic_analysis_results.csv")

print("\n" + "="*70)
print("ANALISIS LINGUISTIK SELESAI!")
print("="*70)
