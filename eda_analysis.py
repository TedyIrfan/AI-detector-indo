"""
Step 2: Exploratory Data Analysis (EDA)
Visualisasi dan analisis dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# Load data
print("\n[1] Load dataset...")
df = pd.read_csv("dataset_final_1500.csv", encoding='utf-8')
print(f"     Shape: {df.shape}")
print(f"     Columns: {list(df.columns)}")

# Info dasar
print("\n[2] Dataset Info...")
print(df.info())

# Missing values
print("\n[3] Missing Values...")
missing = df.isnull().sum()
print(missing)
print(f"     Total missing: {missing.sum()}")

# Distribusi Label
print("\n[4] Label Distribution...")
label_counts = df['label'].value_counts()
print(label_counts)

plt.figure(figsize=(8, 6))
ax = label_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('Distribusi Label (AI vs Manusia)', fontsize=14, fontweight='bold')
plt.xlabel('Label', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.xticks(rotation=0)
# Add value labels
for i, v in enumerate(label_counts.values):
    ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: eda_label_distribution.png")

# Pie chart
plt.figure(figsize=(8, 8))
colors = ['#3498db', '#e74c3c']
plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Persentase AI vs Manusia', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('eda_label_pie.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: eda_label_pie.png")

# Text Length Analysis
print("\n[5] Text Length Analysis...")
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"     Mean text length: {df['text_length'].mean():.2f}")
print(f"     Mean word count: {df['word_count'].mean():.2f}")
print(f"     Min text length: {df['text_length'].min()}")
print(f"     Max text length: {df['text_length'].max()}")

# Text length by label
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for i, label in enumerate(['MANUSIA', 'AI']):
    data = df[df['label'] == label]['text_length']
    axes[i].hist(data, bins=30, alpha=0.7, color=['#3498db', '#e74c3c'][i])
    axes[i].set_title(f'Distribusi Panjang Teks - {label}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Jumlah Karakter', fontsize=10)
    axes[i].set_ylabel('Frekuensi', fontsize=10)

plt.tight_layout()
plt.savefig('eda_text_length_by_label.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: eda_text_length_by_label.png")

# Word count by label
plt.figure(figsize=(10, 6))
for label, color in [('MANUSIA', '#3498db'), ('AI', '#e74c3c')]:
    data = df[df['label'] == label]['word_count']
    plt.hist(data, bins=30, alpha=0.5, label=label, color=color)
plt.title('Distribusi Jumlah Kata per Label', fontsize=14, fontweight='bold')
plt.xlabel('Jumlah Kata', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('eda_word_count_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: eda_word_count_distribution.png")

# Word Cloud
print("\n[6] Word Cloud Analysis...")
try:
    # Word Cloud untuk Manusia
    text_manusia = ' '.join(df[df['label'] == 'MANUSIA']['text'].tolist())
    wordcloud_manusia = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Blues').generate(text_manusia)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_manusia, interpolation='bilinear')
    plt.title('Word Cloud - Teks Manusia', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('eda_wordcloud_manusia.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     Saved: eda_wordcloud_manusia.png")

    # Word Cloud untuk AI
    text_ai = ' '.join(df[df['label'] == 'AI']['text'].tolist())
    wordcloud_ai = WordCloud(width=800, height=400, background_color='white',
                            colormap='Reds').generate(text_ai)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_ai, interpolation='bilinear')
    plt.title('Word Cloud - Teks AI', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('eda_wordcloud_ai.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     Saved: eda_wordcloud_ai.png")

except Exception as e:
    print(f"     WordCloud error (mungkin perlu install): {e}")

# Top Words Analysis
print("\n[7] Top Words Analysis...")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_top_words(texts, n=20):
    all_words = []
    for text in texts:
        words = clean_text(text).split()
        all_words.extend([w for w in words if len(w) > 2])
    return Counter(all_words).most_common(n)

# Top words Manusia
top_manusia = get_top_words(df[df['label'] == 'MANUSIA']['text'])
print("\n     Top 20 Words - MANUSIA:")
for word, count in top_manusia:
    print(f"       {word:<15} {count}")

# Top words AI
top_ai = get_top_words(df[df['label'] == 'AI']['text'])
print("\n     Top 20 Words - AI:")
for word, count in top_ai:
    print(f"       {word:<15} {count}")

# Visualisasi top words
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Manusia
words_manusia, counts_manusia = zip(*top_manusia)
axes[0].barh(words_manusia, counts_manusia, color='#3498db')
axes[0].set_title('Top 20 Words - Manusia', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Frekuensi', fontsize=10)
axes[0].invert_yaxis()

# AI
words_ai, counts_ai = zip(*top_ai)
axes[1].barh(words_ai, counts_ai, color='#e74c3c')
axes[1].set_title('Top 20 Words - AI', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Frekuensi', fontsize=10)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('eda_top_words_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: eda_top_words_comparison.png")

# Correlation Analysis (if applicable)
print("\n[8] Statistical Summary...")
print(df.describe())

print("\n" + "="*60)
print("EDA SELESAI!")
print("="*60)
print("\nFile yang dihasilkan:")
print("  - eda_label_distribution.png")
print("  - eda_label_pie.png")
print("  - eda_text_length_by_label.png")
print("  - eda_word_count_distribution.png")
print("  - eda_wordcloud_manusia.png")
print("  - eda_wordcloud_ai.png")
print("  - eda_top_words_comparison.png")
