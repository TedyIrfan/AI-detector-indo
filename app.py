# =====================================================
# DETEKSI TULISAN AI vs MANUSIA
# Web Interface - UPDATED 2025
# =====================================================

import streamlit as st
import joblib
import numpy as np
import re
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Deteksi AI vs Manusia",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================================
# CSS - MINIMALIS (GRAY SCALE)
# =====================================================
st.markdown("""
<style>
    :root {
        --primary: #1f2937;
        --secondary: #6b7280;
        --border: #e5e7eb;
        --bg-light: #f9fafb;
    }

    .main-title {
        text-align: center;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--primary);
        margin: 0;
    }

    .subtitle {
        text-align: center;
        font-size: 0.875rem;
        color: var(--secondary);
        margin-bottom: 1.5rem;
    }

    .result-box {
        background: var(--bg-light);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-box h2 {
        color: var(--primary);
        font-size: 1.125rem;
        font-weight: 600;
        margin: 0.5rem 0 0 0;
    }

    .result-human {
        border-left: 3px solid #10b981;
    }

    .result-ai {
        border-left: 3px solid #ef4444;
    }

    .info-box {
        background: #1f2937;
        border: 1px solid #374151;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.875rem;
    }

    .model-info {
        background: var(--bg-light);
        border: 1px solid var(--border);
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.8rem;
    }

    .analysis-box {
        background: var(--bg-light);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.875rem;
        margin: 0.5rem 0;
    }

    .analysis-box h4 {
        color: var(--primary);
        margin: 0 0 0.5rem 0;
        font-size: 0.875rem;
        font-weight: 600;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.75rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary);
    }

    .metric-label {
        font-size: 0.7rem;
        color: var(--secondary);
    }

    .footer {
        text-align: center;
        color: var(--secondary);
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# ANALISIS TEKS
# =====================================================
def analyze_text_features(text):
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text)) - 1

    words = text.lower().split()
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words) if len(words) > 0 else 0

    formal_words = ['telah', 'tersebut', 'melalui', 'dalam', 'untuk', 'bagai', 'bagaimana',
                    'apabila', 'yakni', 'ialah', 'merupakan', 'diharapkan', 'implementasi',
                    'comprehensif', 'potensi', 'mengharumkan', 'kancah', 'internasional',
                    'mengimplementasikan', 'pembinaan', 'atlet', 'prestasi', 'meningkatkan',
                    'upaya', 'berbagai', 'program', 'kemas', 'berupaya', 'berdasarkan',
                    'mengenai', 'terkait', 'kesimpulannya', 'pentingnya', 'perlu', 'perlunya']

    informal_words = ['gue', 'elo', 'lu', 'gw', 'gak', 'nggak', 'ga', 'udah',
                      'nih', 'nihh', 'deh', 'dong', 'yuk', 'ayo', 'sih', 'loh', 'kok',
                      'kenapa', 'gimana', 'sampe', 'sampek', 'dah', 'dahh']

    formal_count = sum(1 for word in words if word in formal_words)
    informal_count = sum(1 for word in words if word in informal_words)

    if formal_count + informal_count > 0:
        formality_score = formal_count / (formal_count + informal_count)
    else:
        avg_word_length = sum(len(word) for word in words) / len(words) if len(words) > 0 else 0
        formality_score = min(avg_word_length / 8, 1.0)

    ai_patterns = {'perfect_structure': 0, 'generic_words': 0, 'no_personal': 0, 'repetitive_phrases': 0}

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) > 2:
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        ai_patterns['perfect_structure'] = 1 if length_variance < 10 else 0

    generic_ai_words = ['dalam', 'untuk', 'dengan', 'yang', 'dan', 'pada', 'secara',
                        'telah', 'dapat', 'bagai', 'bagaimana', 'penting', 'perlu']
    generic_ratio = sum(1 for word in words if word in generic_ai_words) / len(words) if len(words) > 0 else 0
    ai_patterns['generic_words'] = 1 if generic_ratio > 0.3 else 0

    personal_words = ['saya', 'aku', 'gue', 'gua', 'kami', 'kita', 'beta']
    has_personal = any(word in words for word in personal_words)
    ai_patterns['no_personal'] = 1 if not has_personal else 0

    if len(words) > 20:
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        unique_bigrams = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 1
        ai_patterns['repetitive_phrases'] = 1 if unique_bigrams < 0.5 else 0

    ai_pattern_score = sum(ai_patterns.values()) / len(ai_patterns)

    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'unique_ratio': unique_ratio,
        'formality_score': formality_score,
        'ai_pattern_score': ai_pattern_score,
        'ai_patterns': ai_patterns
    }

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model_and_vectorizer():
    try:
        base_path = Path("models")
        model_file = base_path / "random_forest_model.pkl"
        vectorizer_file = base_path / "tfidf_vectorizer.pkl"

        if not model_file.exists():
            st.error(f"Model file tidak ditemukan: {model_file}")
            return None, None

        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)

        # Load metadata
        metadata_file = base_path / "metadata.pkl"
        if metadata_file.exists():
            metadata = joblib.load(metadata_file)
            return model, vectorizer, metadata

        return model, vectorizer, None

    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_text(text, model, vectorizer):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    label = "MANUSIA" if prediction == 0 else "AI"
    confidence = probabilities[prediction] * 100
    return label, confidence, probabilities

# =====================================================
# SIDEBAR
# =====================================================
def sidebar():
    with st.sidebar:
        st.markdown("### 🤖 Model Info")
        st.markdown("---")

        st.markdown("""
        <div class="model-info">
        <strong>Random Forest</strong><br>
        Ensemble learning<br>
        Akurasi: 98.68%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Dataset")
        st.markdown(f"**Total:** 1,510 teks")
        st.markdown(f"**AI:** 760 (50.3%)")
        st.markdown(f"**Manusia:** 750 (49.7%)")

        st.markdown("---")
        st.markdown("### 🔧 Metode")
        st.markdown("**Vectorization:** TF-IDF")
        st.markdown("**Features:** 5,000")
        st.markdown("**Split:** 80% Train / 20% Test")

        st.markdown("---")
        st.markdown("### 📁 Sumber AI")
        st.markdown("- OpenRouter (3 model)")
        st.markdown("- Groq (1 model)")
        st.markdown("- HuggingFace (2 model)")

# =====================================================
# MAIN
# =====================================================
def main():
    sidebar()

    # Header
    st.markdown('<h1 class="main-title">Deteksi Tulisan AI vs Manusia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Updated 2025 - Dataset 1510 | Akurasi 98.68%</p>', unsafe_allow_html=True)

    # Load model
    model, vectorizer, metadata = load_model_and_vectorizer()
    if model is None or vectorizer is None:
        st.error("❌ Model tidak dapat dimuat. Pastikan folder 'models/' sudah ada.")
        st.stop()

    st.info("✅ Model berhasil dimuat!")

    st.markdown("""
    <div class="info-box">
        📝 Tempelkan teks yang ingin dianalisis, lalu klik "Analisis Teks".
    </div>
    """, unsafe_allow_html=True)

    # Input
    st.markdown("### Input Teks")
    user_input = st.text_area(
        "",
        height=120,
        placeholder="Masukkan teks di sini...",
        label_visibility="collapsed"
    )

    # Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analisis Teks", type="primary", use_container_width=True):
            if not user_input or len(user_input.strip()) < 10:
                st.warning("⚠️ Mohon masukkan teks minimal 10 karakter")
            else:
                with st.spinner("Menganalisis..."):
                    label, confidence, probabilities = predict_text(user_input, model, vectorizer)
                    features = analyze_text_features(user_input)

                # Result
                st.markdown("---")
                st.markdown("### 🎯 Hasil Analisis")

                if label == "MANUSIA":
                    st.markdown(f"""
                    <div class="result-box result-human">
                        <h2>✍️ Terdeteksi: TULISAN MANUSIA</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-ai">
                        <h2>🤖 Terdeteksi: TULISAN AI</h2>
                    </div>
                    """, unsafe_allow_html=True)

                prob_human = probabilities[0] * 100
                prob_ai = probabilities[1] * 100

                st.markdown(f"**Keyakinan:** {confidence:.1f}%")

                if label == "MANUSIA":
                    st.progress(probabilities[0])
                    st.caption(f"📊 MANUSIA: {prob_human:.1f}% | 🤖 AI: {prob_ai:.1f}%")
                else:
                    st.progress(probabilities[1])
                    st.caption(f"🤖 AI: {prob_ai:.1f}% | 📝 MANUSIA: {prob_human:.1f}%")

                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📝 Manusia", f"{prob_human:.1f}%")
                with col_b:
                    st.metric("🤖 AI", f"{prob_ai:.1f}%")

                # Analysis
                st.markdown("---")
                st.markdown("### 📈 Analisis Fitur")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['char_count']}</div>
                        <div class="metric-label">Karakter</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['word_count']}</div>
                        <div class="metric-label">Kata</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['sentence_count']}</div>
                        <div class="metric-label">Kalimat</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features['unique_ratio']*100:.0f}%</div>
                        <div class="metric-label">Unik</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Details
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                    st.markdown("#### 📊 Statistik")
                    st.markdown(f"- Kata/kalimat: {features['word_count']/max(features['sentence_count'], 1):.1f}")
                    st.markdown(f"- Karakter/kata: {features['char_count']/max(features['word_count'], 1):.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                    st.markdown("#### 📝 Formalitas")
                    formality_pct = features['formality_score'] * 100
                    st.markdown(f"- Tingkat: {formality_pct:.0f}%")
                    st.progress(features['formality_score'])
                    if formality_pct > 70:
                        st.caption("Sangat formal")
                    elif formality_pct > 40:
                        st.caption("Cukup formal")
                    else:
                        st.caption("Informal")
                    st.markdown('</div>', unsafe_allow_html=True)

                # AI Pattern
                st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                st.markdown("#### 🤖 Pola AI")

                ai_pct = features['ai_pattern_score'] * 100
                st.progress(features['ai_pattern_score'])
                st.markdown(f"**Skor Pola AI:** {ai_pct:.0f}%")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("- Struktur rapi" if features['ai_patterns']['perfect_structure'] else "- Struktur variatif")
                    st.markdown("- Kata generik" if features['ai_patterns']['generic_words'] else "- Kosakata beragam")

                with col_b:
                    st.markdown("- Tanpa personal" if features['ai_patterns']['no_personal'] else "- Ada personal")
                    st.markdown("- Frase berulang" if features['ai_patterns']['repetitive_phrases'] else "- Frase variatif")

                if ai_pct > 60:
                    st.warning("⚠️ Teks memiliki pola mirip AI")
                elif ai_pct > 30:
                    st.info("ℹ️ Beberapa pola AI terdeteksi")
                else:
                    st.success("✅ Teks memiliki ciri khas manusia")

                st.markdown('</div>', unsafe_allow_html=True)

    # Example
    with st.expander("💡 Contoh Teks"):
        st.markdown("**📝 Manusia:**")
        st.code("""Jakarta, CNN Indonesia - Timnas Indonesia berhasil meraih kemenangan
dramatis melawan Thailand dengan skor 3-2 dalam laga persahabatan yang
digelar di Stadion Utama Gelora Bung Karno, Senin (15/1).""")

        st.markdown("**🤖 AI:**")
        st.code("""Dalam upaya meningkatkan prestasi olahraga nasional, pemerintah
telah mengimplementasikan berbagai program pembinaan atlet yang komprehensif.
Hal ini diharapkan dapat mencetak talenta-talenta muda yang berpotensi.""")

    # Footer
    st.markdown(f"""
    <div class="footer">
        <p>Streamlit | Random Forest | TF-IDF</p>
        <p>Dataset: 1510 (760 AI + 750 Manusia) | Akurasi: 98.68%</p>
        <p>Updated 2025 - Deteksi Tulisan AI vs Manusia</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
