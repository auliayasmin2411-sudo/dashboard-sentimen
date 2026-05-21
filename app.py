import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Tweet Perfect Crown",
    page_icon="👑",
    layout="wide",
)

st.title("👑 Analisis Sentimen Tweet Perfect Crown")
st.markdown("Aplikasi analisis sentimen tweet menggunakan **TF-IDF + SVM** dengan preprocessing Bahasa Indonesia.")
st.divider()

# ─────────────────────────────────────────────
# Load Sastrawi
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat library NLP (Sastrawi)...")
def load_sastrawi():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "PySastrawi", "-q"], check=True)
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer    = StemmerFactory().create_stemmer()
    default_sw = set(StopWordRemoverFactory().get_stop_words())
    return stemmer, default_sw

stemmer, default_stopwords = load_sastrawi()

# ─────────────────────────────────────────────
# Kamus translate  — IDENTIK notebook
# ─────────────────────────────────────────────
translate_dict = {
    "love": "suka",
    "happy": "senang",
    "sad": "sedih",
    "support": "dukung",
    "amazing": "keren",
    "good": "bagus",
    "great": "hebat",
    "best": "terbaik",
    "nice": "bagus",
    "boring": "bosan",
    "comfort": "nyaman",
    "enjoy": "nikmat",
    "favorite": "favorit",
    "mood": "suasana",
    "beautiful": "indah",
    "wonderful": "luar biasa",
    "awesome": "keren",
    "stress": "stres",
    "scene": "adegan",
    "ending": "akhir",
    "plot": "alur",
    "episode": "episode",
    "watch": "tonton",
    "watching": "menonton",
    "move on": "lupa",
    "recommend": "rekomendasi",
}

# ─────────────────────────────────────────────
# Kamus slang extra  — IDENTIK notebook
# ─────────────────────────────────────────────
extra_slang = {
    "drakor"   : "drama",
    "kpop"     : "k-pop",
    "recomen"  : "rekomendasi",
    "rekomen"  : "rekomendasi",
    "nonton"   : "tonton",
    "nontonin" : "tonton",
    "baper"    : "bawa perasaan",
    "gapapa"   : "tidak apa",
    "gabisa"   : "tidak bisa",
    "kalo"     : "kalau",
    "klau"     : "kalau",
    "knp"      : "kenapa",
    "gmn"      : "bagaimana",
    "gmna"     : "bagaimana",
    "gimana"   : "bagaimana",
    "mulu"     : "selalu",
    "udah"     : "sudah",
    "udh"      : "sudah",
    "belom"    : "belum",
    "blm"      : "belum",
    "bgt"      : "sangat",
    "banget"   : "sangat",
    "keren"    : "bagus",
    "mantep"   : "mantap",
    "asek"     : "asyik",
    "asik"     : "asyik",
    "anjir"    : "astaga",
    "anjay"    : "astaga",
    "anjing"   : "astaga",
    "anjink"   : "astaga",
    "anjiaank" : "astaga",
    "anjinkk"  : "astaga",
    "anying"   : "astaga",
    "brengsek" : "jahat",
    "kampret"  : "jahat",
    "keparat"  : "jahat",
    "bajingan" : "jahat",
    "gilaa"    : "gila",
    "gilak"    : "gila",
    "ep"       : "episode",
    "eps"      : "episode",
    "kdrama"   : "drama",
}

# ─────────────────────────────────────────────
# Stopwords  — IDENTIK notebook (copy paste)
# ─────────────────────────────────────────────
english_stopwords = {
    "drama", "episode", "season", "series", "movie", "film",
    "ian", "byeon", "wooseok", "huiju",
    "ongoing", "sold", "last", "time", "land", "gold",
    "rewatch", "filing", "trying", "going", "salting", "acting",
    "user", "hapy", "lovely", "timer", "end", "eps",
    "scarecrow", "marathon", "romcom", "netflix", "weekend",
    "wonderfol", "mod", "sat", "bws", "btw", "the", "for", "you", "out",
}

extra_stopwords = {
    # Tawa / ekspresi
    "wkwk", "wkwkwk", "wkwkwkwk", "wkwkwkwkwk",
    "haha", "hahaha", "hahahaha", "hehe", "hihi", "hihihi",
    "xixi", "kwkw", "kwkwkw", "huhu", "wkwkwk", "wkwkw",
    # Kata filler
    "nih", "sih", "deh", "dong", "lah", "kak", "ko", "yah", "nah",
    "iya", "iyaa", "iyah", "emang", "emg", "memang",
    "tuh", "tau", "kah", "jir", "oke", "aduh", "argh",
    # Pronoun
    "aku", "kamu", "dia", "mereka", "gue", "gua", "lo", "lu",
    "kita", "kami",
    # Kata umum noise
    "kayak", "kayaknya", "kaya", "pas", "pakai",
    "yg", "yang", "aja", "juga", "tapi", "terus", "trus",
    "udah", "udh", "bgt", "banget", "gt", "gitu",
    "ga", "gak", "nggak", "engga", "enggak", "kagak",
    "mah", "weh", "woi", "hai", "hei", "hoi",
    "bikin", "guys", "main", "lupa", "plot", "pls", "nonton",
    "tonton", "nya", "aja", "banget", "sama", "mau", "jadi", "apa", "kalau",
    "dulu", "buat", "kali", "orang", "semua", "tiap", "sekarang", "lanjut", "habis",
    "tamat", "selesai", "akhir", "banyak", "pengin", "sangat", "karena", "beberapa", "terus", "mau",
    "nanti", "lanjutkan", "abistu", "apaan", "mbak", "orgorg", "malah", "lihat", "langsung",
    "mulai", "cuma", "kan", "soal", "padahal", "selalu",
    "sekali", "scene", "cerita", "ending", "raja", "royal",
    "nemesis", "kitchen", "soldier", "yumi", "cell", "cells", "filling",
    "love", "disney", "wtb", "wts", "kok", "phantom", "lawyer",
    # Waktu
    "malam", "pagi", "siang", "sore",
    "jumat", "sabtu", "minggu", "senin", "selasa", "rabu", "kamis",
    "hari", "jam", "tahun", "bulan", "kemarin", "besok",
    "baru", "bareng", "lagi", "barang", "kata",
    # Twitter noise
    "rt", "via", "amp", "ff", "mf", "cc",
}

STOPWORDS = default_stopwords | english_stopwords | extra_stopwords

# ─────────────────────────────────────────────
# Lexicon labels  — IDENTIK notebook
# ─────────────────────────────────────────────
extra_pos = {
    "seru", "keren", "bagus", "mantap", "menarik",
    "rekomendasi", "favorit", "love", "suka", "terbaik",
    "hebat", "kagum", "salut", "bangga", "terharu",
    "support", "dukung", "worth",
    "good", "great", "nice", "best", "amazing",
    "beautiful", "wonderful", "awesome", "enjoy",
    "gemas", "mood", "comfort",
}

override_neutral = {
    "sedih", "nangis", "stop", "tunggu", "lanjut",
    "mari", "biar", "cepat", "move", "perfect",
    "crown", "gampang", "drama", "buka", "kasih", "kelar",
    "fast", "response", "place", "became",
}

extra_neg = {
    "sedih", "kecewa", "jelek", "buruk",
    "hampa", "bosan", "bosen",
    "capek", "marah", "kesal",
    "bingung", "nangis",
    "overrated", "lebay",
    "gagal", "stress",
    "sinting",
}

# ─────────────────────────────────────────────
# Preprocessing functions  — IDENTIK notebook
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text.strip()

def translate_english(text):
    if not isinstance(text, str):
        return ""
    words = text.split()
    translated = []
    for word in words:
        translated.append(translate_dict.get(word.lower(), word))
    return " ".join(translated)

@st.cache_data(show_spinner="Memuat kamus alay dari GitHub...")
def load_normalization_dict():
    url = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"
    kamus_norm = pd.read_csv(url)
    normalisasi = dict(zip(kamus_norm['slang'].str.lower(), kamus_norm['formal'].str.lower()))
    normalisasi.update(extra_slang)
    return normalisasi

def normalize(text, normalisasi):
    words = text.split()
    return ' '.join([normalisasi.get(word, word) for word in words])

def preprocess_text(text, normalisasi):
    """Stopword removal — dipanggil pada kolom normalisasi, identik notebook."""
    if not isinstance(text, str):
        return []
    tokens   = text.split()
    filtered = [word for word in tokens
                if word not in STOPWORDS
                and len(word) > 2]
    return filtered

def stemming(tokens):
    return " ".join([stemmer.stem(word) for word in tokens])

@st.cache_data(show_spinner="Memuat lexicon InSet...")
def load_lexicon():
    pos_df = pd.read_csv(
        "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
        sep="\t", header=None, names=["word", "weight"]
    )
    neg_df = pd.read_csv(
        "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv",
        sep="\t", header=None, names=["word", "weight"]
    )
    pos_df['sentiment'] = 1
    neg_df['sentiment'] = -1
    lexicon = dict(zip(
        pd.concat([pos_df, neg_df])["word"].str.lower(),
        pd.concat([pos_df, neg_df])["sentiment"]
    ))
    for w in extra_neg:
        lexicon[w] = -1
    for w in extra_pos:
        lexicon[w] = 1
    for w in override_neutral:
        lexicon[w] = 0
    return lexicon

def labeling(text, lexicon):
    if not isinstance(text, str):
        return 0
    score = sum(lexicon.get(word, 0) for word in text.split())
    if score >= 1:
        return 1
    elif score <= -1:
        return -1
    else:
        return 0

# ─────────────────────────────────────────────
# Full Pipeline  — urutan & fungsi IDENTIK notebook
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Menjalankan preprocessing pipeline...")
def run_pipeline(df_raw, _normalisasi, _lexicon):
    df = df_raw[['Tweet']].rename(columns={'Tweet': 'tweet'}).copy()

    # 1. Cleaning
    df['clean'] = df['tweet'].apply(clean_text)

    # 2. Translate EN → ID
    df['translated'] = df['clean'].apply(translate_english)

    # 3. Normalisasi alay
    df['normalisasi'] = df['translated'].apply(lambda x: normalize(x, _normalisasi))

    # 4. Tokenisasi (split biasa, sama seperti notebook)
    df['token'] = df['normalisasi'].apply(lambda x: x.split())

    # 5. Stopword removal  ← dari kolom normalisasi (bukan token)
    df['stopwords'] = df['normalisasi'].apply(lambda x: preprocess_text(x, _normalisasi))

    # 6. Stemming
    df['stemming'] = df['stopwords'].apply(stemming)

    # 7. Labelling
    df['label'] = df['stemming'].apply(lambda x: labeling(x, _lexicon))

    label_map = {1: "Positif", -1: "Negatif", 0: "Netral"}
    df['label_nama'] = df['label'].map(label_map)
    return df

# ─────────────────────────────────────────────
# Sidebar — Dataset
# ─────────────────────────────────────────────
DEFAULT_FILE = "perfect_crown_tweets_rapih.xlsx"
has_default  = os.path.exists(DEFAULT_FILE)

with st.sidebar:
    st.header("📂 Dataset")

    if has_default:
        st.success(f"✅ Dataset default:\n`{DEFAULT_FILE}`")
        st.markdown("Upload file lain di bawah untuk menggantinya (opsional).")
    else:
        st.info(
            f"**Cara pakai dataset default:**\n"
            f"Rename file dataset kamu → `{DEFAULT_FILE}` "
            "lalu taruh di folder yang sama dengan `app.py` di repo GitHub."
        )

    uploaded = st.file_uploader(
        "Upload dataset" + (" (opsional)" if has_default else " *wajib*"),
        type=["csv", "xlsx", "xls"],
    )
    st.markdown("---")
    st.caption("Kolom wajib: **Tweet**  |  Format: `.csv` `.xlsx` `.xls`")

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
df_raw = None

if uploaded is not None:
    fname  = uploaded.name.lower()
    df_raw = pd.read_csv(uploaded) if fname.endswith(".csv") else pd.read_excel(uploaded)
    st.toast("File berhasil dibaca!", icon="✅")
elif has_default:
    ext    = DEFAULT_FILE.rsplit(".", 1)[-1].lower()
    df_raw = pd.read_csv(DEFAULT_FILE) if ext == "csv" else pd.read_excel(DEFAULT_FILE)
else:
    st.warning("👈 Upload file dataset di sidebar untuk memulai analisis.")
    st.stop()

if "Tweet" not in df_raw.columns:
    st.error("❌ Kolom **Tweet** tidak ditemukan. Pastikan nama kolom benar (huruf kapital T).")
    st.stop()

normalisasi_dict = load_normalization_dict()
lexicon          = load_lexicon()
df               = run_pipeline(df_raw, normalisasi_dict, lexicon)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Preprocessing",
    "☁️ Visualisasi Teks",
    "🤖 Modelling SVM",
    "🔍 Prediksi Teks",
])

# ═══════════════════════════════════════════
# TAB 1 — EDA & Preprocessing
# ═══════════════════════════════════════════
with tab1:
    st.subheader("Dataset Awal")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.write(f"**Total data:** {len(df_raw):,} baris")

    st.subheader("Hasil Preprocessing")
    st.dataframe(
        df[["tweet", "clean", "translated", "normalisasi", "stemming", "label_nama"]].head(20),
        use_container_width=True,
    )

    st.subheader("Distribusi Label Sentimen")
    col1, col2, col3 = st.columns(3)
    counts = df['label_nama'].value_counts()
    col1.metric("✅ Positif", counts.get("Positif", 0))
    col2.metric("➖ Netral",  counts.get("Netral",  0))
    col3.metric("❌ Negatif", counts.get("Negatif", 0))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        x=df["label_nama"],
        order=["Positif", "Netral", "Negatif"],
        palette=["#2ecc71", "#95a5a6", "#e74c3c"],
        ax=ax,
    )
    ax.set_title("Distribusi Label Sentimen", fontsize=14)
    ax.set_xlabel("Label"); ax.set_ylabel("Jumlah")
    st.pyplot(fig, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — Visualisasi Teks
# ═══════════════════════════════════════════
with tab2:
    all_words = ' '.join(df['stemming']).split()

    st.subheader("☁️ Word Cloud Keseluruhan")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc); ax.axis('off'); ax.set_title("Word Cloud Tweet Perfect Crown")
    st.pyplot(fig, use_container_width=True)

    st.subheader("📊 Top 20 Kata Terbanyak")
    word_freq  = Counter(all_words)
    top_words  = word_freq.most_common(20)
    words_list, freqs = zip(*top_words)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(words_list, freqs, color='steelblue')
    ax.set_title("Top 20 Kata Terbanyak"); ax.set_xlabel("Kata"); ax.set_ylabel("Frekuensi")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("📊 Top 20 Trigram")
    bigrams   = list(ngrams(all_words, 3))
    bigram_freq = Counter(bigrams)
    top_bigrams = bigram_freq.most_common(20)
    bg_labels, bg_freqs = zip(*[(' '.join(b), f) for b, f in top_bigrams])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(bg_labels, bg_freqs, color='darkorange')
    ax.set_title("Top 20 Trigram Terbanyak"); ax.set_xlabel("Trigram"); ax.set_ylabel("Frekuensi")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("☁️ Word Cloud per Kelas Sentimen")
    positif_text = " ".join(df[df['label'] == 1]['stemming'])
    netral_text  = " ".join(df[df['label'] == 0]['stemming'])
    negatif_text = " ".join(df[df['label'] == -1]['stemming'])

    c1, c2, c3 = st.columns(3)
    for col_ui, text, title in [
        (c1, positif_text, "Positif"),
        (c2, netral_text,  "Netral"),
        (c3, negatif_text, "Negatif"),
    ]:
        if text.strip():
            wc = WordCloud(width=1200, height=600, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            ax.set_title(f"WordCloud Sentimen {title}", fontsize=12)
            col_ui.pyplot(fig, use_container_width=True)
        else:
            col_ui.info(f"Tidak ada data {title}.")

# ═══════════════════════════════════════════
# TAB 3 — Modelling SVM
# ═══════════════════════════════════════════
with tab3:
    st.subheader("⚙️ Konfigurasi Model")

    col_a, col_b = st.columns(2)
    test_size    = col_a.slider("Ukuran Test Set (%)", 10, 40, 20, 5) / 100
    max_features = col_b.number_input("Max Fitur TF-IDF", 1000, 100000, 50000, 1000)
    use_grid     = st.checkbox("Gunakan GridSearchCV (lebih akurat, lebih lambat)", value=False)

    if st.button("🚀 Latih Model SVM", type="primary"):
        with st.spinner("Melatih model... harap tunggu"):
            X = df['stemming']
            y = df['label_nama']

            tfidf   = TfidfVectorizer(max_features=int(max_features))
            X_tfidf = tfidf.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=test_size, random_state=42, stratify=y
            )
            st.write(f"**Train:** {X_train.shape[0]} | **Test:** {X_test.shape[0]}")
            st.write("Distribusi data train:")
            st.write(y_train.value_counts().to_dict())

            if use_grid:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                }
                gs = GridSearchCV(
                    SVC(class_weight='balanced'), param_grid,
                    cv=5, scoring='f1_macro', verbose=0, n_jobs=-1
                )
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                st.success(f"Best params: {gs.best_params_}")
            else:
                model = SVC(C=1, kernel='linear', gamma='scale', class_weight='balanced')
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)

            st.subheader("📈 Hasil Evaluasi")
            st.metric("Accuracy", f"{round(acc, 4)}")

            cr = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(cr).transpose().round(4), use_container_width=True)

            # Confusion Matrix
            st.subheader("🟦 Confusion Matrix")
            classes_order = ['Negatif', 'Netral', 'Positif']
            cm = confusion_matrix(y_test, y_pred, labels=classes_order)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                        linewidths=1, linecolor='black',
                        xticklabels=classes_order, yticklabels=classes_order,
                        cbar=True, ax=ax)
            ax.set_title('Confusion Matrix Model SVM', fontsize=18, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=13)
            ax.set_ylabel('Actual Label', fontsize=13)
            plt.xticks(fontsize=11); plt.yticks(fontsize=11)
            st.pyplot(fig, use_container_width=True)

            # ROC-AUC
            st.subheader("📉 AUC-ROC Curve")
            y_test_bin = label_binarize(y_test, classes=classes_order)
            y_score    = model.decision_function(X_test)
            colors_roc = ['#E74C3C', '#F39C12', '#2ECC71']
            fig, ax    = plt.subplots(figsize=(8, 6))
            auc_scores = []
            for i in range(len(classes_order)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc     = auc(fpr, tpr)
                auc_scores.append(roc_auc)
                ax.plot(fpr, tpr, color=colors_roc[i], lw=2.5,
                        label=f'Kelas {classes_order[i]} (AUC = {roc_auc:.3f})')
            macro_auc = np.mean(auc_scores)
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.500)')
            ax.plot([], [], ' ', label=f'Macro-Average AUC = {macro_auc:.3f}')
            ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=13)
            ax.set_ylabel('True Positive Rate', fontsize=13)
            ax.set_title('AUC-ROC Curve Model SVM', fontsize=18, fontweight='bold')
            ax.legend(loc='lower right', fontsize=11); ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.session_state['model'] = model
            st.session_state['tfidf'] = tfidf

    elif 'model' not in st.session_state:
        st.info("Klik tombol **Latih Model SVM** untuk memulai pelatihan.")

# ═══════════════════════════════════════════
# TAB 4 — Prediksi Teks Baru
# ═══════════════════════════════════════════
with tab4:
    st.subheader("🔍 Prediksi Sentimen Teks Baru")

    if 'model' not in st.session_state:
        st.warning("⚠️ Latih model terlebih dahulu di tab **Modelling SVM**.")
    else:
        user_input = st.text_area(
            "Masukkan teks tweet:",
            height=120,
            placeholder="Contoh: Drama ini bagus banget, ceritanya seru dan aktingnya keren!",
        )
        if st.button("Prediksi", type="primary") and user_input.strip():
            nd = load_normalization_dict()
            lx = load_lexicon()

            # Preprocessing persis pipeline notebook
            step1_clean      = clean_text(user_input)
            step2_translated = translate_english(step1_clean)
            step3_norm       = normalize(step2_translated, nd)
            step4_tokens     = step3_norm.split()                         # tokenisasi = split
            step5_sw         = preprocess_text(step3_norm, nd)            # stopword dari normalisasi
            step6_stem       = stemming(step5_sw)                         # stemming dari list token

            X_input = st.session_state['tfidf'].transform([step6_stem])
            pred    = st.session_state['model'].predict(X_input)[0]

            color_map = {"Positif": "🟢", "Netral": "🟡", "Negatif": "🔴"}
            st.markdown(f"### Hasil Prediksi: {color_map.get(pred, '')} **{pred}**")

            with st.expander("🔎 Detail Preprocessing"):
                st.write(f"**1. Clean:**        {step1_clean}")
                st.write(f"**2. Translated:**   {step2_translated}")
                st.write(f"**3. Normalisasi:**  {step3_norm}")
                st.write(f"**4. Token:**        {step4_tokens}")
                st.write(f"**5. Stopwords:**    {step5_sw}")
                st.write(f"**6. Stemming:**     {step6_stem}")
