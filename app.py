# voice_insights_demo.py
"""
Voice-to-Insights Call-Center Demo  🎙️📝➡️📊
────────────────────────────────────────────────────────────────────────────
Upload a **single WAV / MP3** recording (≤ 5 min).  
The app will:

1. **Transcribe** audio locally with *faster-whisper-tiny* (CPU-friendly).  
2. Split transcript into sentences → embed with MiniLM.  
3. Run **k-means** (k=5 default) to cluster topics.  
4. Reduce embeddings to 2-D via **UMAP** → interactive scatter plot.  
5. Show a TF-IDF keyword list for each cluster.  
6. Let you download the full transcript + cluster labels as CSV.

This is a proof-of-concept. For enterprise speech analytics
(diarization, PII redaction, live streaming), visit **drtomharty.com/bio**.
"""
# ─────────────────────────────────── imports ──────────────────────────────
import os, io, tempfile, itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

# ─────────────────────────── cached loaders ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper():
    # tiny model (~75 MB) good enough for demo, runs on CPU
    return WhisperModel("tiny", device="cpu", compute_type="int8")

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ───────────────────────────── helpers ────────────────────────────────────
def transcribe(audio_path):
    model = load_whisper()
    segments, _ = model.transcribe(audio_path, beam_size=5)
    # concatenate words into sentences (simple '.', '?', '!' split)
    text = " ".join(seg.text for seg in segments)
    sentences = [s.strip() for s in
                 itertools.chain.from_iterable(t.split(".") for t in text.splitlines())
                 if len(s := s.strip()) > 15]
    return sentences, text

def cluster_sentences(sentences, n_clusters=5):
    emb = load_embedder().encode(sentences, batch_size=32, show_progress_bar=False)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(emb)
    umap_2d = umap.UMAP(random_state=42).fit_transform(emb)
    df = pd.DataFrame({
        "sentence": sentences,
        "cluster": km.labels_,
        "x": umap_2d[:,0],
        "y": umap_2d[:,1],
    })
    return df, emb

def top_tfidf_keywords(df, n=5):
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(df["sentence"])
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for c in sorted(df.cluster.unique()):
        idx = np.where(df.cluster == c)[0]
        tfidf_mean = X[idx].mean(axis=0).A1
        top_idx = tfidf_mean.argsort()[-n:][::-1]
        out[c] = ", ".join(vocab[top_idx])
    return out

# ─────────────────────────────── UI ───────────────────────────────────────
st.set_page_config(page_title="Voice-to-Insights Demo", layout="wide")
st.title("🎙️➡️📊 Voice-to-Insights Call-Center Demo")

st.info(
    "🔔 **Demo Notice**  \n"
    "Tiny Whisper model, MiniLM embeddings, CPU only. "
    "Not for production workloads. "
    "[Contact me](https://drtomharty.com/bio) for full speech-analytics stacks.",
    icon="💡",
)

audio_file = st.file_uploader("Upload a WAV or MP3 file (≤5 min, ≤20 MB)", type=["wav","mp3"])
n_clusters = st.slider("Topic clusters (k-means k)", 3, 10, 5)

if st.button("🚀 Transcribe & Analyze") and audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with st.spinner("Transcribing… (may take ~30-60 s)"):
        sentences, full_text = transcribe(tmp_path)

    if not sentences:
        st.error("Transcript empty. Try clearer audio.")
        st.stop()

    with st.spinner("Embedding & clustering…"):
        df, embeddings = cluster_sentences(sentences, n_clusters=n_clusters)
        keywords = top_tfidf_keywords(df)

    # ─── Scatter plot ──────────────────────────────────────────────
    st.subheader("Topic clusters (UMAP 2-D)")
    fig = px.scatter(df, x="x", y="y", color=df.cluster.astype(str),
                     hover_data={"sentence":True}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ─── Keyword table ─────────────────────────────────────────────
    st.subheader("Top keywords per cluster")
    kw_df = pd.DataFrame({"cluster": keywords.keys(), "keywords": keywords.values()})
    st.dataframe(kw_df)

    # ─── Transcript download ──────────────────────────────────────
    out_csv = df[["sentence","cluster"]]
    st.download_button("⬇️ Download transcript + clusters CSV",
                       out_csv.to_csv(index=False).encode(),
                       file_name="transcript_clusters.csv",
                       mime="text/csv")

    with st.expander("📄 Full transcript"):
        st.write(full_text)
else:
    st.caption("Upload audio and click **Transcribe & Analyze**.")
