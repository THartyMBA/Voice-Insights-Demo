# Voice-Insights-Demo

🎙️➡️📊 Voice-to-Insights Call-Center Demo
A Streamlit proof-of-concept that transforms a single call recording into actionable insights:

Transcribe audio via a CPU-friendly Whisper model

Cluster sentences into topics (k-means on MiniLM embeddings)

Visualize clusters in 2D (UMAP scatter plot)

Extract top TF-IDF keywords per cluster

Download the labeled transcript as CSV

Demo only—no diarization, speaker separation, or PII redaction.
For enterprise speech-analytics pipelines, contact me.

✨ Key Features
Local transcription using faster-whisper tiny model (CPU-only)

Sentence embedding via MiniLM (all-in-RAM, no external API)

Topic clustering (k-means) with adjustable k

Dimensionality reduction (UMAP) for interactive scatter plotting

Keyword extraction per topic via TF-IDF

Downloadable output: transcript + cluster labels CSV

🚀 Quick Start (Local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/voice-insights-demo.git
cd voice-insights-demo
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run voice_insights_demo.py
Open http://localhost:8501

Upload a WAV or MP3 file (≤5 min)

Adjust Topic clusters (k) slider and click Transcribe & Analyze

☁️ Deploy on Streamlit Cloud (Free)
Push this repo (public or private) to GitHub under THartyMBA.

Visit streamlit.io/cloud → New app → select your repo/branch → Deploy.

Share the generated URL—no secrets or tokens required.

(Model weights are fetched automatically; runs on the free CPU tier in ~60 s for a 1-min file.)

🛠️ Requirements
shell
Copy
Edit
streamlit>=1.32
faster-whisper
sentence-transformers
scikit-learn
umap-learn
plotly
pandas
numpy
🗂️ Repo Structure
vbnet
Copy
Edit
voice-insights-demo/
├─ voice_insights_demo.py   ← single-file Streamlit app
├─ requirements.txt
└─ README.md                ← you’re reading it
📜 License
CC0 1.0 – Public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid web UIs for Python

faster-whisper – CPU Whisper transcription

Sentence-Transformers – MiniLM embeddings

scikit-learn – k-means clustering & TF-IDF

UMAP – dimensionality reduction

Plotly – interactive scatter plots

Upload, transcribe, cluster, and derive insights—all in one demo! 🚀
