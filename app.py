# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="TED Reverse Learning Engine",
    layout="wide",
    page_icon="🧠"
)

# ===============================
# CLEAN STYLING
# ===============================
st.markdown("""
<style>
body {
    background-color: #0b1220;
}

.header-box {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(90deg,#e62b1e,#a31515);
    color: white;
    margin-bottom: 30px;
}

.recommend-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #151c2c;
    margin-bottom: 15px;
}

.score-badge {
    padding: 6px 12px;
    border-radius: 20px;
    background-color: #2b344a;
    font-size: 12px;
    margin-right: 8px;
    display: inline-block;
}

.watch-btn {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 8px;
    background-color: #e62b1e;
    color: white !important;
    text-decoration: none;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="header-box">
<h1 style="margin:0;">TED Talks Recommendation System</h1>
</div>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA & MODEL (HIDDEN)
# ===============================
@st.cache_data
def load_data():
    data = pd.read_csv("ted_main_v2.csv")
    data = data[['title', 'about_talk', 'tags', 'Link']]
    data.rename(columns={'about_talk': 'description'}, inplace=True)
    data.fillna('', inplace=True)
    data['content'] = data['title'] + " " + data['description'] + " " + data['tags']
    return data

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def generate_embeddings(data):
    model = load_model()
    return model.encode(
        data['content'].tolist(),
        normalize_embeddings=True
    )

data = load_data()
model = load_model()
talk_embeddings = generate_embeddings(data)

knn = NearestNeighbors(n_neighbors=15, metric='cosine')
knn.fit(talk_embeddings)

# ===============================
# USER INPUT
# ===============================
st.subheader("Describe Your Confusion")

user_confusion = st.text_area(
    "",
    placeholder="Example: Why do people resist scientific evidence?",
    height=150
)

analyze_btn = st.button("🚀 Analyze")

# ===============================
# ANALYSIS
# ===============================
if analyze_btn:

    if user_confusion.strip() == "":
        st.warning("Please describe your confusion.")
        st.stop()

    user_vec = model.encode(
        [user_confusion.lower()],
        normalize_embeddings=True
    )

    all_similarities = cosine_similarity(user_vec, talk_embeddings)
    best_score = float(np.max(all_similarities))

    # ===============================
    # MATCH CONFIDENCE
    # ===============================
    st.subheader("Match Confidence")
    st.progress(best_score)

    col1, col2 = st.columns(2)
    col1.metric("Best Similarity Score", round(best_score, 3))
    col2.metric("Relevance Threshold", 0.45)

    st.divider()

    # ===============================
    # INTELLECTUAL HONESTY MODE™
    # ===============================
    if best_score < 0.45:
        st.error("Intellectual Honesty Mode™ Activated")
        st.info("TED Talks may not sufficiently address this confusion.")
        st.stop()

    # ===============================
    # RECOMMENDATIONS
    # ===============================
    distances, indices = knn.kneighbors(user_vec)

    novelty_threshold = 0.85
    recommended_vectors = []

    st.subheader("Reverse-Mapped Learning Path")

    for rank, idx in enumerate(indices[0], start=1):

        talk = data.iloc[idx]
        talk_vec = talk_embeddings[idx]
        similarity_with_query = all_similarities[0][idx]

        if len(recommended_vectors) == 0:
            max_similarity = 0
            novelty_score = 1.0
        else:
            similarities = cosine_similarity([talk_vec], recommended_vectors)
            max_similarity = similarities.max()
            novelty_score = 1 - max_similarity

        # ===============================
        # LEARNING EXHAUSTION DETECTION™
        # ===============================
        if max_similarity > novelty_threshold:
            st.warning("Learning Exhaustion Detected — Stopping to avoid repetition.")
            break

        # ===============================
        # CLEAN CARD DISPLAY
        # ===============================
        st.markdown(f"""
        <div class="recommend-card">
            <h4>#{rank} — {talk['title']}</h4>
            <div>
                <span class="score-badge">
                    Relevance: {round(float(similarity_with_query),2)}
                </span>
                <span class="score-badge">
                    Novelty: {round(float(novelty_score),2)}
                </span>
            </div>
            <br>
            <a href="{talk['Link']}" target="_blank" class="watch-btn">
                ▶ Watch Talk
            </a>
        </div>
        """, unsafe_allow_html=True)

        recommended_vectors.append(talk_vec)
