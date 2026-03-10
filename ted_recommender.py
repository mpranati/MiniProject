# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# LOAD DATASET
# ===============================
print("Loading dataset...")
data = pd.read_csv("ted_main_v2.csv")

data = data[['title', 'about_talk', 'tags', 'Link']]
data.rename(columns={'about_talk': 'description'}, inplace=True)
data.fillna('', inplace=True)

data['content'] = data['title'] + " " + data['description'] + " " + data['tags']

# ===============================
# LOAD MODEL
# ===============================
print("Loading ML model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings for TED Talks...")
talk_embeddings = model.encode(
    data['content'].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True   # IMPORTANT FIX
)

# ===============================
# BUILD KNN MODEL
# ===============================
knn = NearestNeighbors(n_neighbors=20, metric='cosine')
knn.fit(talk_embeddings)

# ===============================
# REVERSE LEARNING ENGINE
# ===============================
print("\nTED Talks Recommendation System")
print("Instead of entering a topic, enter your confusion/problem.")

user_confusion = input("\nDescribe your confusion: ").lower()

print("\nAnalyzing your confusion...\n")

# Encode user query (normalize same way)
user_vec = model.encode(
    [user_confusion],
    normalize_embeddings=True
)

# ===============================
# 🧠 INTELLECTUAL HONESTY MODE™ (FIXED PROPERLY)
# ===============================

# Compute cosine similarity directly
all_similarities = cosine_similarity(user_vec, talk_embeddings)
best_match_score = np.max(all_similarities)

print("Best Similarity Score:", round(best_match_score, 3))

relevance_threshold = 0.45   # Balanced & realistic

if best_match_score < relevance_threshold:
    print("=" * 80)
    print("🧠 Intellectual Honesty Mode™️ Activated")
    print("TED Talks may not be sufficient for this topic.")
    print("Consider textbooks, research papers, or structured courses.")
    print("=" * 80)

else:

    # Get top matches
    distances, indices = knn.kneighbors(user_vec)

    # ===============================
    # 🚀 LEARNING EXHAUSTION DETECTION™
    # ===============================
    novelty_threshold = 0.85
    recommended_vectors = []

    print("=" * 80)
    print("Recommended TED Talks To Resolve Your Confusion:")
    print("=" * 80)

    for rank, idx in enumerate(indices[0], start=1):

        talk = data.iloc[idx]
        talk_vec = talk_embeddings[idx]

        similarity_with_query = all_similarities[0][idx]

        print(f"\n🔹 Recommendation #{rank}")
        print("Title :", talk['title'])
        print(f"Relevance Score : {similarity_with_query:.2f}")

        if len(recommended_vectors) == 0:
            print("Novelty Score : 1.00 (First recommendation)")
            print("Why Recommended : Strong semantic match with your confusion.")
            print("Watch :", talk['Link'])
            print("-" * 80)
            recommended_vectors.append(talk_vec)
            continue

        similarities = cosine_similarity([talk_vec], recommended_vectors)
        max_similarity = similarities.max()
        novelty_score = 1 - max_similarity

        print(f"Similarity with previous talks : {max_similarity:.2f}")
        print(f"Novelty Score : {novelty_score:.2f}")

        if max_similarity > novelty_threshold:
            print("\n🧠 Learning Exhaustion Detected!")
            print("Further talks are too similar.")
            print("Stopping to avoid repetition.")
            print("=" * 80)
            break

        print("Why Recommended : Covers a different angle of your confusion.")
        print("Watch :", talk['Link'])
        print("-" * 80)

        recommended_vectors.append(talk_vec)

    print("\nYour Confusion :", user_confusion.title())
    print("\nSystem Used:")
    print("✔ Reverse Learning Engine")
    print("✔ Intellectual Honesty Mode™")
    print("✔ Learning Exhaustion Detection™")
    print("=" * 80)