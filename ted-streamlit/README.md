# TED Talks Recommender

A personalised TED talk recommender with real ML training.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## Get 500+ talks (recommended)

1. Get a free YouTube API key:
   - Go to https://console.cloud.google.com
   - New project → Enable "YouTube Data API v3"
   - Credentials → Create API Key (free, no billing)

2. Copy `.env.example` to `.env` and add your key:
   ```
   YOUTUBE_API_KEY=your_key_here
   ```

3. Restart the app — it will fetch 500+ TED talks automatically.

Without the key the app uses TED RSS feeds (~300-400 talks).

## Features

| Tab | Description |
|-----|-------------|
| ✦ Ask | Natural language search — type feelings, questions, keywords |
| Discover | Personalised recommendations |
| Browse | All talks paginated |
| Search | Keyword search |
| Profile | Your interests and mastery |
| Feedback | Your ratings history |
| 🧠 ML Training | Live view of models training from your interactions |
| 📊 Evaluation | Precision@K, NDCG@K, Coverage, Diversity vs baselines |

## ML Systems

- **TF-IDF + SVD + KNN** — base retrieval trained on talk corpus
- **Online SGD** — updates on every watch/skip (incremental learning)
- **Gradient Boosted Trees** — retrained every 5 interactions from history
- **Evaluation metrics** — Precision@5, NDCG@5, Coverage, Diversity vs Random and Popularity baselines
