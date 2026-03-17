"""
engine.py — TED Recommender ML Engine

ML components:
  1. TF-IDF + SVD       — converts talk text into vectors
  2. KNN                — finds talks similar to ones you liked
  3. Online SGD         — learns your taste one interaction at a time
  4. Hybrid scoring     — combines all signals into one final score

Removed: GradientBoostingClassifier (needs too much data for a single user),
         Mastery/Exhaustion/Honesty scoring (rule-based, not ML).
"""

import os, json, time, math, pickle, logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
USER_PATH     = os.path.join(DATA_DIR, "user.json")
MODEL_PATH    = os.path.join(DATA_DIR, "model.pkl")
TRAINING_LOG  = os.path.join(DATA_DIR, "training_log.json")

# Hybrid blend weights
W_TFIDF  = 0.55
W_KNN    = 0.45
W_SGD    = 0.35   # blended in after 3+ interactions
MIN_SCORE = 0.25

REACTION_TAGS = {
    "mind-blowing":          {"sentiment": 1.0},
    "highly relevant":       {"sentiment": 0.9},
    "new perspective":       {"sentiment": 0.85},
    "well explained":        {"sentiment": 0.8},
    "want more like this":   {"sentiment": 0.9},
    "already knew this":     {"sentiment": 0.2},
    "too long":              {"sentiment": 0.4},
    "too short":             {"sentiment": 0.5},
    "too basic":             {"sentiment": 0.1},
    "not relevant":          {"sentiment": 0.0},
    "wrong topic":           {"sentiment": 0.0},
    "don't recommend again": {"sentiment": 0.0},
}
STAR_TO_RATING = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}


# ── Training Log ──────────────────────────────────────────────────

class TrainingLog:
    """Records every SGD training event for transparency."""

    def __init__(self):
        self.events: List[Dict] = []
        self._load()

    def _load(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(TRAINING_LOG):
            try:
                self.events = json.load(open(TRAINING_LOG)).get("events", [])
            except Exception:
                self.events = []

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        json.dump({"events": self.events[-200:]},
                  open(TRAINING_LOG, "w"), indent=2)

    def record(self, event_type: str, details: Dict):
        self.events.append({"ts": time.time(), "type": event_type, **details})
        self._save()

    def recent(self, n: int = 30) -> List[Dict]:
        return sorted(self.events, key=lambda e: -e["ts"])[:n]

    def summary(self) -> Dict:
        return {
            "total_events": len(self.events),
            "sgd_updates":  sum(1 for e in self.events if e["type"] == "sgd_update"),
            "knn_trains":   sum(1 for e in self.events if e["type"] == "knn_train"),
        }

    def clear(self):
        self.events = []
        self._save()


# ── User Profile ──────────────────────────────────────────────────

class UserProfile:
    """Stores and persists all user interaction state."""

    def __init__(self):
        self.watched:             Dict[str, float] = {}   # id → timestamp
        self.ratings:             Dict[str, float] = {}   # id → 0-1
        self.skipped:             Dict[str, float] = {}   # id → timestamp
        self.topic_exposure:      Dict[str, int]   = defaultdict(int)
        self.speaker_exposure:    Dict[str, int]   = defaultdict(int)
        self.interest_vector:     Dict[str, float] = {}
        self._explicit_interests: set              = set()
        self.session_talks:       List[str]        = []
        self.total_interactions:  int              = 0
        self._load()

    def _load(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(USER_PATH):
            return
        try:
            d = json.load(open(USER_PATH))
            self.watched             = d.get("watched", {})
            self.ratings             = d.get("ratings", {})
            self.skipped             = d.get("skipped", {})
            self.topic_exposure      = defaultdict(int, d.get("topic_exposure", {}))
            self.speaker_exposure    = defaultdict(int, d.get("speaker_exposure", {}))
            self.interest_vector     = d.get("interest_vector", {})
            self._explicit_interests = set(d.get("explicit_interests", []))
            self.total_interactions  = d.get("total_interactions", 0)
        except Exception as e:
            logger.warning(f"Could not load user: {e}")

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            json.dump({
                "watched":            self.watched,
                "ratings":            self.ratings,
                "skipped":            self.skipped,
                "topic_exposure":     dict(self.topic_exposure),
                "speaker_exposure":   dict(self.speaker_exposure),
                "interest_vector":    self.interest_vector,
                "explicit_interests": list(self._explicit_interests),
                "total_interactions": self.total_interactions,
            }, open(USER_PATH, "w"), indent=2)
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def reset(self):
        self.__init__()
        if os.path.exists(USER_PATH): os.remove(USER_PATH)

    def record_watch(self, talk: Dict, rating: Optional[float] = None):
        tid = str(talk["id"])
        self.watched[tid]  = time.time()
        self.ratings[tid]  = rating if rating is not None else 0.7
        self.session_talks.append(tid)
        self.total_interactions += 1
        for tag in talk.get("tags", []):
            self.topic_exposure[tag] += 1
        if talk.get("speaker"):
            self.speaker_exposure[talk["speaker"]] += 1
        self._update_interest(talk, self.ratings[tid])
        self.save()

    def record_skip(self, talk: Dict):
        self.skipped[str(talk["id"])] = time.time()
        self.total_interactions += 1
        self._update_interest(talk, 0.1)
        self.save()

    def set_interests(self, topics: List[str]):
        """Replace explicit interests — keeps learned signals from watches."""
        learned = {t: w for t, w in self.interest_vector.items()
                   if abs(w) > 0.1 and t not in self._explicit_interests}
        self._explicit_interests = set(topics)
        self.interest_vector = learned
        for t in topics:
            self.interest_vector[t] = min(1.0, self.interest_vector.get(t, 0.0) + 0.6)
        self.save()

    def _update_interest(self, talk: Dict, rating: float):
        """Shift interest vector based on watch/skip signal."""
        alpha  = 0.5
        signal = (rating - 0.5) * 2   # maps 0-1 rating to -1 to +1
        for tag in talk.get("tags", []):
            cur = self.interest_vector.get(tag, 0.0)
            self.interest_vector[tag] = max(-1.0, min(1.0,
                cur + alpha * signal * (1 - abs(cur) * 0.5)))

    def has_watched(self, tid: str) -> bool: return str(tid) in self.watched
    def has_skipped(self, tid: str) -> bool: return str(tid) in self.skipped
    def get_rating(self,  tid: str) -> float: return self.ratings.get(str(tid), 0.5)
    def positive_topics(self) -> List[str]:
        return [t for t, w in self.interest_vector.items() if w > 0.2]
    def negative_topics(self) -> List[str]:
        return [t for t, w in self.interest_vector.items() if w < -0.1]

    def stats(self) -> Dict:
        return {
            "watched_count":      len(self.watched),
            "total_interactions": self.total_interactions,
            "topics_explored":    len(self.topic_exposure),
            "speakers_seen":      len(self.speaker_exposure),
            "top_interests":      sorted(
                [(k, round(v, 2)) for k, v in self.interest_vector.items() if v > 0],
                key=lambda x: -x[1])[:10],
        }


# ── Scoring Engine ────────────────────────────────────────────────

class ScoringEngine:
    """
    Computes relevance and novelty scores for a talk given user profile.
    These feed into the hybrid score alongside TF-IDF, KNN, and SGD.
    """

    def __init__(self, profile: UserProfile):
        self.p = profile

    def relevance(self, talk: Dict, ml_score: float) -> float:
        """How well does this talk match the user's interests?"""
        tags = set(talk.get("tags", []))
        pos  = set(self.p.positive_topics())
        neg  = set(self.p.negative_topics())
        base = ml_score
        if tags:
            pos_ratio = len(tags & pos) / len(tags)
            neg_ratio = len(tags & neg) / len(tags)
            base = base * 0.6 + (pos_ratio - 0.5 * neg_ratio) * 0.4
        sp_exp = self.p.speaker_exposure.get(talk.get("speaker", ""), 0)
        if sp_exp == 1:   base += 0.05   # slight boost for speakers seen once
        elif sp_exp >= 3: base -= 0.08   # penalty for overexposure
        return max(0.0, min(1.0, base))

    def novelty(self, talk: Dict) -> float:
        """How fresh / unexplored is this talk for the user?"""
        n      = 1.0
        sp_exp = self.p.speaker_exposure.get(talk.get("speaker", ""), 0)
        n -= min(1.0, sp_exp * 0.2) * 0.4
        tags = talk.get("tags", [])
        if tags:
            avg = sum(
                1.0 / (1.0 + math.log(1 + self.p.topic_exposure[t]))
                for t in tags
            ) / len(tags)
            n = n * 0.4 + avg * 0.6
        if sp_exp == 0 and talk.get("speaker"):
            n = min(1.0, n + 0.15)   # bonus for brand new speaker
        return max(0.0, min(1.0, n))

    def honesty(self, talk: Dict) -> float:
        """
        Honest mode — penalises talks that are too safe/familiar,
        rewards talks that challenge the user or cover unexplored ground.
        """
        tags       = set(talk.get("tags", []))
        pos        = set(self.p.positive_topics())
        neg        = set(self.p.negative_topics())
        score      = 0.5
        unexplored = tags - pos - neg
        # Reward talks that venture outside known territory
        if unexplored:
            score += 0.2 * (len(unexplored) / max(1, len(tags)))
        # Small reward for talks that touch on disliked topics (broadening)
        mild = tags & neg
        if mild and len(mild) / max(1, len(tags)) < 0.5:
            score += 0.1 * (len(mild) / max(1, len(tags)))
        # Penalise talks entirely within already-loved topics (echo chamber)
        if tags and tags <= pos:
            score -= 0.15
        return max(0.0, min(1.0, score))

    def composite(self, talk: Dict, ml_score: float) -> Tuple[float, Dict]:
        """Combine relevance + novelty + honesty with ML score."""
        r     = self.relevance(talk, ml_score)
        n     = self.novelty(talk)
        h     = self.honesty(talk)
        final = max(0.0, min(1.0, 0.5 * r + 0.3 * n + 0.2 * h))
        return final, {
            "relevance": round(r, 3),
            "novelty":   round(n, 3),
            "honesty":   round(h, 3),
            "ml_score":  round(ml_score, 3),
            "final":     round(final, 3),
        }


# ── ReverseLearner (kept for Profile tab pivot suggestions) ───────

class ReverseLearner:
    """
    Tracks topic mastery for the Profile tab.
    Used only for display — not part of recommendation scoring.
    """

    def __init__(self, profile: UserProfile):
        self.p = profile

    def mastery_level(self, topic: str) -> str:
        exp = self.p.topic_exposure.get(topic, 0)
        if exp >= 10: return "expert"
        if exp >= 6:  return "proficient"
        if exp >= 3:  return "learning"
        return "novice"

    def mastery_report(self) -> Dict:
        return {
            topic: {
                "level":         self.mastery_level(topic),
                "talks_watched": count,
            }
            for topic, count in self.p.topic_exposure.items()
            if count >= 2
        }

    def suggested_pivots(self) -> List[str]:
        ADJACENT = {
            "psychology":  ["neuroscience","philosophy","relationships"],
            "creativity":  ["design","innovation","art"],
            "leadership":  ["business","motivation","communication"],
            "science":     ["technology","environment","health"],
            "technology":  ["ai","science","innovation"],
            "happiness":   ["philosophy","health","relationships"],
            "health":      ["neuroscience","psychology","science"],
            "ai":          ["technology","philosophy","science"],
            "education":   ["psychology","neuroscience","creativity"],
            "business":    ["economics","leadership","innovation"],
        }
        explored = set(self.p.topic_exposure.keys())
        pivots   = set()
        for topic in explored:
            for adj in ADJACENT.get(topic, []):
                if adj not in explored:
                    pivots.add(adj)
        return sorted(pivots)[:5]


# ── KNN Recommender ───────────────────────────────────────────────

class KNNRecommender:
    """
    Item-based K-Nearest Neighbours.
    For each talk you liked, finds the K most similar talks
    in TF-IDF/SVD vector space and votes them up.
    """

    def __init__(self, k: int = 10):
        self.k      = k
        self.model: Optional[NearestNeighbors] = None
        self.matrix: Optional[np.ndarray]      = None
        self.ids:    List[str]                  = []
        self.fitted  = False

    def fit(self, matrix: np.ndarray, ids: List[str]):
        k = min(self.k, matrix.shape[0] - 1)
        self.model  = NearestNeighbors(
            n_neighbors=k + 1, metric="cosine",
            algorithm="brute", n_jobs=-1
        )
        self.model.fit(matrix)
        self.matrix = matrix
        self.ids    = ids
        self.fitted = True

    def neighbours(self, idx: int) -> List[Tuple[int, float]]:
        if not self.fitted: return []
        dists, indices = self.model.kneighbors(self.matrix[idx].reshape(1, -1))
        return [
            (int(i), max(0.0, 1.0 - float(d)))
            for d, i in zip(dists[0], indices[0]) if i != idx
        ]

    def score_all(self, liked_ids: List[str],
                  ratings: Dict[str, float],
                  exclude: set) -> Dict[str, float]:
        if not self.fitted or not liked_ids: return {}
        votes: Dict[int, float] = {}
        for lid in liked_ids:
            if lid not in self.ids: continue
            w = ratings.get(lid, 0.5)
            if w < 0.6: continue   # only use talks rated positively
            for nb_idx, sim in self.neighbours(self.ids.index(lid)):
                nb_id = self.ids[nb_idx]
                if nb_id not in exclude:
                    votes[nb_idx] = votes.get(nb_idx, 0.0) + sim * w
        if not votes: return {}
        mx = max(votes.values())
        return {self.ids[i]: v / mx for i, v in votes.items()} if mx > 0 else {}

    def info(self) -> Dict:
        return {"fitted": self.fitted, "k": self.k,
                "corpus_size": len(self.ids)}


# ── Online SGD Learner ────────────────────────────────────────────

class OnlineLearner:
    """
    Binary SGD classifier: learns to predict liked (1) vs disliked (0).
    Uses partial_fit() — one gradient step per interaction.
    No retraining from scratch — true incremental online learning.
    """

    def __init__(self):
        self.model   = SGDClassifier(
            loss="log_loss", learning_rate="optimal",
            alpha=0.001, random_state=42
        )
        self.scaler       = StandardScaler()
        self.trained      = False
        self.n_seen       = 0
        self.n_liked      = 0
        self.n_disliked   = 0
        self._classes_set = False

    def _features(self, talk: Dict, profile: UserProfile,
                  tfidf_score: float = 0.5) -> np.ndarray:
        """8-dimensional feature vector for one (user, talk) interaction."""
        tags     = set(talk.get("tags", []))
        pos_tags = set(profile.positive_topics())
        neg_tags = set(profile.negative_topics())
        n_tags   = max(1, len(tags))

        sp_exp      = min(5, profile.speaker_exposure.get(talk.get("speaker", ""), 0))
        avg_tag_exp = sum(profile.topic_exposure.get(t, 0) for t in tags) / n_tags
        past_r      = list(profile.ratings.values())
        avg_rating  = float(np.mean(past_r)) if past_r else 0.5

        # Recency-weighted engagement
        now   = time.time()
        decay = sum(
            profile.get_rating(tid) / (1.0 + (now - ts) / 86400 / 30)
            for tid, ts in profile.watched.items()
        )
        decay = min(1.0, decay / max(1, len(profile.watched)))

        return np.array([
            tfidf_score,
            len(tags & pos_tags) / n_tags,
            len(tags & neg_tags) / n_tags,
            min(1.0, sp_exp / 5),
            min(1.0, avg_tag_exp / 10),
            avg_rating,
            decay,
            min(1.0, float(len(profile.watched)) / 20),
        ], dtype=float)

    def update(self, talk: Dict, profile: UserProfile,
               tfidf_score: float, liked: bool,
               log: Optional[TrainingLog] = None):
        """One incremental gradient step on this single interaction."""
        x = self._features(talk, profile, tfidf_score).reshape(1, -1)
        y = np.array([1 if liked else 0])
        self.scaler.partial_fit(x)
        x_s = self.scaler.transform(x)
        if not self._classes_set:
            self.model.partial_fit(x_s, y, classes=np.array([0, 1]))
            self._classes_set = True
        else:
            self.model.partial_fit(x_s, y)
        self.trained = True
        self.n_seen += 1
        if liked: self.n_liked   += 1
        else:     self.n_disliked += 1
        if log:
            log.record("sgd_update", {
                "talk_title": talk.get("title", "")[:50],
                "liked":      liked,
                "n_seen":     self.n_seen,
            })

    def predict_proba(self, talk: Dict, profile: UserProfile,
                      tfidf_score: float = 0.5) -> float:
        """Returns P(user likes this talk). Falls back to tfidf_score if untrained."""
        if not self.trained or self.n_seen < 3:
            return tfidf_score
        try:
            x = self._features(talk, profile, tfidf_score).reshape(1, -1)
            return float(self.model.predict_proba(
                self.scaler.transform(x))[0][1])
        except Exception:
            return tfidf_score

    def info(self) -> Dict:
        return {
            "trained":    self.trained,
            "n_seen":     self.n_seen,
            "n_liked":    self.n_liked,
            "n_disliked": self.n_disliked,
        }


# ── ML Engine ─────────────────────────────────────────────────────

def _clean(text: str) -> str:
    import re
    STOP = {
        "the","a","an","and","or","in","on","at","to","for","of","with","by",
        "this","it","is","was","are","be","have","do","will","not","we","they",
        "he","she","you","i","my","your","his","her","our","its","that","as"
    }
    text = re.sub(r"http\S+|[^a-z0-9\s]", " ", text.lower())
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 2)


class MLEngine:
    """
    Hybrid recommender: TF-IDF + SVD + KNN + Online SGD.

    Scoring pipeline:
      1. TF-IDF cosine similarity  — content match to user profile
      2. KNN item similarity       — similar to talks you liked
      3. Hybrid blend              — 0.55 × TF-IDF + 0.45 × KNN
      4. ScoringEngine             — relevance + novelty adjustment
      5. SGD blend                 — 0.65 × above + 0.35 × SGD (after 3 interactions)
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000, ngram_range=(1, 2),
            stop_words="english", min_df=1, sublinear_tf=True
        )
        self.svd     = TruncatedSVD(n_components=50, random_state=42)
        self.knn     = KNNRecommender(k=10)
        self.online  = OnlineLearner()
        self.log     = TrainingLog()
        self.matrix: Optional[np.ndarray] = None
        self.ids:    List[str]             = []
        self.corpus: List[Dict]            = []
        self.trained = False

    def _to_doc(self, talk: Dict) -> str:
        tags = " ".join(talk.get("tags", []))
        return _clean(
            f"{talk.get('title','')} {talk.get('description','')} "
            f"{talk.get('speaker','')} {tags} {tags}"
        )

    def train(self, talks: List[Dict]):
        """Initial corpus training — builds TF-IDF matrix and KNN index."""
        if not talks: return
        self.corpus = talks
        self.ids    = [str(t["id"]) for t in talks]
        docs        = [self._to_doc(t) for t in talks]
        try:
            tfidf = self.vectorizer.fit_transform(docs)
            if tfidf.shape[0] >= 10 and tfidf.shape[1] >= 50:
                self.matrix = normalize(self.svd.fit_transform(tfidf))
            else:
                self.matrix = normalize(tfidf.toarray())
            self.knn.fit(self.matrix, self.ids)
            self.trained = True
            self._save()
            self.log.record("knn_train", {
                "n_talks":      len(talks),
                "matrix_shape": list(self.matrix.shape),
            })
            logger.info(f"Trained: {len(talks)} talks, {self.matrix.shape}")
        except Exception as e:
            logger.error(f"Training failed: {e}")

    def on_interaction(self, talk: Dict, profile: UserProfile, liked: bool):
        """
        Called after every Watch or Skip.
        Triggers one SGD gradient step immediately.
        """
        tid       = str(talk["id"])
        ts        = self.tfidf_scores(profile)
        tfidf_s   = ts.get(tid, 0.5)
        self.online.update(talk, profile, tfidf_s, liked, self.log)
        self._save()

    def _user_vec(self, profile: UserProfile) -> Optional[np.ndarray]:
        """Build user preference vector from watch history + interest vector."""
        if not self.trained or self.matrix is None: return None
        vecs, total_w = [], 0.0
        for tid in profile.watched:
            if tid not in self.ids: continue
            r   = profile.get_rating(tid)
            age = (time.time() - profile.watched[tid]) / 86400
            w   = r / (1.0 + age / 30)   # time-decay: older watches count less
            vecs.append(self.matrix[self.ids.index(tid)] * w)
            total_w += w
        if not vecs or total_w == 0: return None
        uv = np.sum(vecs, axis=0) / total_w
        # Blend with explicit interest vector (30% weight)
        if profile.interest_vector:
            doc = " ".join(
                t * max(1, int(w * 5))
                for t, w in profile.interest_vector.items() if w > 0
            )
            if doc.strip():
                try:
                    tv = self.vectorizer.transform([doc])
                    tv = (self.svd.transform(tv)[0]
                          if hasattr(self.svd, "components_")
                          else tv.toarray()[0])
                    uv = uv * 0.7 + normalize(tv.reshape(1, -1))[0] * 0.3
                except Exception:
                    pass
        return normalize(uv.reshape(1, -1))[0]

    def tfidf_scores(self, profile: UserProfile) -> Dict[str, float]:
        """Cosine similarity between user vector and every talk."""
        if not self.trained or self.matrix is None: return {}
        uv = self._user_vec(profile)
        if uv is None:
            # Cold start — use interest vector directly if available
            if profile.interest_vector:
                doc = " ".join(
                    t * max(1, int(w * 5))
                    for t, w in profile.interest_vector.items() if w > 0
                )
                if doc.strip():
                    try:
                        tv = self.vectorizer.transform([doc])
                        tv = (self.svd.transform(tv)[0]
                              if hasattr(self.svd, "components_")
                              else tv.toarray()[0])
                        uv = normalize(tv.reshape(1, -1))[0]
                    except Exception:
                        pass
            if uv is None:
                uv = normalize(self.matrix.mean(axis=0).reshape(1, -1))[0]
        sims = cosine_similarity(uv.reshape(1, -1), self.matrix)[0]
        return {self.ids[i]: float(sims[i]) for i in range(len(self.ids))}

    def knn_scores(self, profile: UserProfile,
                   exclude: set) -> Dict[str, float]:
        liked = [tid for tid in profile.watched
                 if profile.get_rating(tid) >= 0.6]
        return self.knn.score_all(liked, profile.ratings, exclude)

    def recommend(self, profile: UserProfile, n: int = 6) -> List[Dict]:
        if not self.corpus: return []

        exclude  = set(profile.watched) | set(profile.skipped)
        scoring  = ScoringEngine(profile)
        ts       = self.tfidf_scores(profile)
        ks       = self.knn_scores(profile, exclude)
        candidates = []

        for talk in self.corpus:
            tid = str(talk["id"])
            if tid in exclude: continue

            tfidf_s  = ts.get(tid, 0.0)
            knn_s    = ks.get(tid, 0.0)

            # Step 1: hybrid TF-IDF + KNN
            hybrid_s = (W_KNN * knn_s + W_TFIDF * tfidf_s
                        if knn_s > 0 else tfidf_s)

            # Step 2: relevance + novelty adjustment
            final, bd = scoring.composite(talk, hybrid_s)

            # Step 3: blend in SGD signal after enough data
            sgd_s = self.online.predict_proba(talk, profile, tfidf_s)
            if self.online.n_seen >= 3:
                final = final * (1 - W_SGD) + sgd_s * W_SGD

            final = max(0.0, min(1.0, final))
            bd.update({
                "tfidf_score":  round(tfidf_s, 3),
                "knn_score":    round(knn_s, 3),
                "hybrid_score": round(hybrid_s, 3),
                "sgd_score":    round(sgd_s, 3),
                "final":        round(final, 3),
            })

            if final >= MIN_SCORE:
                candidates.append({
                    **talk,
                    "score":           final,
                    "score_breakdown": bd,
                })

        candidates.sort(key=lambda x: -x["score"])
        return candidates[:n]

    def info(self) -> Dict:
        return {
            "trained":      self.trained,
            "corpus_size":  len(self.corpus),
            "matrix_shape": list(self.matrix.shape) if self.matrix is not None else None,
            "knn":          self.knn.info(),
            "online_sgd":   self.online.info(),
        }

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "svd":        self.svd,
                "matrix":     self.matrix,
                "ids":        self.ids,
                "knn":        self.knn,
                "online":     self.online,
            }, open(MODEL_PATH, "wb"))
        except Exception as e:
            logger.warning(f"Save failed: {e}")

    def load(self) -> bool:
        if not os.path.exists(MODEL_PATH): return False
        try:
            d = pickle.load(open(MODEL_PATH, "rb"))
            self.vectorizer = d["vectorizer"]
            self.svd        = d["svd"]
            self.matrix     = d["matrix"]
            self.ids        = d["ids"]
            self.knn        = d.get("knn",    KNNRecommender())
            self.online     = d.get("online", OnlineLearner())
            self.trained    = True
            logger.info(f"Model loaded: {len(self.ids)} talks")
            return True
        except Exception as e:
            logger.warning(f"Load failed: {e}")
            return False


# ── Feedback Store ────────────────────────────────────────────────

FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.json")


class FeedbackStore:

    def __init__(self):
        self.entries: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(FEEDBACK_PATH):
            try:
                self.entries = json.load(open(FEEDBACK_PATH)).get("entries", {})
            except Exception:
                pass

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        json.dump({"entries": self.entries}, open(FEEDBACK_PATH, "w"), indent=2)

    def submit(self, talk_id: str, talk_title: str,
               stars: Optional[int] = None,
               tags: Optional[List[str]] = None,
               note: str = "",
               rec_helpful: Optional[bool] = None,
               report: Optional[str] = None) -> Dict:
        tid = str(talk_id)
        e   = self.entries.get(tid, {
            "talk_id": tid, "talk_title": talk_title,
            "stars": None, "tags": [], "note": "",
            "rec_helpful": None, "report": None, "ts": 0,
        })
        if stars is not None:       e["stars"]       = stars
        if tags:                    e["tags"]         = list(dict.fromkeys(e["tags"] + tags))
        if note:                    e["note"]         = note[:500]
        if rec_helpful is not None: e["rec_helpful"]  = rec_helpful
        if report:                  e["report"]       = report
        e["ts"] = time.time()
        self.entries[tid] = e
        self._save()
        return {"action": "updated"}

    def get(self, talk_id: str) -> Optional[Dict]:
        return self.entries.get(str(talk_id))

    def delete(self, talk_id: str):
        self.entries.pop(str(talk_id), None)
        self._save()

    def ml_rating(self, talk_id: str) -> Optional[float]:
        e = self.entries.get(str(talk_id))
        if not e: return None
        signals = []
        if e.get("stars"):
            signals.append(STAR_TO_RATING.get(e["stars"], 0.5))
        for tag in e.get("tags", []):
            meta = REACTION_TAGS.get(tag)
            if meta: signals.append(max(0.0, meta["sentiment"]))
        return sum(signals) / len(signals) if signals else None

    def apply_to_profile(self, profile: UserProfile) -> int:
        applied = 0
        for tid in self.entries:
            r = self.ml_rating(tid)
            if r is not None:
                existing = profile.ratings.get(tid, 0.5)
                if abs(r - 0.5) > abs(existing - 0.5):
                    profile.ratings[tid] = r
                    applied += 1
        if applied: profile.save()
        return applied

    def summary(self) -> Dict:
        all_e = list(self.entries.values())
        if not all_e:
            return {"total": 0, "avg_stars": None,
                    "helpful_pct": None, "top_tags": [], "flagged": 0}
        stars   = [e["stars"] for e in all_e if e.get("stars")]
        helpful = [e["rec_helpful"] for e in all_e
                   if e.get("rec_helpful") is not None]
        tag_counts: Dict[str, int] = defaultdict(int)
        for e in all_e:
            for t in e.get("tags", []): tag_counts[t] += 1
        return {
            "total":       len(all_e),
            "avg_stars":   round(sum(stars) / len(stars), 1) if stars else None,
            "helpful_pct": round(sum(helpful) / len(helpful) * 100) if helpful else None,
            "top_tags":    [{"tag": t, "count": c}
                            for t, c in sorted(tag_counts.items(),
                                               key=lambda x: -x[1])[:8]],
            "flagged":     sum(1 for e in all_e if e.get("report")),
        }

    def all(self, limit: int = 100) -> List[Dict]:
        return sorted(self.entries.values(),
                      key=lambda e: -e.get("ts", 0))[:limit]
