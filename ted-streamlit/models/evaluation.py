"""
evaluation.py — Recommendation quality metrics.

Metrics:
  Precision@K    — fraction of top-K recs the user liked
  Recall@K       — fraction of liked talks that appeared in top-K
  NDCG@K         — ranking quality (liked talks ranked higher = better)
  Coverage       — what % of catalogue ever gets recommended
  Diversity      — how different the recommendations are from each other
  Serendipity    — surprising yet relevant recommendations
  Feedback rate  — % of rated recs that got thumbs-up
  Baseline       — vs random and vs popularity recommender
"""

import math, random
from typing import List, Dict, Set


def precision_at_k(rec_ids: List[str], liked_ids: Set[str], k: int = 5) -> float:
    """Of top-K recommendations, fraction the user actually liked. 0-1."""
    if not rec_ids or not liked_ids:
        return 0.0
    top = rec_ids[:k]
    return round(sum(1 for t in top if t in liked_ids) / len(top), 4)


def recall_at_k(rec_ids: List[str], liked_ids: Set[str], k: int = 5) -> float:
    """Of all liked talks, fraction that appeared in top-K. 0-1."""
    if not rec_ids or not liked_ids:
        return 0.0
    top = rec_ids[:k]
    return round(sum(1 for t in top if t in liked_ids) / len(liked_ids), 4)


def ndcg_at_k(rec_ids: List[str], ratings: Dict[str, float], k: int = 5) -> float:
    """
    Normalised Discounted Cumulative Gain.
    Measures ranking quality — liked talks appearing earlier score higher.
    0 = worst ranking, 1 = perfect ranking.
    """
    if not rec_ids or not ratings:
        return 0.0
    top  = rec_ids[:k]
    dcg  = sum(ratings.get(t, 0.0) / math.log2(i + 2) for i, t in enumerate(top))
    rels = sorted([ratings.get(t, 0.0) for t in top], reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels) if r > 0)
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def coverage(history: List[str], catalogue_size: int) -> float:
    """% of catalogue that has ever been recommended. 0-1."""
    if not catalogue_size:
        return 0.0
    return round(len(set(history)) / catalogue_size, 4)


def diversity(rec_talks: List[Dict]) -> float:
    """
    Average pairwise tag dissimilarity across recommendations.
    0 = all same topic, 1 = all completely different topics.
    """
    if len(rec_talks) < 2:
        return 0.0
    tag_sets = [set(t.get("tags", [])) for t in rec_talks]
    pairs, total = 0, 0.0
    for i in range(len(tag_sets)):
        for j in range(i + 1, len(tag_sets)):
            a, b = tag_sets[i], tag_sets[j]
            if a or b:
                sim = len(a & b) / len(a | b) if (a | b) else 1.0
                total += 1.0 - sim
            pairs += 1
    return round(total / pairs, 4) if pairs else 0.0


def serendipity(rec_talks: List[Dict],
                profile_topics: Set[str],
                liked_ids: Set[str]) -> float:
    """
    Fraction of recommendations that are surprising (outside usual interests)
    yet relevant (user liked them). 0-1.
    """
    if not rec_talks:
        return 0.0
    count = 0
    for t in rec_talks:
        tags = set(t.get("tags", []))
        overlap = len(tags & profile_topics) / max(1, len(tags))
        if overlap < 0.3 and str(t["id"]) in liked_ids:
            count += 1
    return round(count / len(rec_talks), 4)


def feedback_rate(entries: List[Dict]) -> float:
    """% of rated recommendations that got thumbs-up."""
    rated = [e for e in entries if e.get("rec_helpful") is not None]
    if not rated:
        return 0.0
    pos = sum(1 for e in rated if e["rec_helpful"] is True)
    return round(pos / len(rated), 4)


def random_baseline(talks: List[Dict], n: int, exclude: Set[str]) -> List[str]:
    pool = [str(t["id"]) for t in talks if str(t["id"]) not in exclude]
    return random.sample(pool, min(n, len(pool)))


def popularity_baseline(talks: List[Dict], n: int, exclude: Set[str]) -> List[str]:
    pool = [t for t in talks if str(t["id"]) not in exclude]
    pool.sort(key=lambda t: t.get("views", 0), reverse=True)
    return [str(t["id"]) for t in pool[:n]]


class Evaluator:
    """Computes all evaluation metrics. One instance per app session."""

    def __init__(self):
        self.rec_history: List[str]  = []   # all ever-recommended IDs
        self.last_recs:   List[Dict] = []   # current recommendations

    def record(self, talks: List[Dict]):
        """Call after every recommend() call."""
        ids = [str(t["id"]) for t in talks]
        self.rec_history.extend(ids)
        self.last_recs = talks

    def report(self, profile, all_talks: List[Dict],
               feedback_entries: List[Dict], k: int = 5) -> Dict:
        """Full metrics report dict."""
        liked    = {tid for tid, r in profile.ratings.items() if r >= 0.6}
        exclude  = set(profile.watched) | set(profile.skipped)
        pos_topics = set(profile.positive_topics())
        rec_ids  = [str(t["id"]) for t in self.last_recs]
        n_watched = len(profile.watched)
        enough   = n_watched >= 5

        p5   = precision_at_k(rec_ids, liked, k)
        r5   = recall_at_k(rec_ids, liked, k)
        n5   = ndcg_at_k(rec_ids, profile.ratings, k)
        cov  = coverage(self.rec_history, len(all_talks))
        div  = diversity(self.last_recs)
        ser  = serendipity(self.last_recs, pos_topics, liked)
        fb   = feedback_rate(feedback_entries)

        # Baseline comparison
        baselines = None
        if liked:
            rand_ids = random_baseline(all_talks, k, exclude)
            pop_ids  = popularity_baseline(all_talks, k, exclude)
            baselines = {
                "ml":         {"precision": p5, "ndcg": n5},
                "random":     {"precision": precision_at_k(rand_ids, liked, k),
                               "ndcg":      ndcg_at_k(rand_ids, profile.ratings, k)},
                "popularity": {"precision": precision_at_k(pop_ids, liked, k),
                               "ndcg":      ndcg_at_k(pop_ids, profile.ratings, k)},
            }

        return {
            "enough_data":   enough,
            "n_watched":     n_watched,
            "n_liked":       len(liked),
            "precision_at_k": p5,
            "recall_at_k":    r5,
            "ndcg_at_k":      n5,
            "coverage":       cov,
            "diversity":      div,
            "serendipity":    ser,
            "feedback_rate":  fb,
            "baselines":      baselines,
            "k":              k,
        }


# ── EvaluationEngine — full report used by app.py ─────────────────

class EvaluationEngine:
    """Full evaluation report matching app.py's expected interface."""

    def __init__(self, profile, talks: List[Dict]):
        self.profile = profile
        self.talks   = talks
        self._rec_history: List[str] = []

    def full_report(self, recs: List[Dict],
                    tfidf_scores: Dict[str, float],
                    feedback_store) -> Dict:
        profile = self.profile
        talks   = self.talks

        liked_ids    = {tid for tid, r in profile.ratings.items() if r >= 0.6}
        pos_topics   = set(profile.positive_topics())
        rec_ids      = [str(t["id"]) for t in recs]
        self._rec_history.extend(rec_ids)
        exclude      = set(profile.watched) | set(profile.skipped)

        # Core metrics
        p5   = precision_at_k(rec_ids, liked_ids, k=5)
        cov  = coverage(self._rec_history, len(talks))
        div  = diversity(recs)
        ser  = serendipity(recs, pos_topics, liked_ids)
        fb   = feedback_rate(feedback_store.all(limit=200)
                             if hasattr(feedback_store, "all") else [])

        # Novelty: avg inverse log popularity of recommended talks
        nov = 0.0
        if recs:
            nov_scores = []
            for t in recs:
                exp = sum(profile.topic_exposure.get(tag, 0)
                          for tag in t.get("tags", []))
                nov_scores.append(1.0 / (1.0 + math.log(1 + exp)))
            nov = round(float(sum(nov_scores) / len(nov_scores)), 4)

        # Topic spread
        all_tags = set()
        for t in recs:
            all_tags.update(t.get("tags", []))
        topic_spread = len(all_tags)

        # Baseline comparison
        rand_ids = random_baseline(talks, 6, exclude)
        pop_ids  = popularity_baseline(talks, 6, exclude)

        def _nov(ids):
            ts = [t for t in talks if str(t["id"]) in set(ids)]
            if not ts: return 0.0
            scores = []
            for t in ts:
                exp = sum(profile.topic_exposure.get(tag, 0)
                          for tag in t.get("tags", []))
                scores.append(1.0 / (1.0 + math.log(1 + exp)))
            return round(float(sum(scores)/len(scores)), 4)

        rand_talks = [t for t in talks if str(t["id"]) in set(rand_ids)]
        pop_talks  = [t for t in talks if str(t["id"]) in set(pop_ids)]

        baseline = {
            "ml":     {"label": "Your ML Model",
                       "diversity": div, "novelty": nov, "serendipity": ser},
            "random": {"label": "Random",
                       "diversity": diversity(rand_talks),
                       "novelty":   _nov(rand_ids),
                       "serendipity": serendipity(rand_talks, pos_topics, liked_ids)},
            "popular":{"label": "Most Popular",
                       "diversity": diversity(pop_talks),
                       "novelty":   _nov(pop_ids),
                       "serendipity": serendipity(pop_talks, pos_topics, liked_ids)},
        }

        return {
            "coverage":        cov,
            "diversity":       div,
            "novelty":         nov,
            "serendipity":     ser,
            "topic_spread":    topic_spread,
            "precision_at_5":  p5 if liked_ids else None,
            "feedback_rate":   fb if fb > 0 else None,
            "n_watched":       len(profile.watched),
            "n_skipped":       len(profile.skipped),
            "baseline":        baseline,
        }
