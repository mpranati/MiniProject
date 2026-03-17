"""
app.py — TED Talks Recommender
Run: streamlit run app.py
"""
import sys, os, re, datetime
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as components
from backend.ted_fetcher import fetch_talks, search_talks, refresh as refresh_cache
from models.engine import (
    UserProfile, MLEngine, FeedbackStore,
    ReverseLearner, ScoringEngine, TrainingLog,
)
from models.evaluation import Evaluator, EvaluationEngine

st.set_page_config(page_title="TED Talks Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
  .stApp { background: #f5f5f5 !important; }
  header[data-testid="stHeader"] {
    background: #fff !important; border-bottom: 1px solid #ddd !important; }
  header[data-testid="stHeader"] * { color: #111 !important; }
  [data-testid="stSidebar"] { background: #fff !important; border-right: 1px solid #ddd; }
  [data-testid="stSidebar"] * { color: #111 !important; }
  .stApp, .stApp p, .stApp span, .stApp div,
  .stApp label, .stApp li { color: #111 !important; }
  [data-testid="stMetricLabel"] p { color: #555 !important; font-size: 12px !important; }
  [data-testid="stMetricValue"]   { color: #111 !important; }
  .stTabs [data-baseweb="tab"] { color: #555 !important; font-size: 13px; }
  .stTabs [aria-selected="true"] { color: #111 !important; font-weight: 600; }
  .tcard {
    background: #fff; border: 1px solid #ddd;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px; }
  .tcard-title { font-size: 15px; font-weight: 700; margin-bottom: 2px; }
  .tcard-title a { color: #c0392b !important; text-decoration: none; }
  .tcard-title a:hover { text-decoration: underline; }
  .tcard-speaker { font-size: 12px; color: #e74c3c !important; font-weight: 500; margin-bottom: 6px; }
  .tcard-desc { font-size: 12px; color: #444 !important; line-height: 1.5; margin-bottom: 8px; }
  .tag { display:inline-block; background:#f0f0f0; border:1px solid #ddd;
         border-radius:10px; padding:2px 8px; font-size:11px; color:#555 !important;
         margin:2px 2px 0 0; }
  .scores { display:flex; gap:6px; flex-wrap:wrap; margin:6px 0; }
  .spill  { background:#f8f8f8; border:1px solid #ddd; border-radius:5px;
            padding:3px 9px; font-size:11px; display:inline-flex; gap:4px; }
  .spill-lbl { color:#888 !important; }
  .s-hi { color:#1a7a40 !important; font-weight:700; }
  .s-md { color:#8a6000 !important; font-weight:700; }
  .s-lo { color:#b52b1e !important; font-weight:700; }
  .warn { background:#fffbea; border:1px solid #e8d44d;
          border-left:3px solid #d4a017; border-radius:6px;
          padding:10px 14px; margin-bottom:10px;
          font-size:12px; color:#4a3800 !important; }
  .div { border:none; border-top:1px solid #e5e5e5; margin:6px 0 10px; }
  .stTextInput input, .stTextArea textarea {
    background:#fff !important; color:#111 !important; border:1px solid #ccc !important; }
  .stTextInput input::placeholder, .stTextArea textarea::placeholder { color:#aaa !important; }
  .stButton > button {
    background:#fff !important; color:#111 !important; border:1px solid #ccc !important; }
  .stButton > button:hover { background:#f5f5f5 !important; }
</style>
""", unsafe_allow_html=True)


# ── Intent map ────────────────────────────────────────────────────

INTENT_MAP = {
    "lost":["purpose","motivation","meaning","identity"],
    "confused":["clarity","decision","thinking","philosophy"],
    "stuck":["creativity","motivation","change","growth"],
    "anxious":["health","psychology","stress"],
    "sad":["happiness","resilience","empathy"],
    "lonely":["relationships","connection","community"],
    "bored":["creativity","curiosity","learning"],
    "overwhelmed":["stress","productivity","focus"],
    "scared":["courage","resilience","psychology"],
    "unmotivated":["motivation","purpose","habits"],
    "tired":["health","sleep","wellbeing"],
    "curious":["science","learning","philosophy"],
    "meaning":["purpose","philosophy","happiness"],
    "purpose":["purpose","motivation","meaning","career"],
    "career":["career","leadership","creativity","business"],
    "money":["economics","finance","happiness"],
    "future":["technology","ai","environment","innovation"],
    "change":["growth","resilience","motivation"],
    "love":["relationships","psychology","vulnerability"],
    "work":["productivity","leadership","creativity","business"],
    "fail":["resilience","growth","courage","motivation"],
    "success":["leadership","motivation","business"],
    "learn":["education","neuroscience","skills"],
    "why":["philosophy","science","psychology"],
    "how":["education","science","technology"],
}

STOPWORDS = {
    "i","me","my","the","a","an","is","are","was","were","be","been","have","has",
    "do","does","did","will","would","could","should","can","to","of","in","on",
    "at","for","with","about","as","by","from","or","and","but","if","so","that",
    "this","it","he","she","they","we","you","not","no","very","just","feel",
    "feeling","am","im","dont","cant","really","always","never","want","like",
    "know","think","get","make","go","going","something","anything","everyone",
}

TOPICS = [
    "psychology","creativity","science","technology","ai","education",
    "happiness","leadership","health","philosophy","environment","economics",
    "art","design","neuroscience","relationships","language","consciousness",
    "motivation","storytelling","business","innovation",
]


def parse_intent(text):
    lower    = text.lower().strip()
    words    = re.findall(r"[a-z']+", lower)
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    topics   = []
    for trigger, t_list in INTENT_MAP.items():
        if trigger in lower:
            topics.extend(t_list)
    topics.extend(keywords)
    seen, deduped = set(), []
    for t in topics:
        if t not in seen: seen.add(t); deduped.append(t)
    neg_words = {"sad","lost","anxious","stuck","scared","overwhelmed","tired","lonely","fail"}
    sentiment = "negative" if any(w in neg_words for w in words) else "neutral"
    return {"keywords": keywords[:8], "topics": deduped[:12], "sentiment": sentiment}


def smart_search(text, talks, top_n=8):
    intent = parse_intent(text)
    kws    = set(intent["keywords"])
    topics = set(intent["topics"])
    scored = []
    for talk in talks:
        score    = 0.0
        title    = talk.get("title","").lower()
        desc     = talk.get("description","").lower()
        tags     = [t.lower() for t in talk.get("tags",[])]
        combined = title + " " + desc + " " + " ".join(tags)
        for kw in kws:
            if kw in title:  score += 0.4
            elif kw in desc: score += 0.2
            for tag in tags:
                if kw in tag: score += 0.15
        for topic in topics:
            for tag in tags:
                if topic in tag or tag in topic: score += 0.25
            if topic in title: score += 0.3
            if topic in desc:  score += 0.1
        words = re.findall(r"[a-z]+", text.lower())
        for i in range(len(words)-1):
            if words[i]+" "+words[i+1] in combined: score += 0.35
        if score > 0:
            scored.append({**talk, "score": round(score,3)})
    scored.sort(key=lambda x: -x["score"])

    # Normalise scores to 0-1 so percentages never exceed 100%
    if scored:
        max_s = scored[0]["score"]
        if max_s > 1.0:
            scored = [{**t, "score": round(t["score"] / max_s, 3)} for t in scored]

    if not scored:
        scored = search_talks(" ".join(intent["keywords"][:3]) or text, limit=top_n)
    results   = scored[:top_n]
    top_score = results[0].get("score",0) if results else 0
    warnings  = []

    if not results:
        warnings.append(
            "🔍 **Not enough talks available on this topic.** "
            "The catalogue doesn't have sufficient content matching your input. "
            "Try broader keywords like *creativity*, *health*, or *leadership*."
        )
    elif len(results) < 3:
        warnings.append(
            f"⚠️ **Limited coverage — only {len(results)} talk(s) found.** "
            "The catalogue doesn't have much content on this specific topic. "
            "Results may not fully address what you're looking for."
        )
    elif top_score < 0.4:
        warnings.append(
            "⚠️ **Weak match.** No talks directly cover this topic. "
            "Showing the closest available results — they touch on related themes "
            "but may not be exactly what you need."
        )
    return results, warnings


# ── Session state — initialise once ──────────────────────────────
# Use setdefault so objects are created only on first run,
# then reused as-is on every rerun. This is the correct Streamlit pattern.

if "profile"  not in st.session_state: st.session_state.profile  = UserProfile()
if "engine"   not in st.session_state: st.session_state.engine   = MLEngine()
if "feedback" not in st.session_state: st.session_state.feedback = FeedbackStore()
if "talks"    not in st.session_state: st.session_state.talks    = []
if "ready"    not in st.session_state: st.session_state.ready    = False


# ── Always read from session_state directly — never cache locally ─
# This ensures every rerun sees the latest mutated state.

def profile():  return st.session_state.profile
def engine():   return st.session_state.engine
def feedback(): return st.session_state.feedback


def startup():
    if not st.session_state.ready:
        with st.spinner("Loading talks from TED…"):
            talks = fetch_talks(limit=500)
            engine().corpus = talks
            if not engine().load() or len(engine().ids) != len(talks):
                engine().train(talks)
            st.session_state.talks = talks
            st.session_state.ready = True


def find_talk(tid):
    return next((t for t in st.session_state.talks
                 if str(t["id"]) == str(tid)), None)


def score_cls(v):
    return "s-hi" if v >= 0.65 else ("s-md" if v >= 0.4 else "s-lo")


def score_pills(talk, show=True):
    if not show: return ""
    pills = []
    s  = min(1.0, talk.get("score", 0))
    bd = talk.get("score_breakdown") or {}
    if s:
        pills.append(f'<div class="spill"><span class="spill-lbl">Match</span>'
                     f'<span class="{score_cls(s)}">{round(s*100)}%</span></div>')
    for label, key in [("Relevance","relevance"),("Novelty","novelty"),("Honesty","honesty")]:
        v = bd.get(key)
        if v is not None:
            v = min(1.0, max(0.0, v))
            pills.append(f'<div class="spill"><span class="spill-lbl">{label}</span>'
                         f'<span class="{score_cls(v)}">{round(v*100)}%</span></div>')
    return '<div class="scores">'+"\n".join(pills)+"</div>" if pills else ""


def render_talk(talk, show_score=False, key_prefix=""):
    tid   = str(talk["id"])
    desc  = talk.get("description","")[:200]
    tags  = "".join(f'<span class="tag">{t}</span>' for t in talk.get("tags",[])[:4])
    found = find_talk(tid)  # resolve once, used by all buttons

    st.markdown(f"""
<div class="tcard">
  <div class="tcard-title"><a href="{talk.get('url','#')}" target="_blank">{talk['title']}</a></div>
  <div class="tcard-speaker">{talk.get('speaker','')}</div>
  {'<div class="tcard-desc">'+desc+'</div>' if desc else ''}
  {score_pills(talk, show_score)}
  <div style="margin-top:5px">{tags}</div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("✓ Watched", key=f"{key_prefix}_w_{tid}"):
            if found:
                profile().record_watch(found, rating=0.8)
                engine().on_interaction(found, profile(), liked=True)
                st.toast("Watched — model updated.")
                st.rerun()
    with c2:
        if st.button("Skip", key=f"{key_prefix}_s_{tid}"):
            if found:
                profile().record_skip(found)
                engine().on_interaction(found, profile(), liked=False)
                st.toast("Skipped — model updated.")
                st.rerun()
    with c3:
        with st.expander("Rate"):
            stars = st.feedback("stars", key=f"{key_prefix}_stars_{tid}")
            helpful = st.radio("Helpful?", ["👍 Yes","👎 No"],
                               index=None, horizontal=True,
                               key=f"{key_prefix}_h_{tid}",
                               label_visibility="collapsed")
            note = st.text_input("Note", key=f"{key_prefix}_n_{tid}",
                                 label_visibility="collapsed",
                                 placeholder="Optional note…")
            if st.button("Save", key=f"{key_prefix}_fb_{tid}"):
                if stars is not None or helpful or note.strip():
                    star_val = stars + 1 if stars is not None else None  # 0-4 → 1-5
                    # Map stars to rating: 1★=0.0 2★=0.25 3★=0.5 4★=0.75 5★=1.0
                    star_rating_map = {1:0.0, 2:0.25, 3:0.5, 4:0.75, 5:1.0}
                    if helpful == "👍 Yes" or (star_val and star_val >= 4):
                        profile().ratings[tid] = star_rating_map.get(star_val, 1.0) if star_val else 1.0
                        if found:
                            engine().on_interaction(found, profile(), liked=True)
                    elif helpful == "👎 No" or (star_val and star_val <= 2):
                        profile().ratings[tid] = star_rating_map.get(star_val, 0.1) if star_val else 0.1
                        if found:
                            engine().on_interaction(found, profile(), liked=False)
                    profile().save()
                    feedback().submit(
                        talk_id=tid, talk_title=talk["title"],
                        stars=star_val, tags=[], note=note.strip(),
                        rec_helpful=(True if helpful=="👍 Yes" else
                                     False if helpful=="👎 No" else None),
                        report=None,
                    )
                    feedback().apply_to_profile(profile())
                    st.toast("Feedback saved — model updated.")
                else:
                    st.warning("Add a star rating, thumbs, or note.")
    st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── Startup ───────────────────────────────────────────────────────
startup()


# ── Sidebar — reads fresh state on every rerun ────────────────────
with st.sidebar:
    st.markdown(
        '<h2 style="color:#e62b1e;font-weight:900;margin-bottom:14px">'
        '🎬 TED Talks Recommender</h2>',
        unsafe_allow_html=True
    )

    # Read stats fresh from session_state every time
    stats = profile().stats()

    for label, val in [
        ("Watched",  stats.get("watched_count",  0)),
        ("Topics",   stats.get("topics_explored", 0)),
        ("Speakers", stats.get("speakers_seen",   0)),
    ]:
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:7px 10px;background:#f5f5f5;border-radius:6px;
            border:1px solid #ddd;margin-bottom:5px">
  <span style="font-size:12px;color:#555">{label}</span>
  <span style="font-size:16px;font-weight:700;color:#111">{val}</span>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("**Interests**")
    current = [t for t in TOPICS if t in profile()._explicit_interests]
    sel = st.multiselect("Topics", TOPICS, default=current,
                         label_visibility="collapsed")
    if st.button("Apply", use_container_width=True):
        if sel:
            profile().set_interests(sel)
            st.success("Interests updated.")
            st.rerun()
        else:
            st.warning("Select at least one topic.")

    top_int = stats.get("top_interests", [])
    if top_int:
        st.divider()
        st.markdown("**Top Interests**")
        for topic, w in top_int[:5]:
            st.progress(float(w), text=topic)

    st.divider()
    if st.button("↺ Refresh Talks", use_container_width=True):
        refresh_cache()
        st.session_state.ready = False
        st.rerun()
    if st.button("Reset Profile", use_container_width=True, type="secondary"):
        profile().reset()
        st.session_state.profile = UserProfile()
        st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────
tabs = st.tabs(["✦ Ask","Discover","Browse","Search","Profile",
                "History","Feedback","📊 Evaluation"])
(tab_ask, tab_discover, tab_browse, tab_search,
 tab_profile, tab_history, tab_feedback, tab_eval) = tabs


# ── Ask ───────────────────────────────────────────────────────────
with tab_ask:
    st.subheader("Find talks for anything")
    st.caption("Type or speak a feeling, question, confusion, or keyword.")

    examples = [
        "I feel lost and don't know my purpose",
        "Why do I keep procrastinating?",
        "I want to be a better leader",
        "I'm anxious about the future",
        "How does the brain work?",
        "creativity and innovation",
        "I failed and don't know what to do",
        "I feel lonely even around people",
    ]

    # ── Voice search component ────────────────────────────────────
    voice_html = """
<div style="margin-bottom:10px">
  <button id="voiceBtn" onclick="toggleVoice()" style="
    display:inline-flex;align-items:center;gap:8px;
    padding:8px 16px;background:#fff;border:1px solid #ccc;
    border-radius:6px;cursor:pointer;font-size:13px;color:#111;
    font-family:sans-serif;transition:all .2s">
    🎙 Speak
  </button>
  <span id="voiceStatus" style="margin-left:10px;font-size:12px;color:#888;font-family:sans-serif"></span>
  <div id="voiceResult" style="margin-top:8px;padding:8px 12px;
    background:#f5f5f5;border:1px solid #ddd;border-radius:6px;
    font-size:13px;color:#111;font-family:sans-serif;display:none"></div>
</div>

<script>
let recognition = null;
let listening = false;

function toggleVoice() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    document.getElementById('voiceStatus').textContent = 'Voice not supported in this browser. Try Chrome.';
    return;
  }
  if (listening) {
    recognition.stop();
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  const btn    = document.getElementById('voiceBtn');
  const status = document.getElementById('voiceStatus');
  const result = document.getElementById('voiceResult');

  recognition.onstart = () => {
    listening = true;
    btn.textContent = '⏹ Stop';
    btn.style.borderColor = '#e74c3c';
    btn.style.color = '#e74c3c';
    status.textContent = 'Listening…';
    result.style.display = 'none';
  };

  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    result.style.display = 'block';
    result.textContent = '🎙 "' + transcript + '"';
    if (event.results[event.results.length-1].isFinal) {
      status.textContent = 'Got it — sending to search…';
      // Send to Streamlit via URL hash trick
      window.parent.postMessage({type: 'streamlit:setComponentValue', value: transcript}, '*');
      // Also try direct input injection into the textarea
      setTimeout(() => {
        const textareas = window.parent.document.querySelectorAll('textarea');
        textareas.forEach(ta => {
          if (ta.placeholder && ta.placeholder.includes('stuck')) {
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            nativeInputValueSetter.call(ta, transcript);
            ta.dispatchEvent(new Event('input', { bubbles: true }));
          }
        });
      }, 100);
    }
  };

  recognition.onerror = (event) => {
    status.textContent = 'Error: ' + event.error + '. Try again.';
    listening = false;
    btn.textContent = '🎙 Speak';
    btn.style.borderColor = '#ccc';
    btn.style.color = '#111';
  };

  recognition.onend = () => {
    listening = false;
    btn.textContent = '🎙 Speak';
    btn.style.borderColor = '#ccc';
    btn.style.color = '#111';
    if (status.textContent === 'Listening…') {
      status.textContent = 'Done. Paste the text above into the search box if needed.';
    }
  };

  recognition.start();
}
</script>
"""
    st.components.v1.html(voice_html, height=110)

    user_input = st.text_area("Input",
        value=st.session_state.get("ask_input",""),
        placeholder="e.g. I feel stuck and don't know what to do…",
        height=80, label_visibility="collapsed")

    st.caption("Examples:")
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i%4].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["ask_input"] = ex
            st.rerun()

    if user_input and user_input.strip():
        st.session_state["ask_input"] = user_input
        intent  = parse_intent(user_input)
        results, warnings = smart_search(user_input, st.session_state.talks)
        scorer  = ScoringEngine(profile())
        ts      = engine().tfidf_scores(profile())
        enriched = []
        for t in results:
            _, bd = scorer.composite(t, ts.get(str(t["id"]),0.5))
            enriched.append({**t, "score_breakdown": bd})

        kw = ", ".join(intent["keywords"][:6]) or "—"
        st.info(f"📌 Interpreted as: **{kw}**")
        for w in warnings:
            st.markdown(f'<div class="warn">⚠ {w}</div>', unsafe_allow_html=True)
        if enriched:
            st.caption(f"{len(enriched)} talks found")
            for talk in enriched:
                render_talk(talk, show_score=True, key_prefix="ask")


# ── Discover ──────────────────────────────────────────────────────
with tab_discover:
    st.subheader("Discover")

    n_watched = len(profile().watched)
    n_skipped = len(profile().skipped)
    n_total   = n_watched + n_skipped

    col_ref, col_info = st.columns([1,4])
    with col_ref:
        if st.button("↺ Refresh"):
            st.rerun()
    with col_info:
        if n_total == 0:
            st.caption("Set interests or watch/skip talks — scores will improve with each interaction.")
        elif n_total < 5:
            st.caption(f"🟡 {n_total} interactions so far — keep going, scores improve after 5+")
        else:
            st.caption(f"🟢 {n_total} interactions — model is actively learning your preferences.")

    recs = engine().recommend(profile(), n=6)
    if not recs:
        st.info("Set your interests or watch a few talks to get recommendations.")
    else:
        scorer = ScoringEngine(profile())
        ts     = engine().tfidf_scores(profile())
        for talk in recs:
            _, bd = scorer.composite(talk, ts.get(str(talk["id"]),0.5))
            render_talk({**talk,"score_breakdown":bd}, show_score=True, key_prefix="disc")


# ── Browse ────────────────────────────────────────────────────────
with tab_browse:
    st.subheader(f"Browse — {len(st.session_state.talks)} talks")
    page  = st.number_input("Page", min_value=1, value=1, step=1,
                            label_visibility="collapsed")
    per   = 10
    start = (page-1)*per
    for talk in st.session_state.talks[start:start+per]:
        render_talk(talk, key_prefix=f"br{page}")
    st.caption(f"Page {page} · {start+1}–{min(start+per,len(st.session_state.talks))} "
               f"of {len(st.session_state.talks)}")


# ── Search ────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Search")
    query = st.text_input("Search", label_visibility="collapsed",
                          placeholder="e.g. creativity, Brené Brown…")
    if query.strip():
        results = search_talks(query.strip(), limit=20)
        scorer  = ScoringEngine(profile())
        ts      = engine().tfidf_scores(profile())
        scored  = []
        for t in results:
            final, bd = scorer.composite(t, ts.get(str(t["id"]),0.5))
            scored.append({**t, "score":round(final,3), "score_breakdown":bd})
        scored.sort(key=lambda x: -x["score"])
        if scored:
            st.caption(f"{len(scored)} results")
            for talk in scored:
                render_talk(talk, show_score=True, key_prefix="srch")
        else:
            st.info("No results found.")


# ── Profile ───────────────────────────────────────────────────────
with tab_profile:
    st.subheader("Profile")
    stats  = profile().stats()
    pivots = ReverseLearner(profile()).suggested_pivots()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stats**")
        for label, val in [
            ("Watched",            stats.get("watched_count",0)),
            ("Topics explored",    stats.get("topics_explored",0)),
            ("Speakers seen",      stats.get("speakers_seen",0)),
            ("Total interactions", stats.get("total_interactions",0)),
        ]:
            st.write(f"{label}: **{val}**")
    with c2:
        st.markdown("**Topics by exposure**")
        topic_exp = profile().topic_exposure
        if topic_exp:
            for topic, count in sorted(topic_exp.items(), key=lambda x: -x[1])[:8]:
                st.write(f"{topic}: **{count}** talk{'s' if count != 1 else ''}")
        else:
            st.caption("Watch some talks to see your topic breakdown.")

    if pivots:
        st.markdown("**Suggested next topics**")
        st.write(", ".join(pivots))

    if stats.get("top_interests"):
        st.markdown("**Top Interests**")
        for topic, w in stats["top_interests"][:8]:
            st.progress(float(w), text=topic)


# ── Watch History ─────────────────────────────────────────────────
with tab_history:
    st.subheader("Watch History")

    watched  = profile().watched    # {tid: timestamp}
    ratings  = profile().ratings    # {tid: 0-1 float}
    skipped  = profile().skipped    # {tid: timestamp}
    talk_map = {str(t["id"]): t for t in st.session_state.talks}

    # Sort watched by most recent first
    watched_sorted = sorted(watched.items(), key=lambda x: -x[1])
    skipped_sorted = sorted(skipped.items(), key=lambda x: -x[1])

    if not watched_sorted and not skipped_sorted:
        st.info("No history yet. Watch or skip talks to build your history.")
    else:
        # Summary row
        c1, c2, c3 = st.columns(3)
        c1.metric("Watched", len(watched_sorted))
        c2.metric("Skipped", len(skipped_sorted))
        avg_r = sum(ratings.values()) / len(ratings) if ratings else 0
        c3.metric("Avg Rating", f"{round(avg_r * 5, 1)} / 5")

        st.divider()

        # Filter
        filter_opt = st.radio("Show", ["Watched", "Skipped", "All"],
                              horizontal=True, index=0,
                              label_visibility="collapsed")

        st.divider()

        # Watched talks
        if filter_opt in ("Watched", "All") and watched_sorted:
            if filter_opt == "All":
                st.markdown("**Watched**")

            for tid, ts in watched_sorted:
                talk    = talk_map.get(tid)
                title   = talk["title"] if talk else tid
                speaker = talk.get("speaker","") if talk else ""
                url     = talk.get("url","#") if talk else "#"
                tags    = talk.get("tags",[])[:3] if talk else []
                rating  = ratings.get(tid, 0.5)
                stars   = round(rating * 5)
                date    = datetime.datetime.fromtimestamp(ts).strftime("%b %d, %Y")
                star_str = "★" * stars + "☆" * (5 - stars)
                tags_html = "".join(
                    f'<span class="tag">{t}</span>' for t in tags)

                st.markdown(f"""
<div class="tcard" style="padding:12px 16px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div style="flex:1;min-width:0">
      <div class="tcard-title">
        <a href="{url}" target="_blank">{title}</a>
      </div>
      <div class="tcard-speaker">{speaker}</div>
      <div style="margin-top:4px">{tags_html}</div>
    </div>
    <div style="text-align:right;flex-shrink:0;margin-left:12px">
      <div style="color:#e5a000;font-size:14px">{star_str}</div>
      <div style="font-size:11px;color:#888;margin-top:2px">{date}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                # Remove from history button
                if st.button("Remove", key=f"rmwatch_{tid}"):
                    del profile().watched[tid]
                    if tid in profile().ratings:
                        del profile().ratings[tid]
                    profile().save()
                    st.toast("Removed from history.")
                    st.rerun()

                st.markdown('<hr class="div">', unsafe_allow_html=True)

        # Skipped talks
        if filter_opt in ("Skipped", "All") and skipped_sorted:
            if filter_opt == "All":
                st.markdown("**Skipped**")

            for tid, ts in skipped_sorted:
                talk    = talk_map.get(tid)
                title   = talk["title"] if talk else tid
                speaker = talk.get("speaker","") if talk else ""
                url     = talk.get("url","#") if talk else "#"
                tags    = talk.get("tags",[])[:3] if talk else []
                date    = datetime.datetime.fromtimestamp(ts).strftime("%b %d, %Y")
                tags_html = "".join(
                    f'<span class="tag">{t}</span>' for t in tags)

                st.markdown(f"""
<div class="tcard" style="padding:12px 16px;border-left:3px solid #ddd">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div style="flex:1;min-width:0">
      <div class="tcard-title">
        <a href="{url}" target="_blank">{title}</a>
      </div>
      <div class="tcard-speaker">{speaker}</div>
      <div style="margin-top:4px">{tags_html}</div>
    </div>
    <div style="text-align:right;flex-shrink:0;margin-left:12px">
      <div style="font-size:11px;color:#aaa">skipped</div>
      <div style="font-size:11px;color:#888;margin-top:2px">{date}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                if st.button("Remove", key=f"rmskip_{tid}"):
                    del profile().skipped[tid]
                    profile().save()
                    st.toast("Removed from skipped.")
                    st.rerun()

                st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── Feedback ──────────────────────────────────────────────────────
with tab_feedback:
    st.subheader("Feedback")
    summary = feedback().summary()
    entries = feedback().all(limit=50)

    if summary.get("total",0):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rated",   summary.get("total",0))
        c2.metric("Avg ★",   summary.get("avg_stars") or "—")
        c3.metric("Helpful", f"{summary['helpful_pct']}%"
                  if summary.get("helpful_pct") is not None else "—")
        c4.metric("Flagged", summary.get("flagged",0))
        st.divider()

    if not entries:
        st.info("No feedback yet. Click 'Rate' on any talk.")
    else:
        for e in entries:
            stars_val = e.get("stars")
            stars_str = "★" * stars_val + "☆" * (5 - stars_val) if stars_val else ""
            thumb = ("👍" if e.get("rec_helpful") is True else
                     "👎" if e.get("rec_helpful") is False else "")
            ts   = e.get("ts")
            date = datetime.datetime.fromtimestamp(ts).strftime("%b %d, %Y") if ts else ""
            ct, cb = st.columns([1,10])
            ct.markdown(
                f"<div style='font-size:1.1rem;color:#e5a000'>{stars_str}</div>"
                if stars_str else
                f"<span style='font-size:1.4rem'>{thumb or '—'}</span>",
                unsafe_allow_html=True)
            with cb:
                st.markdown(f"**{e.get('talk_title') or e.get('talk_id','')}**")
                meta = " · ".join(filter(None, [date, thumb]))
                if meta: st.caption(meta)
                if e.get("note"): st.caption(f'"{e["note"]}"')
                if st.button("Remove", key=f"delfb_{e['talk_id']}"):
                    feedback().delete(e["talk_id"])
                    st.rerun()
            st.markdown('<hr class="div">', unsafe_allow_html=True)


# ── Evaluation ────────────────────────────────────────────────────
with tab_eval:
    st.subheader("📊 Model Evaluation")
    st.caption("How well is the recommender working for you?")

    if st.button("Run Evaluation"):
        recs = engine().recommend(profile(), n=6)
        if not recs:
            st.warning("No recommendations yet — set interests or watch some talks first.")
        else:
            evaluator = EvaluationEngine(profile(), st.session_state.talks)
            report    = evaluator.full_report(recs, engine().tfidf_scores(profile()), feedback())

            st.divider()
            st.markdown("**Recommendation Quality**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Coverage",
                      f"{round(report['coverage']*100)}%",
                      help="% of catalogue ever recommended to you")
            c2.metric("Diversity",
                      f"{round(report['diversity']*100)}%",
                      help="How different the 6 recommendations are from each other")
            c3.metric("Novelty",
                      f"{round(report['novelty']*100)}%",
                      help="How unfamiliar the recommended topics are to you")

            c4, c5, c6 = st.columns(3)
            c4.metric("Serendipity",
                      f"{round(report['serendipity']*100)}%",
                      help="Surprising picks outside your usual topics that you still liked")
            c5.metric("Feedback Rate",
                      f"{round(report['feedback_rate']*100)}%" if report['feedback_rate'] else "No feedback yet",
                      help="% of rated talks you found helpful")
            c6.metric("Interactions",
                      report["n_watched"] + report["n_skipped"],
                      help="Total watches + skips the model has trained on")

            st.divider()
            st.markdown("**What each metric means**")
            for metric, explanation in [
                ("Coverage",      "What % of the catalogue has been recommended. Low means the model keeps showing the same small set of talks."),
                ("Diversity",     "How different the recommendations are from each other. Low means a filter bubble — all on the same topic."),
                ("Novelty",       "How unfamiliar the recommended topics are to you. Low means the model is being too repetitive."),
                ("Serendipity",   "Talks outside your usual interests that you still enjoyed. Hard to increase — requires exploring new topics."),
                ("Feedback Rate", "% of talks you rated that you found helpful. The most direct signal of recommendation quality."),
            ]:
                st.markdown(f"**{metric}** — {explanation}")
    else:
        st.info("Click 'Run Evaluation' to measure recommendation quality.")
