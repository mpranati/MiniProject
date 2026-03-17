"""
ted_fetcher.py — Fetches TED talks.

Priority:
  1. YouTube Data API v3   — 500+ talks with rich metadata (needs free API key)
  2. Multiple TED RSS feeds — ~300-400 talks combined (no key needed)
  3. Single RSS feed        — ~50 recent talks
  4. Seed data              — 25 classics (offline fallback)

YouTube API setup (free, 5 minutes):
  1. Go to https://console.cloud.google.com
  2. New project → Enable "YouTube Data API v3"
  3. Credentials → Create API Key
  4. Add to .env:  YOUTUBE_API_KEY=your_key_here
"""

import os, re, time, logging
from typing import List, Dict, Optional

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TED_CHANNEL_ID  = "UCAuUUnT6oDeKwE6v1NGQxug"
YOUTUBE_API     = "https://www.googleapis.com/youtube/v3"
CACHE_TTL       = 3600

TOPIC_RSS_FEEDS = [
    "https://www.ted.com/talks/rss",
    "https://www.ted.com/topics/technology/rss",
    "https://www.ted.com/topics/psychology/rss",
    "https://www.ted.com/topics/science/rss",
    "https://www.ted.com/topics/health/rss",
    "https://www.ted.com/topics/education/rss",
    "https://www.ted.com/topics/creativity/rss",
    "https://www.ted.com/topics/business/rss",
    "https://www.ted.com/topics/leadership/rss",
    "https://www.ted.com/topics/happiness/rss",
    "https://www.ted.com/topics/motivation/rss",
    "https://www.ted.com/topics/philosophy/rss",
    "https://www.ted.com/topics/environment/rss",
    "https://www.ted.com/topics/ai/rss",
    "https://www.ted.com/topics/neuroscience/rss",
    "https://www.ted.com/topics/design/rss",
    "https://www.ted.com/topics/economics/rss",
    "https://www.ted.com/topics/relationships/rss",
    "https://www.ted.com/topics/storytelling/rss",
    "https://www.ted.com/topics/innovation/rss",
    "https://www.ted.com/topics/consciousness/rss",
    "https://www.ted.com/topics/language/rss",
    "https://www.ted.com/topics/art/rss",
]

_cache: Dict = {}
_cache_time: float = 0


# ── Helpers ───────────────────────────────────────────────────────

def _clean(html: str) -> str:
    if not html: return ""
    if "<" in html:
        html = BeautifulSoup(html, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", html).strip()


def _make_talk(id_, title, speaker, description, tags, url,
               views=0, duration=0, published_at="") -> Dict:
    return {
        "id":           str(id_),
        "title":        title.strip(),
        "speaker":      speaker.strip(),
        "description":  description[:600].strip(),
        "tags":         [t.lower().strip() for t in tags if t.strip()],
        "url":          url,
        "views":        int(views or 0),
        "duration":     int(duration or 0),
        "published_at": published_at,
    }


# ── Source 1: YouTube Data API ────────────────────────────────────

def _parse_iso_duration(s: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m: return 0
    h, mi, sec = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + sec


def _fetch_youtube(limit: int = 500) -> List[Dict]:
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key:
        return []

    talks     = []
    page_token = None
    fetched    = 0
    max_pages  = limit // 50 + 1

    try:
        # Step 1: get video IDs from channel
        video_ids = []
        for _ in range(max_pages):
            params = {
                "key": key, "channelId": TED_CHANNEL_ID,
                "part": "id", "type": "video",
                "maxResults": 50, "order": "date",
            }
            if page_token:
                params["pageToken"] = page_token

            r = requests.get(f"{YOUTUBE_API}/search", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()

            ids = [item["id"]["videoId"] for item in data.get("items", [])
                   if item.get("id", {}).get("videoId")]
            video_ids.extend(ids)
            page_token = data.get("nextPageToken")
            if not page_token or len(video_ids) >= limit:
                break

        if not video_ids:
            return []

        # Step 2: get full video details in batches of 50
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            params = {
                "key": key,
                "id": ",".join(batch),
                "part": "snippet,contentDetails,statistics",
            }
            r = requests.get(f"{YOUTUBE_API}/videos", params=params, timeout=15)
            r.raise_for_status()
            items = r.json().get("items", [])

            for item in items:
                try:
                    snip    = item.get("snippet", {})
                    details = item.get("contentDetails", {})
                    stats   = item.get("statistics", {})

                    title   = snip.get("title", "")
                    desc    = snip.get("description", "")[:600]
                    tags    = snip.get("tags", [])
                    pub     = snip.get("publishedAt", "")[:10]
                    vid_id  = item.get("id", "")
                    views   = int(stats.get("viewCount", 0) or 0)
                    dur     = _parse_iso_duration(details.get("duration", ""))

                    # Extract speaker from title — TED format: "Title | Speaker | TED"
                    speaker = ""
                    parts   = [p.strip() for p in title.split("|")]
                    if len(parts) >= 2:
                        # Last meaningful part before "TED" is usually speaker
                        for p in reversed(parts):
                            if p.lower() not in ("ted", "tedx", "ted talk", ""):
                                speaker = p
                                break
                        title = parts[0]

                    # Auto-generate topic tags from title/desc if YouTube tags sparse
                    TOPIC_KEYWORDS = {
                        "psychology": ["psychology","mind","behavior","mental","emotion","cognitive"],
                        "creativity": ["creativity","creative","art","design","imagination"],
                        "science":    ["science","research","study","experiment","discovery"],
                        "technology": ["technology","tech","digital","software","computer","internet"],
                        "ai":         ["artificial intelligence","machine learning","ai","algorithm"],
                        "education":  ["education","learning","school","teaching","student"],
                        "happiness":  ["happiness","happy","joy","wellbeing","flourish"],
                        "leadership": ["leadership","leader","manage","team","organization"],
                        "health":     ["health","medical","medicine","body","brain","disease"],
                        "philosophy": ["philosophy","meaning","ethics","moral","existence"],
                        "environment":["environment","climate","nature","sustainability","planet"],
                        "economics":  ["economics","economy","money","finance","inequality"],
                        "motivation": ["motivation","inspire","purpose","goal","grit","passion"],
                        "relationships":["relationship","love","connection","family","community"],
                        "neuroscience":["neuroscience","brain","neuron","cognition","memory"],
                        "language":   ["language","linguistics","words","speech","communication"],
                        "storytelling":["story","storytelling","narrative","fiction","culture"],
                        "business":   ["business","startup","entrepreneur","company","work"],
                        "innovation": ["innovation","future","change","disruption","invention"],
                    }
                    combined = (title + " " + desc).lower()
                    for topic, keywords in TOPIC_KEYWORDS.items():
                        if any(kw in combined for kw in keywords):
                            if topic not in tags:
                                tags.append(topic)

                    url = f"https://www.youtube.com/watch?v={vid_id}"
                    talks.append(_make_talk(
                        vid_id, title, speaker, desc,
                        tags, url, views, dur, pub
                    ))
                except Exception:
                    continue

        logger.info(f"YouTube API: fetched {len(talks)} talks")
        return talks

    except requests.HTTPError as e:
        if e.response.status_code == 403:
            logger.warning("YouTube API: quota exceeded or invalid key")
        else:
            logger.warning(f"YouTube API error: {e}")
        return []
    except Exception as e:
        logger.warning(f"YouTube fetch failed: {e}")
        return []


# ── Source 2: Multiple TED RSS feeds ─────────────────────────────

TOPIC_KEYWORDS = {
    "psychology":    ["psychology","mind","behavior","mental","emotion","cognitive","personality"],
    "creativity":    ["creativity","creative","art","imagination","design","innovation"],
    "science":       ["science","research","study","experiment","discovery","biology","physics","chemistry"],
    "technology":    ["technology","tech","digital","software","computer","internet","data"],
    "ai":            ["artificial intelligence","machine learning","ai","algorithm","robot"],
    "education":     ["education","learning","school","teaching","student","university","knowledge"],
    "happiness":     ["happiness","happy","joy","wellbeing","flourish","positive","life satisfaction"],
    "leadership":    ["leadership","leader","manage","team","organization","inspire","vision"],
    "health":        ["health","medical","medicine","body","disease","fitness","nutrition","brain"],
    "philosophy":    ["philosophy","meaning","ethics","moral","existence","consciousness","truth"],
    "environment":   ["environment","climate","nature","sustainability","planet","ecology","green"],
    "economics":     ["economics","economy","money","finance","inequality","poverty","wealth","market"],
    "motivation":    ["motivation","inspire","purpose","goal","grit","passion","drive","success"],
    "relationships": ["relationship","love","connection","family","community","friendship","marriage"],
    "neuroscience":  ["neuroscience","brain","neuron","cognition","memory","perception","neural"],
    "language":      ["language","linguistics","words","speech","communication","writing","reading"],
    "storytelling":  ["story","storytelling","narrative","fiction","culture","history","memoir"],
    "business":      ["business","startup","entrepreneur","company","work","career","management"],
    "innovation":    ["innovation","future","change","disruption","invention","progress"],
    "consciousness": ["consciousness","awareness","self","identity","mind","perception","meditation"],
}

def _infer_tags(text: str) -> list:
    """Infer topic tags from text when none are provided."""
    text_lower = text.lower()
    matched = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(topic)
    return matched[:5]


def _normalize_rss(entry) -> Optional[Dict]:
    try:
        title   = entry.get("title", "")
        speaker = ""
        if ":" in title:
            speaker, title = title.split(":", 1)
            speaker, title = speaker.strip(), title.strip()

        desc = _clean(entry.get("summary", "") or entry.get("description", ""))
        link = entry.get("link", "")
        slug = link.rstrip("/").split("/")[-1].split("?")[0]
        tags = [t.term for t in getattr(entry, "tags", []) if hasattr(t, "term")]

        # Auto-generate topic tags from title+desc if RSS gives none
        if not tags:
            tags = _infer_tags(title + " " + desc)

        return _make_talk(
            slug or str(abs(hash(title)))[:10],
            title, speaker, desc, tags, link,
            published_at=entry.get("published", "")
        )
    except Exception:
        return None


def _fetch_multi_rss(limit: int = 500) -> List[Dict]:
    seen  = {}
    total = 0
    for feed_url in TOPIC_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                t = _normalize_rss(entry)
                if t and t["id"] and t["id"] not in seen:
                    seen[t["id"]] = t
                    total += 1
            if total >= limit:
                break
        except Exception as e:
            logger.debug(f"RSS feed failed {feed_url}: {e}")
            continue
    logger.info(f"Multi-RSS: fetched {len(seen)} unique talks")
    return list(seen.values())


def _fetch_single_rss(limit: int = 50) -> List[Dict]:
    try:
        feed  = feedparser.parse("https://www.ted.com/feeds/talks.rss")
        talks = [t for t in (_normalize_rss(e) for e in feed.entries[:limit]) if t]
        logger.info(f"Single RSS: fetched {len(talks)} talks")
        return talks
    except Exception as e:
        logger.warning(f"Single RSS failed: {e}")
        return []


# ── Source 3: Seed data ───────────────────────────────────────────

def _seed_talks() -> List[Dict]:
    return [
        _make_talk("1","Do schools kill creativity?","Ken Robinson","Ken Robinson makes a case for creating an education system that nurtures creativity.",["creativity","education","culture"],"https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity",72000000,1194,"2006-06-27"),
        _make_talk("2","Your body language may shape who you are","Amy Cuddy","Body language affects how others see us, but it may also change how we see ourselves.",["psychology","body language","science"],"https://www.ted.com/talks/amy_cuddy_your_body_language_may_shape_who_you_are",58000000,1258,"2012-10-01"),
        _make_talk("3","The power of vulnerability","Brené Brown","Brené Brown studies human connection — our ability to empathize, belong, love.",["vulnerability","psychology","empathy","relationships"],"https://www.ted.com/talks/brene_brown_the_power_of_vulnerability",55000000,1219,"2010-12-23"),
        _make_talk("4","How great leaders inspire action","Simon Sinek","Simon Sinek presents a model for inspirational leadership.",["leadership","business","motivation"],"https://www.ted.com/talks/simon_sinek_how_great_leaders_inspire_action",50000000,1088,"2010-09-28"),
        _make_talk("5","Inside the mind of a master procrastinator","Tim Urban","Tim Urban explores the strange habits of the procrastinator.",["procrastination","psychology","humor"],"https://www.ted.com/talks/tim_urban_inside_the_mind_of_a_master_procrastinator",45000000,850,"2016-04-06"),
        _make_talk("6","The surprising science of happiness","Dan Gilbert","Dan Gilbert challenges the idea that we'll be miserable if we don't get what we want.",["happiness","psychology","science"],"https://www.ted.com/talks/dan_gilbert_the_surprising_science_of_happiness",17000000,1297,"2004-02-10"),
        _make_talk("7","What makes a good life?","Robert Waldinger","A 75-year Harvard study on happiness points to one powerful conclusion.",["happiness","health","relationships","science"],"https://www.ted.com/talks/robert_waldinger_what_makes_a_good_life_lessons_from_the_longest_study_on_happiness",37000000,755,"2015-12-23"),
        _make_talk("8","The puzzle of motivation","Dan Pink","Career analyst Dan Pink examines the puzzle of motivation.",["motivation","business","economics","psychology"],"https://www.ted.com/talks/dan_pink_the_puzzle_of_motivation",24000000,1089,"2009-09-01"),
        _make_talk("9","The danger of a single story","Chimamanda Ngozi Adichie","Our lives and cultures are composed of many overlapping stories.",["storytelling","culture","identity"],"https://www.ted.com/talks/chimamanda_ngozi_adichie_the_danger_of_a_single_story",28000000,1116,"2009-10-07"),
        _make_talk("10","How to speak so that people want to listen","Julian Treasure","Julian Treasure shares seven deadly sins of speaking.",["communication","language","science"],"https://www.ted.com/talks/julian_treasure_how_to_speak_so_that_people_want_to_listen",35000000,960,"2014-06-27"),
        _make_talk("11","Grit: The power of passion and perseverance","Angela Duckworth","Angela Lee Duckworth explains her theory of grit.",["motivation","education","psychology","success"],"https://www.ted.com/talks/angela_lee_duckworth_grit_the_power_of_passion_and_perseverance",23000000,370,"2013-05-09"),
        _make_talk("12","How to make stress your friend","Kelly McGonigal","Kelly McGonigal urges us to see stress as a positive force.",["health","psychology","science","stress"],"https://www.ted.com/talks/kelly_mcgonigal_how_to_make_stress_your_friend",28000000,856,"2013-09-04"),
        _make_talk("13","The power of introverts","Susan Cain","In a culture that prizes being outgoing, it can be difficult to be an introvert.",["psychology","culture","personality"],"https://www.ted.com/talks/susan_cain_the_power_of_introverts",32000000,1128,"2012-03-02"),
        _make_talk("14","How language shapes the way we think","Lera Boroditsky","There are 7000 languages with different sounds and structures.",["language","linguistics","neuroscience","science"],"https://www.ted.com/talks/lera_boroditsky_how_language_shapes_the_way_we_think",9000000,853,"2018-05-02"),
        _make_talk("15","Can we build AI without losing control?","Sam Harris","Can we build an AI that has goals aligned with human values?",["ai","technology","philosophy","future"],"https://www.ted.com/talks/sam_harris_can_we_build_ai_without_losing_control_over_it",6000000,1434,"2016-06-28"),
        _make_talk("16","How do you explain consciousness?","David Chalmers","Philosopher David Chalmers presents the hard problem of consciousness.",["consciousness","philosophy","neuroscience"],"https://www.ted.com/talks/david_chalmers_how_do_you_explain_consciousness",4000000,1136,"2014-03-20"),
        _make_talk("17","What makes a great leader?","Roselinde Torres","Roselinde Torres describes 21st century leadership.",["leadership","business","management"],"https://www.ted.com/talks/roselinde_torres_what_it_takes_to_be_a_great_leader",3000000,900,"2013-11-01"),
        _make_talk("18","The mathematics of love","Hannah Fry","Finding the right partner might be easier with this formula.",["relationships","science","mathematics"],"https://www.ted.com/talks/hannah_fry_the_mathematics_of_love",7000000,983,"2014-02-13"),
        _make_talk("19","Why we do what we do","Tony Robbins","Tony Robbins discusses the invisible forces that motivate us.",["motivation","psychology","emotion"],"https://www.ted.com/talks/tony_robbins_asks_why_we_do_what_we_do",22000000,1333,"2006-06-15"),
        _make_talk("20","The happy secret to better work","Shawn Achor","Positive psychology for the workplace.",["happiness","business","psychology","productivity"],"https://www.ted.com/talks/shawn_achor_the_happy_secret_to_better_work",26000000,780,"2011-05-04"),
        _make_talk("21","Your elusive creative genius","Elizabeth Gilbert","Elizabeth Gilbert on nurturing creativity.",["creativity","art","storytelling"],"https://www.ted.com/talks/elizabeth_gilbert_your_elusive_creative_genius",6000000,1161,"2009-02-05"),
        _make_talk("22","The skill of self confidence","Ivan Joseph","Self confidence as a trainable skill.",["motivation","psychology","education"],"https://www.ted.com/talks/ivan_joseph_the_skill_of_self_confidence",11000000,918,"2012-11-20"),
        _make_talk("23","How to build your creative confidence","David Kelley","The divide between creatives and practical people is false.",["creativity","design","innovation","education"],"https://www.ted.com/talks/david_kelley_how_to_build_your_creative_confidence",10000000,998,"2012-03-06"),
        _make_talk("24","The next outbreak? We're not ready","Bill Gates","We mobilized to fight Ebola but almost didn't contain it.",["health","science","technology","global issues"],"https://www.ted.com/talks/bill_gates_the_next_outbreak_we_re_not_ready",40000000,506,"2015-03-18"),
        _make_talk("25","How to find work you love","Scott Dinsmore","Scott Dinsmore quit his job to find work that matters.",["business","motivation","happiness","career"],"https://www.ted.com/talks/scott_dinsmore_how_to_find_work_you_love",6000000,1080,"2012-06-26"),
    ]


# ── Main fetch function ───────────────────────────────────────────

def fetch_talks(limit: int = 500) -> List[Dict]:
    """
    Fetch talks from best available source.
    Returns deduplicated list up to limit.
    """
    global _cache, _cache_time

    if _cache and (time.time() - _cache_time) < CACHE_TTL:
        logger.info(f"Cache hit: {len(_cache)} talks")
        return list(_cache.values())[:limit]

    talks = []

    # 1. YouTube API (richest source — 500+ talks)
    yt_talks = _fetch_youtube(limit)
    if yt_talks:
        talks = yt_talks
        logger.info(f"Using YouTube API: {len(talks)} talks")

    # 2. Multi-RSS from TED website (fallback — 300-400 talks)
    if not talks:
        talks = _fetch_multi_rss(limit)
        logger.info(f"Using multi-RSS: {len(talks)} talks")

    # 3. Single RSS (last resort before seed)
    if not talks:
        talks = _fetch_single_rss(50)

    # 4. Seed data (offline only)
    if not talks:
        logger.warning("All network sources failed — using seed data")
        talks = _seed_talks()

    # Merge seed talks in if total is still small
    if len(talks) < 50:
        seed = _seed_talks()
        existing_ids = {t["id"] for t in talks}
        for t in seed:
            if t["id"] not in existing_ids:
                talks.append(t)

    # Deduplicate by id
    seen = {}
    for t in talks:
        if t["id"] and t["id"] not in seen:
            seen[t["id"]] = t

    _cache      = seen
    _cache_time = time.time()

    result = list(seen.values())[:limit]
    logger.info(f"fetch_talks: returning {len(result)} talks")
    return result


def search_talks(query: str, limit: int = 30) -> List[Dict]:
    """Search cached talks by keyword across title, description, tags."""
    all_talks = fetch_talks()
    q = query.lower().strip()
    if not q:
        return []
    results = []
    for t in all_talks:
        score = 0
        if q in t["title"].lower():        score += 3
        if q in t["speaker"].lower():      score += 2
        if q in t["description"].lower():  score += 1
        if any(q in tag for tag in t.get("tags", [])): score += 2
        if score > 0:
            results.append((score, t))
    results.sort(key=lambda x: -x[0])
    return [t for _, t in results[:limit]]


def refresh():
    """Force re-fetch on next call."""
    global _cache, _cache_time
    _cache, _cache_time = {}, 0
