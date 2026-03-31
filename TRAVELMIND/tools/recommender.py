import os
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import EMBED_MODEL, TOP_K_DESTINATIONS, DATA_PATH

# ── Constants ────────────────────────────────────────────────────────────
CHEAP_MIN    = 7
SAFE_MIN     = 7
VISA_FREE    = 10
UNSAFE_MAX   = 3   # always exclude safety <= 3 (Mali, Burkina, Libya, Sudan)

METADATA_CSV = os.path.join(DATA_PATH, "updated_country_metadata.csv")

# ── Model ────────────────────────────────────────────────────────────────
model = SentenceTransformer(EMBED_MODEL)

# ── Load data ────────────────────────────────────────────────────────────
def load_destinations():
    destinations = []
    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            destinations.append({
                "country":    row["country"],
                "visa":       row.get("visa_morocco", ""),
                "visa_score": int(row.get("visa_ease_score", 0)),
                "afford":     int(row.get("affordability_score", 0)),
                "safety":     int(row.get("safety_score", 0)),
                "tags":       row.get("travel_tags", ""),
                "profiles":   row.get("ideal_profiles", ""),
                "continent":  row.get("continent", ""),
            })
    return destinations

_destinations = load_destinations()
_tag_texts    = [d["tags"] for d in _destinations]
_tag_embeds   = model.encode(_tag_texts)


# ── Intent parser ────────────────────────────────────────────────────────
def parse_intent(query: str) -> dict:
    q = query.lower()
    return {
        # Full keyword lists so "without visa", "visa-free" etc. all trigger
        "visa_free": any(k in q for k in ["no visa", "without visa", "visa-free", "visa free", "visa free"]),
        "cheap":     any(k in q for k in ["cheap", "budget", "affordable", "low cost"]),
        "safe":      any(k in q for k in ["safe", "solo", "family", "alone", "kids"]),
        "is_solo":   any(k in q for k in ["solo", "alone", "by myself"]),
        "is_family": any(k in q for k in ["family", "kids", "children"]),
        "query":     query,
    }


# ── Hard filters ─────────────────────────────────────────────────────────
def apply_filters(destinations: list, intent: dict) -> list:
    result   = []
    excluded = []

    for d in destinations:
        # Always exclude dangerous destinations regardless of query
        if d["safety"] <= UNSAFE_MAX:
            excluded.append((d["country"], f"safety={d['safety']} unsafe"))
            continue

        reasons = []
        if intent["visa_free"] and d["visa_score"] < VISA_FREE:
            reasons.append(f"visa not free (score={d['visa_score']})")
        if intent["cheap"] and d["afford"] < CHEAP_MIN:
            reasons.append(f"not affordable (score={d['afford']})")
        if intent["safe"] and d["safety"] < SAFE_MIN:
            reasons.append(f"safety too low (score={d['safety']})")

        if reasons:
            excluded.append((d["country"], "; ".join(reasons)))
        else:
            result.append(d)

    '''if excluded:
        print(f"[Recommender] Excluded {len(excluded)} destinations:")
        for name, reason in excluded[:5]:
            print(f"   ✗ {name}: {reason}")
        if len(excluded) > 5:
            print(f"   ... and {len(excluded) - 5} more.")'''

    return result


# ── Ranking ───────────────────────────────────────────────────────────────
def rank(destinations: list, query: str, intent: dict = None) -> list:
    if not destinations:
        return []

    q      = query.lower()
    intent = intent or parse_intent(query)   # fallback if not passed in

    # ── Query expansion ───────────────────────────────────────────────────
    # Bridge vocabulary gap between user words and travel_tags
    EXPANSIONS = {
        "cheap":        "affordable budget low cost",
        "luxury":       "luxury high-end exclusive resort premium",
        "no visa":      "visa-free easy entry",
        "without visa": "visa-free easy entry",
        "safe":         "safe family-friendly",
        "safari":       "safari wildlife game reserve",
        "honeymoon":    "honeymoon romantic luxury island",
        "adventure":    "adventure outdoor hiking",
        "beach":        "beach coast seaside",
        "island":       "island tropical paradise",
        "history":      "historical culture heritage",
        "diving":       "diving snorkeling marine",
    }
    expanded = query
    for kw, exp in EXPANSIONS.items():
        if kw in q:
            expanded += " " + exp

    # ── Semantic similarity ───────────────────────────────────────────────
    query_embed = model.encode([expanded])
    tags        = [d["tags"] for d in destinations]
    embeds      = model.encode(tags)
    sims        = cosine_similarity(query_embed, embeds)[0]

    # ── Tag direct bonus + specificity penalty ────────────────────────────
    TAG_BONUS_MAP = {
        "luxury":    ("luxury",    0.15),
        "safari":    ("safari",    0.15),
        "honeymoon": ("honeymoon", 0.15),
        "romantic":  ("romantic",  0.15),
        "diving":    ("diving",    0.12),
        "beach":     ("beach",     0.08),
        "adventure": ("adventure", 0.08),
        "island":    ("island",    0.08),
        "history":   ("historical",0.08),
        "culture":   ("culture",   0.06),
    }
    # High-signal keywords: cap similarity if country has none of the required tags
    HIGH_SIGNAL = {
        "honeymoon": ["honeymoon", "romantic", "luxury", "island", "beach"],
        "luxury":    ["luxury", "honeymoon", "resort"],
        "safari":    ["safari", "wildlife"],
        "diving":    ["diving", "snorkeling"],
    }

    results = []
    for i, d in enumerate(destinations):
        sim       = float(sims[i])
        tags_low  = d["tags"].lower()
        prof_low  = d["profiles"].lower()

        # Tag direct bonus
        tag_bonus = 0.0
        for ukw, (tkw, bonus) in TAG_BONUS_MAP.items():
            if ukw in q and tkw in tags_low:
                tag_bonus = max(tag_bonus, bonus)

        # Specificity penalty — cap irrelevant countries at 0.30
        for signal_kw, required_tags in HIGH_SIGNAL.items():
            if signal_kw in q:
                if not any(rt in tags_low for rt in required_tags):
                    sim = min(sim, 0.30)
                break

        # Profile bonus
        profile_bonus = 0.0
        if intent["is_solo"] and "solo" in prof_low:          profile_bonus = 0.10
        elif intent["is_family"] and "family" in prof_low:    profile_bonus = 0.10
        elif "luxury" in q and "luxury" in prof_low:          profile_bonus = 0.08
        elif "adventure" in q and "adventure" in prof_low:    profile_bonus = 0.08
        elif "honeymoon" in q and "honeymoon" in prof_low:    profile_bonus = 0.08

        # Need intent for profile bonus — re-parse from query
        # ── Quality bonus ──────────────────────────────────────────────
        # When semantic scores are close, surface better destinations first.
        # Weighted: visa ease (most important) + affordability + safety.
        # Normalised to a small range (0.0 – 0.08) so it acts as tiebreaker
        # without overriding a genuinely better semantic match.
        quality_bonus = round(
            (d["visa_score"] / 10) * 0.04 +   # visa ease  → up to 0.04
            (d["afford"]     / 10) * 0.02 +   # affordability → up to 0.02
            (d["safety"]     / 10) * 0.02,    # safety        → up to 0.02
            3
        )

        results.append({**d,
            "score":          round(sim + tag_bonus + profile_bonus + quality_bonus, 3),
            "tag_bonus":      round(tag_bonus, 3),
            "profile_bonus":  round(profile_bonus, 3),
            "quality_bonus":  round(quality_bonus, 3),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ── Main function ─────────────────────────────────────────────────────────
def recommend(query: str, top_k=TOP_K_DESTINATIONS) -> list:
    intent   = parse_intent(query)
    filtered = apply_filters(_destinations, intent)

    if not filtered:
        # Relax safe filter and retry
        intent["safe"] = False
        filtered = apply_filters(_destinations, intent)

    if not filtered:
        return []

    # Pass intent to rank so profile bonus works correctly
    ranked = rank(filtered, query, intent)
    # Attach intent to each result for use in retriever prompt
    for r in ranked:
        r["intent"] = intent

    return ranked[:top_k]


# ── Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "cheap and safe country without visa",
        "luxury travel",
        "safari adventure",
        "beach vacation",
        "honeymoon romantic getaway",
        "solo traveler culture and history",
        "family trip safe with beaches",
    ]

    for q in queries:
        print("\n" + "="*55)
        print(q)
        print("-"*55)
        results = recommend(q)
        for r in results:
            print(f"  {r['country']:<22} score={r['score']:.3f} "
                  f"visa={r['visa']:<18} afford={r['afford']} safety={r['safety']}")