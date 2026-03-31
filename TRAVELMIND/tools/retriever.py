import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, TOP_K_RESULTS
from .recommender import recommend

# ── Setup ──────────────────────────────────────────────────────────────────
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
client      = chromadb.PersistentClient(path=CHROMA_PATH)
collection  = client.get_collection("travel_knowledge")
embedder    = SentenceTransformer(EMBED_MODEL)


# ── Per-country context retrieval ──────────────────────────────────────────
def get_rich_context(countries: list, query: str) -> str:
    query_embed = embedder.encode([query]).tolist()
    seen        = set()
    all_chunks  = []

    for country in countries:
        try:
            results = collection.query(
                query_embeddings=query_embed,
                n_results=TOP_K_RESULTS,
                where={"$and": [
                    {"source": {"$eq": country}},
                    {"type":   {"$eq": "destination_guide"}}
                ]}
            )
            chunks = results["documents"][0] if results["documents"] else []
        except Exception:
            chunks = []

        if not chunks:
            try:
                results = collection.query(
                    query_embeddings=query_embed,
                    n_results=TOP_K_RESULTS,
                    where={"source": {"$eq": country}}
                )
                chunks = results["documents"][0] if results["documents"] else []
            except Exception:
                chunks = []

        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(f"[{country}]\n{chunk}")

    if not all_chunks:
        try:
            results    = collection.query(query_embeddings=query_embed, n_results=TOP_K_RESULTS)
            all_chunks = results["documents"][0] if results["documents"] else []
        except Exception:
            pass

    return "\n\n---\n\n".join(all_chunks[:TOP_K_RESULTS * 2])


# ── Visa details retrieval ─────────────────────────────────────────────────
def get_visa_details(country: str, query_embed: list) -> str:
    try:
        results = collection.query(
            query_embeddings=query_embed,
            n_results=1,
            where={"$and": [
                {"source": {"$eq": country}},
                {"type":   {"$eq": "visa"}}
            ]}
        )
        if results["documents"] and results["documents"][0]:
            return results["documents"][0][0]
    except Exception:
        pass
    return ""


# ── Query normalizer ──────────────────────────────────────────────────────
# Maps vague preference statements into concrete travel search terms
# so the semantic ranker has something meaningful to work with.
PREFERENCE_EXPANSIONS = {
    "historical":  "historical sites ancient ruins monuments heritage culture",
    "history":     "historical sites ancient ruins monuments heritage culture",
    "ruins":       "historical ancient ruins archaeological heritage",
    "monument":    "historical monuments ancient culture heritage",
    "beach":       "beach coast seaside tropical island",
    "nature":      "nature wildlife national park outdoor adventure",
    "wildlife":    "safari wildlife game reserve nature",
    "safari":      "safari wildlife game reserve nature adventure",
    "luxury":      "luxury high-end exclusive resort premium",
    "budget":      "affordable cheap budget low cost",
    "cheap":       "affordable cheap budget low cost",
    "adventure":   "adventure outdoor hiking trekking",
    "romantic":    "honeymoon romantic luxury island beach",
    "honeymoon":   "honeymoon romantic luxury island beach",
    "family":      "family friendly safe beaches",
    "solo":        "solo traveler safe culture adventure",
    "food":        "local food cuisine gastronomy market",
    "diving":      "diving snorkeling marine underwater",
    "culture":     "culture arts heritage local life",
    "music":       "culture festivals local life arts",
    "relax":       "beach island resort luxury relaxation",
}

def _expand_query(query: str) -> str:
    """Turn preference statements into richer travel search queries."""
    q = query.lower()
    extras = []
    for keyword, expansion in PREFERENCE_EXPANSIONS.items():
        if keyword in q and expansion not in q:
            extras.append(expansion)
    if extras:
        return query + " " + " ".join(extras)
    # If query looks like a pure preference statement with no destination intent,
    # prefix it to make it sound like a travel search
    preference_phrases = ["i like", "i love", "i enjoy", "i prefer", "i want", "i'm interested in"]
    if any(p in q for p in preference_phrases) and "africa" not in q:
        return "recommend african destination for traveler who likes " + query
    return query


# ── Main retrieve function — NO LLM call ───────────────────────────────────
def retrieve(query: str) -> str:
    """
    Returns structured text with recommendations + context.
    The LLM formatting is handled upstream in agent.py (one call total).
    """
    enriched = _expand_query(query)
    recs = recommend(enriched)

    # Fallback: if enriched query still returns nothing, try raw query
    if not recs:
        recs = recommend(query)

    if not recs:
        return ""

    top_countries = [r["country"] for r in recs]

    # Structured scores
    recs_text = ""
    for r in recs:
        recs_text += (
            f"\nCountry: {r['country']}\n"
            f"  Visa: {r['visa']} (ease {r['visa_score']}/10)\n"
            f"  Affordability: {r['afford']}/10\n"
            f"  Safety: {r['safety']}/10\n"
            f"  Tags: {r['tags']}\n"
            f"  Ideal for: {r['profiles']}\n"
        )

    # ChromaDB context
    context = get_rich_context(top_countries, query)

    # Visa details
    query_embed  = embedder.encode([query]).tolist()
    visa_details = ""
    for country in top_countries:
        info = get_visa_details(country, query_embed)
        if info:
            visa_details += f"\n{country}: {info[:200]}\n"

    # Return plain structured data — no LLM
    parts = ["=== RECOMMENDED DESTINATIONS ===", recs_text]
    if visa_details:
        parts += ["\n=== VISA DETAILS ===", visa_details]
    if context:
        parts += ["\n=== CONTEXT ===", context[:2000]]

    return "\n".join(parts)