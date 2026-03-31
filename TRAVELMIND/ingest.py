import os
import re
import csv
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_PATH, DATA_PATH, EMBED_MODEL

# ── Setup ──────────────────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("travel_knowledge")
model      = SentenceTransformer(EMBED_MODEL)

# Key data file paths
METADATA_CSV = os.path.join(DATA_PATH, "updated_country_metadata.csv")
LOGIC_TXT    = os.path.join(DATA_PATH, "Recommendation_Logic.txt")

# ── Name aliases ───────────────────────────────────────────────────────────
# Maps long names used in Recommendation_Logic.txt → short names in the CSV.
# Add entries here whenever a new country has a parenthetical alternate name.
NAME_ALIASES = {
    "Ivory Coast (Côte d'Ivoire)": "Ivory Coast",
    "Cape Verde (Cabo Verde)":     "Cape Verde",
    "Eswatini (Swaziland)":        "Eswatini",
}

# ── Chunking ───────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=50):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ── Readers ────────────────────────────────────────────────────────────────
def read_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# ── Metadata loader ────────────────────────────────────────────────────────
def load_country_metadata():
    """
    Load updated_country_metadata.csv into a dict keyed by country name.
    Used to enrich ChromaDB metadata with structured scores for filtering.

    Scores power the hard filters from Recommendation_Logic.txt:
      - visa_ease_score      → hard filter: visa-free (score = 10)
      - affordability_score  → hard filter: cheap = score >= 7
      - safety_score         → hard filter: solo/family = score >= 7

    Returns:
        dict: { "Senegal": { visa_morocco, visa_ease_score, affordability_score,
                             safety_score, continent, travel_tags, ideal_profiles }, ... }
    """
    metadata_map = {}
    if not os.path.exists(METADATA_CSV):
        print(f"  ⚠️  {METADATA_CSV} not found. Score-enriched metadata will be skipped.")
        return metadata_map

    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row["country"].strip()
            metadata_map[country] = {
                "visa_morocco":        row.get("visa_morocco", "unknown").strip(),
                "visa_ease_score":     int(row.get("visa_ease_score", 0)),
                "affordability_score": int(row.get("affordability_score", 0)),
                "safety_score":        int(row.get("safety_score", 0)),
                "continent":           row.get("continent", "").strip(),
                "travel_tags":         row.get("travel_tags", "").strip(),
                "ideal_profiles":      row.get("ideal_profiles", "").strip(),
            }
    print(f"  ✅ Loaded metadata for {len(metadata_map)} countries from updated_country_metadata.csv")
    return metadata_map

# ── Ingest functions ───────────────────────────────────────────────────────
def ingest_country_metadata(docs, ids, metas, doc_id, country_metadata):
    """
    Ingest updated_country_metadata.csv as natural-language chunks.
    Each row becomes an embeddable sentence AND carries structured score
    metadata so the retriever can later filter by visa / affordability / safety.

    Hard filters from Recommendation_Logic.txt:
      - Visa-free filter      → visa_ease_score == 10
      - Cheap/budget filter   → affordability_score >= 7
      - Solo/family filter    → safety_score >= 7
    """
    if not country_metadata:
        print("  ⚠️  No country metadata loaded. Skipping metadata ingestion.")
        return doc_id

    print(f"  Ingesting {len(country_metadata)} country metadata rows...")
    for country, meta in country_metadata.items():
        text = (
            f"Country: {country}. "
            f"Continent: {meta['continent']}. "
            f"Visa for Moroccan citizens: {meta['visa_morocco']} "
            f"(ease score: {meta['visa_ease_score']}/10). "
            f"Affordability score: {meta['affordability_score']}/10. "
            f"Safety score: {meta['safety_score']}/10. "
            f"Travel experience tags: {meta['travel_tags']}. "
            f"Ideal traveler profiles: {meta['ideal_profiles']}."
        )
        docs.append(text)
        ids.append(f"doc_{doc_id}")
        metas.append({
            "source":              country,
            "type":                "country_metadata",
            "file":                "updated_country_metadata.csv",
            "visa_morocco":        meta["visa_morocco"],
            "visa_ease_score":     meta["visa_ease_score"],
            "affordability_score": meta["affordability_score"],
            "safety_score":        meta["safety_score"],
            "continent":           meta["continent"],
            "travel_tags":         meta["travel_tags"],
            "ideal_profiles":      meta["ideal_profiles"],
        })
        doc_id += 1

    return doc_id


def ingest_recommendation_logic(docs, ids, metas, doc_id, country_metadata):
    """
    Option B: Ingest Recommendation_Logic.txt directly.
    No split_guides.py needed — replaces ingest_destinations() entirely.

    Strategy:
      1. Extract the filtering/reasoning rules at the top of the file and
         ingest as a dedicated chunk so the LLM can retrieve and apply them.
      2. Split the rest by 'Country:' headers into individual country profiles.
      3. Chunk each country profile separately (country-specific retrieval).
      4. Enrich every chunk's ChromaDB metadata with scores from
         updated_country_metadata.csv, linking rich narrative text to the
         hard-filter scores defined in Recommendation_Logic.txt.

    Recommendation_Logic.txt rules applied here:
      - Hard filter: visa        → visa_ease_score == 10 (visa-free)
      - Hard filter: budget      → affordability_score >= 7 ("cheap")
      - Hard filter: safety      → safety_score >= 7 (solo / family traveler)
      - Soft matching            → travel_tags (beach, safari, luxury, etc.)
      - Traveler profile match   → ideal_profiles (solo, family, couple, etc.)
      - Similar destinations     → embedded in the narrative text for fallback
    """
    if not os.path.exists(LOGIC_TXT):
        print(f"  ⚠️  {LOGIC_TXT} not found. Skipping recommendation logic ingestion.")
        return doc_id

    content = read_txt(LOGIC_TXT)

    # ── Step 1: Extract and ingest the rules / reasoning section ──────────
    # Everything before the first "Country:" line is the filtering logic
    first_country_match = re.search(r"\nCountry:", content)
    if first_country_match:
        rules_text = content[:first_country_match.start()].strip()
    else:
        rules_text = content.strip()

    if rules_text:
        for chunk in chunk_text(rules_text, chunk_size=400, overlap=50):
            docs.append(chunk)
            ids.append(f"doc_{doc_id}")
            metas.append({
                "source": "recommendation_rules",
                "type":   "recommendation_logic",
                "file":   "Recommendation_Logic.txt",
            })
            doc_id += 1
        print(f"  ✅ Ingested recommendation rules section.")

    # ── Step 2: Split by Country: headers ─────────────────────────────────
    country_splits   = re.split(r"(?=\nCountry:\s)", content)
    country_sections = [
        s.strip() for s in country_splits
        if re.match(r"Country:\s", s.strip())
    ]
    print(f"  Found {len(country_sections)} country profiles in Recommendation_Logic.txt...")

    # ── Steps 3 & 4: Chunk each profile, enrich with CSV scores ───────────
    matched   = 0
    unmatched = []

    for section in country_sections:
        # Extract country name from header
        name_match = re.match(r"Country:\s*(.+)", section)
        if not name_match:
            continue
        country_name = name_match.group(1).strip()

        # Resolve alias: e.g. "Ivory Coast (Côte d'Ivoire)" → "Ivory Coast"
        csv_name   = NAME_ALIASES.get(country_name, country_name)

        # Look up scores from updated_country_metadata.csv
        meta_entry = country_metadata.get(csv_name, {})
        if meta_entry:
            matched += 1
        else:
            unmatched.append(country_name)

        # Base ChromaDB metadata — use csv_name as source so it matches
        # country names in updated_country_metadata.csv and recommender.py
        chunk_meta = {
            "source": csv_name,
            "type":   "destination_guide",
            "file":   "Recommendation_Logic.txt",
        }

        # Enrich with hard-filter scores if available
        if meta_entry:
            chunk_meta.update({
                # visa_ease_score == 10  → qualifies for "no visa" hard filter
                "visa_morocco":        meta_entry["visa_morocco"],
                "visa_ease_score":     meta_entry["visa_ease_score"],
                # affordability_score >= 7 → qualifies for "cheap" hard filter
                "affordability_score": meta_entry["affordability_score"],
                # safety_score >= 7 → qualifies for solo traveler / family hard filter
                "safety_score":        meta_entry["safety_score"],
                "continent":           meta_entry["continent"],
                # Comma-separated tags for soft matching (beach, safari, luxury…)
                "travel_tags":         meta_entry["travel_tags"],
                # Comma-separated profiles for traveler-type matching
                "ideal_profiles":      meta_entry["ideal_profiles"],
            })

        for chunk in chunk_text(section, chunk_size=500, overlap=50):
            docs.append(chunk)
            ids.append(f"doc_{doc_id}")
            metas.append(chunk_meta)
            doc_id += 1

    print(f"  ✅ {matched} countries matched to CSV scores.")
    if unmatched:
        print(f"  ⚠️  {len(unmatched)} unmatched (no CSV row): {', '.join(unmatched)}")

    return doc_id


def ingest_safety(docs, ids, metas, doc_id):
    """Ingest safety_advisories.txt."""
    fpath = os.path.join(DATA_PATH, "safety_advisories.txt")
    if not os.path.exists(fpath):
        print("  ⚠️  safety_advisories.txt not found. Skipping.")
        return doc_id

    text = read_txt(fpath)
    for chunk in chunk_text(text, chunk_size=300):
        docs.append(chunk)
        ids.append(f"doc_{doc_id}")
        metas.append({
            "source": "safety_advisories",
            "type":   "safety",
            "file":   "safety_advisories.txt",
        })
        doc_id += 1

    return doc_id


def ingest_visa(docs, ids, metas, doc_id, country_metadata):
    """
    Ingest visa data. Uses dedicated visa_requirements.csv if present,
    otherwise falls back to updated_country_metadata.csv.

    Implements hard filter from Recommendation_Logic.txt:
      'If user says no visa → filter to visa-free countries.'
    """
    fpath = os.path.join(DATA_PATH, "visa_requirements.csv")

    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    f"Visa information: {row['passport_country']} passport traveling to "
                    f"{row['destination']}. Visa required: {row['visa_required']}. "
                    f"Type: {row['visa_type']}. Cost: ${row['cost_usd']}. "
                    f"Maximum stay: {row['max_stay_days']} days. Notes: {row['notes']}"
                )
                docs.append(text)
                ids.append(f"doc_{doc_id}")
                metas.append({
                    "source":      "visa_requirements",
                    "type":        "visa",
                    "passport":    row["passport_country"],
                    "destination": row["destination"],
                    "file":        "visa_requirements.csv",
                })
                doc_id += 1
    elif country_metadata:
        print("  ℹ️  visa_requirements.csv not found — using updated_country_metadata.csv for visa data.")
        for country, meta in country_metadata.items():
            text = (
                f"Visa information for Moroccan citizens traveling to {country}: "
                f"{meta['visa_morocco']} (ease score: {meta['visa_ease_score']}/10). "
                f"Visa-free means no application needed — direct entry allowed."
            )
            docs.append(text)
            ids.append(f"doc_{doc_id}")
            metas.append({
                "source":          "country_metadata_visa",
                "type":            "visa",
                "destination":     country,
                "visa_morocco":    meta["visa_morocco"],
                "visa_ease_score": meta["visa_ease_score"],
                "file":            "updated_country_metadata.csv",
            })
            doc_id += 1
    else:
        print("  ⚠️  No visa data source found. Skipping.")

    return doc_id


def ingest_budget(docs, ids, metas, doc_id, country_metadata):
    """
    Ingest budget data. Uses dedicated budget_estimates.csv if present,
    otherwise falls back to affordability scores from updated_country_metadata.csv.

    Implements hard filter from Recommendation_Logic.txt:
      'If user says cheap → prioritize affordability score >= 7.'
    """
    fpath = os.path.join(DATA_PATH, "budget_estimates.csv")

    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    f"Budget information for {row['destination']}, {row['country']}. "
                    f"Budget traveler: ${row['budget_per_day_usd']}/day. "
                    f"Mid-range: ${row['midrange_per_day_usd']}/day. "
                    f"Luxury: ${row['luxury_per_day_usd']}/day. "
                    f"Budget hostel: ${row['budget_hostel_usd']}/night. "
                    f"Budget hotel: ${row['budget_hotel_usd']}/night. "
                    f"Street meal: ${row['street_meal_usd']}. "
                    f"Mid restaurant: ${row['mid_restaurant_usd']}. "
                    f"Overall budget level: {row['overall_budget_level']}. "
                    f"Notes: {row['notes']}"
                )
                docs.append(text)
                ids.append(f"doc_{doc_id}")
                metas.append({
                    "source":      "budget_estimates",
                    "type":        "budget",
                    "destination": row["destination"],
                    "file":        "budget_estimates.csv",
                })
                doc_id += 1
    elif country_metadata:
        print("  ℹ️  budget_estimates.csv not found — using affordability scores from updated_country_metadata.csv.")
        for country, meta in country_metadata.items():
            text = (
                f"Budget information for {country}. "
                f"Overall affordability score: {meta['affordability_score']}/10 "
                f"(10 = cheapest, 1 = most expensive). "
                f"A score of 7 or above qualifies as a budget-friendly destination."
            )
            docs.append(text)
            ids.append(f"doc_{doc_id}")
            metas.append({
                "source":              "country_metadata_budget",
                "type":                "budget",
                "destination":         country,
                "affordability_score": meta["affordability_score"],
                "file":                "updated_country_metadata.csv",
            })
            doc_id += 1
    else:
        print("  ⚠️  No budget data source found. Skipping.")

    return doc_id


def ingest_seasonal(docs, ids, metas, doc_id):
    """Ingest seasonal_data.csv — each row becomes a searchable chunk."""
    fpath = os.path.join(DATA_PATH, "seasonal_data.csv")
    if not os.path.exists(fpath):
        print("  ⚠️  seasonal_data.csv not found. Skipping.")
        return doc_id

    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (
                f"Seasonal travel information for {row['destination']}. "
                f"Best months to visit: {row['best_months']}. "
                f"Shoulder season: {row['shoulder_months']}. "
                f"Avoid: {row['avoid_months']}. "
                f"Peak season: {row['peak_season']}. "
                f"Average temperature in best months: {row['avg_temp_best_c']}°C. "
                f"Rainy season: {row['rainy_season']}. "
                f"Special events and festivals: {row['special_events']}"
            )
            docs.append(text)
            ids.append(f"doc_{doc_id}")
            metas.append({
                "source":      "seasonal_data",
                "type":        "seasonal",
                "destination": row["destination"],
                "file":        "seasonal_data.csv",
            })
            doc_id += 1

    return doc_id


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs, ids, metas = [], [], []
    doc_id = 0

    print("\n🌍 TravelMind AI — Ingestion Pipeline")
    print("=" * 45)

    # Load structured metadata first — powers all hard filters downstream
    print("\n📊 Loading country metadata scores...")
    country_metadata = load_country_metadata()

    print("\n📊 Ingesting country metadata (scores + tags)...")
    prev   = doc_id
    doc_id = ingest_country_metadata(docs, ids, metas, doc_id, country_metadata)
    print(f"     → {doc_id - prev} chunks added")

    # Option B: replaces ingest_destinations() — no split_guides.py needed
    print("\n📖 Ingesting Recommendation_Logic.txt (rules + country profiles)...")
    prev   = doc_id
    doc_id = ingest_recommendation_logic(docs, ids, metas, doc_id, country_metadata)
    print(f"     → {doc_id - prev} chunks added")

    print("\n⚠️  Ingesting safety advisories...")
    prev   = doc_id
    doc_id = ingest_safety(docs, ids, metas, doc_id)
    print(f"     → {doc_id - prev} chunks added")

    print("\n✈️  Ingesting visa requirements...")
    prev   = doc_id
    doc_id = ingest_visa(docs, ids, metas, doc_id, country_metadata)
    print(f"     → {doc_id - prev} chunks added")

    print("\n💰 Ingesting budget estimates...")
    prev   = doc_id
    doc_id = ingest_budget(docs, ids, metas, doc_id, country_metadata)
    print(f"     → {doc_id - prev} chunks added")

    print("\n📅 Ingesting seasonal data...")
    prev   = doc_id
    doc_id = ingest_seasonal(docs, ids, metas, doc_id)
    print(f"     → {doc_id - prev} chunks added")

    print(f"\n🔢 Total chunks to embed: {len(docs)}")
    print("⏳ Embedding... (this may take 1-2 minutes)")

    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    print("\n💾 Saving to ChromaDB...")
    collection.add(
        documents=docs,
        ids=ids,
        metadatas=metas,
        embeddings=embeddings,
    )

    print(f"\n✅ Done! {len(docs)} chunks loaded into ChromaDB.")
    print(f"   Collection: 'travel_knowledge'")
    print(f"   Location:   {CHROMA_PATH}")
    print("\n   Breakdown:")
    type_counts = {}
    for m in metas:
        t = m["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, count in sorted(type_counts.items()):
        print(f"   - {t:<25} {count} chunks")