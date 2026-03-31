"""
planner.py — build, refine, and summarize travel itineraries using Gemini 2.5 Flash.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import requests
from config import GEMINI_API_KEY, LLM_MODEL, MAX_TOKENS


def call_llm(messages: list) -> str:
    system_text, contents = "", []
    for m in messages:
        role, text = m["role"], m["content"]
        if role == "system":
            system_text += text + "\n\n"
        elif role in ("user", "human"):
            combined = (system_text + text).strip() if system_text and not contents else text
            contents.append({"role": "user",  "parts": [{"text": combined}]})
            system_text = ""
        elif role in ("assistant", "ai", "model"):
            contents.append({"role": "model", "parts": [{"text": text}]})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": system_text.strip()}]}]

    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={"contents": contents,
                  "generationConfig": {"maxOutputTokens": MAX_TOKENS, "temperature": 0.7}},
            timeout=45,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.HTTPError as e:
        print(f"[Planner ERROR] {e.response.status_code}: {e.response.text[:200]}")
        return ""
    except Exception as e:
        print(f"[Planner ERROR] {e}")
        return ""


def _build_messages(system: str, user: str) -> list:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _get_context(destination: str, preferences: str) -> str:
    try:
        from tools.retriever import get_rich_context
        return get_rich_context([destination], f"{destination} {preferences} travel itinerary")
    except Exception:
        return ""


def _get_country_scores(destination: str) -> dict:
    try:
        from tools.recommender import _destinations
        for d in _destinations:
            if d["country"].lower() == destination.lower():
                return d
    except Exception:
        pass
    return {}


def build_itinerary(destination, days, preferences, budget, passport="Moroccan", context="") -> str:
    if not context:
        context = _get_context(destination, preferences)

    scores = _get_country_scores(destination)
    if scores:
        context = (
            f"Visa: {scores.get('visa','?')} (ease {scores.get('visa_score','?')}/10), "
            f"Affordability: {scores.get('afford','?')}/10, "
            f"Safety: {scores.get('safety','?')}/10\n\n"
        ) + context

    system = f"""You are an expert travel planner for Moroccan travelers visiting African destinations.
Generate a complete, realistic {days}-day itinerary for {destination} within a ${budget} budget.

Use this exact structure — fill every field with real values:

TRIP OVERVIEW
-------------
Destination:           {destination}
Duration:              {days} days
Estimated Total Cost:  $[number] USD
Travel Style:          [style matching preferences]
Visa for Moroccans:    [visa type and conditions]

BUDGET BREAKDOWN
----------------
Flights from Morocco:  ~$[number]
Accommodation:         $[number]/night x {days} nights = $[number]
Food:                  ~$[number]/day x {days} days = $[number]
Transport:             ~$[number] total
Activities:            ~$[number] total
Buffer (10%):          ~$[number]
TOTAL:                 ~$[number] USD

DAY 1 — [Theme]
Morning:   [activity] — [practical tip]
Afternoon: [activity] — [practical tip]
Evening:   [activity] — [practical tip]
Daily Cost: ~$[number] USD  ← food/day + accommodation/night + actual activities cost for THIS day
Tip: [insider tip]

[Repeat DAY block for all {days} days]

IMPORTANT: Daily Cost must reflect the ACTUAL cost of that specific day.
Days with expensive activities (safaris, permits, guided tours) cost more.
Days with free or cheap activities cost less. Not every day should be the same number.
The sum of all Daily Costs + Flights + Buffer must equal the TOTAL.

Preferences: {preferences}
Passport: {passport}
Context: {context}"""

    return call_llm(_build_messages(system,
        f"Write the full {days}-day itinerary for {destination}, budget ${budget}."))


def refine_itinerary(itinerary: str, feedback: str) -> str:
    system = """You are an expert travel planner. Refine the itinerary based on the traveler's feedback.

Rules:
- Only change what was explicitly requested.
- Keep the exact same structure and format as the original.
- Recalculate Daily Cost and TOTAL if activities or accommodation change.
- After the full updated itinerary, add:

CHANGES MADE
------------
[bullet list of what changed and why]"""

    return call_llm(_build_messages(system,
        f"Original itinerary:\n{itinerary}\n\nFeedback: {feedback}"))


def summarize_itinerary(itinerary: str) -> str:
    system = """Extract the key facts from this itinerary and present them clearly.

Use this exact format:

Destination:   [name]
Duration:      [X] days
Total Budget:  $[number]
Visa:          [visa info]

[DAY 1]: [one sentence summary]
[DAY 2]: [one sentence summary]
[repeat for each day]

Budget Breakdown:
  Flights:              ~$[number]
  Accommodation:        ~$[number]
  Food:                 ~$[number]
  Transport + Activities: ~$[number]

Tip: [one practical tip from the itinerary]"""

    return call_llm(_build_messages(system, itinerary))


if __name__ == "__main__":
    it = build_itinerary("Egypt", 3, "history, local food", "1500")
    print(it)
    print("\n--- REFINE ---")
    print(refine_itinerary(it, "Make day 2 a Nile cruise instead"))
    print("\n--- SUMMARY ---")
    print(summarize_itinerary(it))