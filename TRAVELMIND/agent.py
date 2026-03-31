'''# agent.py

import os, sys, re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import requests

from config import GEMINI_API_KEY, LLM_MODEL
from tools.retriever import retrieve
from tools.planner import build_itinerary, refine_itinerary, summarize_itinerary


# ── Gemini call ────────────────────────────────────────────────────────────
def call_llm(messages: list, max_tokens: int = 1024) -> str:
    """Convert OpenAI-style messages to Gemini format and call the API."""
    system, contents = "", []
    for m in messages:
        role, text = m["role"], m["content"]
        if role == "system":
            system += text + "\n\n"
        elif role in ("user", "human"):
            combined = (system + text).strip() if system and not contents else text
            contents.append({"role": "user",  "parts": [{"text": combined}]})
            system = ""
        elif role in ("assistant", "ai", "model"):
            contents.append({"role": "model", "parts": [{"text": text}]})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": system.strip()}]}]

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent",
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY},
        json={"contents": contents,
              "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:            str
    response:         str
    intent:           str
    history:          list
    plan_destination: str
    plan_days:        int
    plan_budget:      str
    plan_preferences: str
    plan_itinerary:   str


# ── Node 1: classify ───────────────────────────────────────────────────────
def classify_query(state: AgentState) -> AgentState:
    has_itinerary = bool(state.get("plan_itinerary", "").strip())
    history       = state.get("history", [])
    recent        = "\n".join(f"{m['role'].upper()}: {m['content'][:120]}" for m in history[-4:])

    prompt = (
        "You are an intent classifier. Reply with ONE word only.\n\n"
        "Intents:\n"
        "- recommend  : travel advice, destination suggestions, greetings, follow-ups\n"
        "- plan       : user wants a day-by-day itinerary OR asks about budget/cost/price for a destination\n"
        "- refine     : user wants to change an existing itinerary\n"
        "- summarize  : user wants a summary of an existing itinerary\n"
        "- off_topic  : nothing to do with travel\n\n"
        "Examples → plan: 'what is the budget?', 'how much will it cost?', 'yes what about the budget', "
        "'give me a plan', 'plan 3 days in Kenya', 'how much does it cost to go there'\n"
        f"Has active itinerary: {has_itinerary} (refine/summarize need this to be True)\n"
        + (f"\nRecent conversation:\n{recent}\n" if recent else "")
        + f"\nMessage: {state['query']}"
    )

    try:
        raw    = call_llm([{"role": "user", "content": prompt}], max_tokens=5).lower().split()[0]
        valid  = {"recommend", "plan", "refine", "summarize", "off_topic"}
        intent = raw if raw in valid else "recommend"
        if intent in ("refine", "summarize") and not has_itinerary:
            intent = "recommend"
    except Exception:
        intent = "recommend"

    state["intent"] = intent
    return state


# ── Node 2: extract plan params ────────────────────────────────────────────
_DAY_RE    = re.compile(r"\b(\d+)\s*(?:days?|nights?)\b", re.I)
_BUDGET_RE = re.compile(r"\$\s*(\d[\d,]*)|(\d[\d,]*)\s*(?:usd|dollars?)", re.I)
_DEST_RE   = re.compile(
    r"\b(?:to|for|in|visit(?:ing)?|travel(?:ing)?\s+to)\s+([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+)?)",
    re.I
)
DEST_ALIASES = {
    "algerie": "Algeria", "egypte": "Egypt", "tunisie": "Tunisia",
    "maroc": "Morocco",   "senegal": "Senegal", "ethiopie": "Ethiopia",
    "tanzanie": "Tanzania", "zambie": "Zambia", "ouganda": "Uganda",
    "namibie": "Namibia",  "botswana": "Botswana",
    "seychelle": "Seychelles", "seychelles": "Seychelles",
    "maurice": "Mauritius", "ile maurice": "Mauritius",
    "madagascar": "Madagascar", "cap vert": "Cape Verde",
    "afrique du sud": "South Africa", "maroc": "Morocco",
}

def extract_plan_params(state: AgentState) -> AgentState:
    q = state["query"]

    m = _DAY_RE.search(q)
    state["plan_days"] = int(m.group(1)) if m else 3

    m = _BUDGET_RE.search(q)
    state["plan_budget"] = (m.group(1) or m.group(2)).replace(",", "") if m else state.get("plan_budget") or "1500"

    # Known African countries — scanned directly from text (no preposition needed)
    KNOWN_COUNTRIES = [
        "Egypt", "Algeria", "Tunisia", "Morocco", "Senegal", "Ethiopia",
        "Tanzania", "Kenya", "Ghana", "Nigeria", "Zimbabwe", "Zambia",
        "Uganda", "Rwanda", "Mozambique", "Namibia", "Botswana",
        "Seychelles", "Mauritius", "Madagascar", "Cape Verde",
        "South Africa", "Ivory Coast", "Cameroon", "Angola",
    ]

    def _find_country(text: str) -> str:
        """Return first known country found in text (case-insensitive)."""
        text_lower = text.lower()
        # Check aliases first
        for alias, canonical in DEST_ALIASES.items():
            if alias in text_lower:
                return canonical
        # Then check canonical names
        for country in KNOWN_COUNTRIES:
            if country.lower() in text_lower:
                return country
        # Fallback: regex with prepositions
        m2 = _DEST_RE.search(text)
        return m2.group(1).strip().title() if m2 else ""

    # 1. Try current query first
    dest = _find_country(q)

    # 2. If not found, scan full conversation history (most recent first)
    if not dest:
        for msg in reversed(state.get("history", [])):
            dest = _find_country(msg["content"])
            if dest:
                break

    state["plan_destination"] = dest or state.get("plan_destination") or "Egypt"

    prefs = re.sub(r"\b(\d+\s*days?|\d+\s*nights?|\$\d[\d,]*|plan|itinerary|trip|travel|create|make|build)\b",
                   "", q, flags=re.I).strip(" ,.-")
    state["plan_preferences"] = prefs or "general sightseeing"
    return state


# ── Node 3: planner ────────────────────────────────────────────────────────
def run_planner(state: AgentState) -> AgentState:
    # If user asked about budget without specifying days, default to 3
    days   = state["plan_days"]   or 3
    budget = state["plan_budget"] or "1500"
    dest   = state["plan_destination"] or "Egypt"

    result = build_itinerary(
        destination=dest,
        days=days,
        preferences=state["plan_preferences"],
        budget=budget,
        passport="Moroccan",
    )
    state["plan_itinerary"] = state["response"] = result or f"Sorry, couldn't generate an itinerary for {dest}. Please try again."
    return state


# ── Node 4: refine ─────────────────────────────────────────────────────────
def run_refine(state: AgentState) -> AgentState:
    result = refine_itinerary(state["plan_itinerary"], state["query"])
    state["plan_itinerary"] = state["response"] = result
    return state


# ── Node 5: summarize ──────────────────────────────────────────────────────
def run_summarize(state: AgentState) -> AgentState:
    state["response"] = summarize_itinerary(state["plan_itinerary"])
    return state


# ── Node 6: recommend ──────────────────────────────────────────────────────
def get_recommendations(state: AgentState) -> AgentState:
    query = state["query"]

    # Enrich vague follow-ups with previous context
    if any(s in query.lower() for s in ["more", "other", "else", "another", "different", "also"]):
        for msg in reversed(state.get("history", [])):
            if msg["role"] == "assistant":
                query += " " + msg["content"][:200]
                break

    raw = retrieve(query)

    if not raw or len(raw) < 30:
        state["response"] = "I couldn't find matching destinations. Try: 'safe beach without visa' or 'budget safari East Africa'."
        return state

    # Split RAW DATA into recommendations vs context
    rec_lines, ctx_lines = [], []
    section = ""
    for line in raw.splitlines():
        if line.startswith("==="):
            section = line
        elif "RECOMMENDED" in section:
            rec_lines.append(line)
        else:
            ctx_lines.append(line)

    rec_text = "\n".join(rec_lines).strip()
    ctx_text = "\n".join(ctx_lines).strip()

    # Detect budget question early — give Gemini a focused prompt instead of a multi-rule prompt
    q_lower = state["query"].lower()
    is_budget_q = any(w in q_lower for w in ["budget", "cost", "price", "how much", "expensive", "afford", "money"])

    if is_budget_q:
        # Find the country being discussed from history
        country_ctx = ""
        for msg in reversed(state.get("history", [])):
            if msg["role"] == "assistant" and len(msg["content"]) > 30:
                country_ctx = msg["content"][:400]
                break

        budget_messages = [
            {"role": "system", "content": (
                "You are TravelMind AI. The user is asking about the budget for a destination "
                "already discussed in the conversation.\n"
                "Reply ONLY in this exact format, nothing else:\n\n"
                "Flights from Morocco: ~$X\n"
                "Accommodation: ~$X/night\n"
                "Food: ~$X/day\n"
                "Transport: ~$X total\n"
                "Activities: ~$X total\n"
                "Estimated total (X days): ~$X\n\n"
                "[One sentence: what affects the price most]\n\n"
                f"Context from conversation:\n{country_ctx}"
            )},
            {"role": "user", "content": state["query"]},
        ]
        state["response"] = call_llm(budget_messages, max_tokens=200) or "I couldn't estimate the budget right now. Please try again."
        return state

    messages = [
        {"role": "system", "content": (
            "You are TravelMind AI, a helpful travel assistant for Moroccan travelers exploring Africa.\n"
            "Never mention Morocco — that is where the traveler is FROM.\n\n"
            "RESPONSE RULES:\n"
            "- Greeting: introduce yourself warmly, ask what kind of trip. No destinations.\n"
            "- Recommendations: ONE line per destination: [Country]: why it fits. (visa status). Short closing question.\n"
            "- Interest in a specific country: one rich paragraph — highlights, visa, best time, one tip. Ask if they want a full plan.\n"
            "- Follow-up: answer directly and concisely.\n"
            "- Never use markdown, bullets, or bold text. Plain sentences only.\n\n"
            "RECOMMENDED COUNTRIES:\n"
            f"{rec_text}\n\n"
            "ADDITIONAL CONTEXT:\n"
            f"{ctx_text}"
        )},
        *state.get("history", []),
        {"role": "user", "content": state["query"]},
    ]

    state["response"] = call_llm(messages, max_tokens=1024) or "I found some matches but couldn't format a response. Please try again."
    return state


# ── Node 8: qa — answer a direct question about a destination in context ────
def handle_qa(state: AgentState) -> AgentState:
    # Build context from last few assistant messages
    history  = state.get("history", [])
    ctx      = "\n".join(m["content"][:300] for m in history[-6:] if m["role"] == "assistant")

    messages = [
        {"role": "system", "content": (
            "You are TravelMind AI. The user is asking a specific question about a destination "
            "already discussed in the conversation.\n"
            "Answer directly and concisely using the conversation context below.\n"
            "Give concrete numbers where possible (prices in USD, scores, dates).\n"
            "Plain sentences only — no markdown, no bullets. Max 5 sentences.\n\n"
            f"CONVERSATION CONTEXT:\n{ctx}"
        )},
        {"role": "user", "content": state["query"]},
    ]
    state["response"] = call_llm(messages, max_tokens=300) or "Could you clarify what you'd like to know?"
    return state


# ── Node 9: off-topic ──────────────────────────────────────────────────────
def handle_off_topic(state: AgentState) -> AgentState:
    state["response"] = call_llm([
        {"role": "system", "content": "You are TravelMind AI. Only handle African travel for Moroccan travelers. Redirect off-topic warmly. Max 30 words."},
        {"role": "user",   "content": state["query"]},
    ], max_tokens=60) or "I only help with African travel! What destination are you curious about?"
    return state


# ── Graph ──────────────────────────────────────────────────────────────────
def route(state: AgentState) -> Literal["plan", "refine", "summarize", "recommend", "qa", "off_topic"]:
    return state["intent"]

def create_agent():
    g = StateGraph(AgentState)
    g.add_node("classify",  classify_query)
    g.add_node("extract",   extract_plan_params)
    g.add_node("plan",      run_planner)
    g.add_node("refine",    run_refine)
    g.add_node("summarize", run_summarize)
    g.add_node("recommend", get_recommendations)
    g.add_node("qa",        handle_qa)
    g.add_node("off_topic", handle_off_topic)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {
        "plan": "extract", "refine": "refine", "summarize": "summarize",
        "recommend": "recommend", "qa": "qa", "off_topic": "off_topic",
    })
    g.add_edge("extract",   "plan")
    g.add_edge("plan",      END)
    g.add_edge("refine",    END)
    g.add_edge("summarize", END)
    g.add_edge("recommend", END)
    g.add_edge("qa",        END)
    g.add_edge("off_topic", END)
    return g.compile()


# ── Chatbot class ──────────────────────────────────────────────────────────
class TravelMindAgent:
    def __init__(self):
        self.agent      = create_agent()
        self.history    = []
        self.plan_state = {
            "plan_destination": "",
            "plan_days":        3,
            "plan_budget":      "1500",
            "plan_preferences": "",
            "plan_itinerary":   "",
        }

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        result = self.agent.invoke({
            "query":    user_input,
            "response": "",
            "intent":   "",
            "history":  list(self.history[:-1]),
            **self.plan_state,
        })

        response = result.get("response") or "Sorry, I couldn't generate a response."

        for key in self.plan_state:
            if result.get(key):
                self.plan_state[key] = result[key]

        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return response

    def reset(self):
        self.history    = []
        self.plan_state = {"plan_destination": "", "plan_days": 3,
                           "plan_budget": "1500", "plan_preferences": "", "plan_itinerary": ""}


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌍 TravelMind AI\n")
    bot = TravelMindAgent()
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input: continue
        if user_input.lower() in ["quit", "exit"]: break
        if user_input.lower() in ["reset", "new chat"]:
            bot.reset(); print("Reset."); continue
        print("TravelMind:", bot.chat(user_input), "\n")'''

# agent.py

import os, sys, re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import requests

from config import GEMINI_API_KEY, LLM_MODEL
from tools.retriever import retrieve
from tools.planner import build_itinerary, refine_itinerary, summarize_itinerary


# ── Gemini call ────────────────────────────────────────────────────────────
def call_llm(messages: list, max_tokens: int = 1024) -> str:
    """Convert OpenAI-style messages to Gemini format and call the API."""
    system, contents = "", []
    for m in messages:
        role, text = m["role"], m["content"]
        if role == "system":
            system += text + "\n\n"
        elif role in ("user", "human"):
            combined = (system + text).strip() if system and not contents else text
            contents.append({"role": "user",  "parts": [{"text": combined}]})
            system = ""
        elif role in ("assistant", "ai", "model"):
            contents.append({"role": "model", "parts": [{"text": text}]})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": system.strip()}]}]

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent",
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY},
        json={"contents": contents,
              "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── State ──────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:            str
    response:         str
    intent:           str
    history:          list
    plan_destination: str
    plan_days:        int
    plan_budget:      str
    plan_preferences: str
    plan_itinerary:   str


# ── Node 1: classify ───────────────────────────────────────────────────────
def classify_query(state: AgentState) -> AgentState:
    has_itinerary = bool(state.get("plan_itinerary", "").strip())
    history       = state.get("history", [])
    recent        = "\n".join(f"{m['role'].upper()}: {m['content'][:120]}" for m in history[-4:])

    prompt = (
        "You are an intent classifier. Reply with ONE word only.\n\n"
        "Intents:\n"
        "- recommend  : travel advice, destination suggestions, greetings, follow-ups\n"
        "- plan       : user wants a day-by-day itinerary OR asks about budget/cost/price for a destination\n"
        "- refine     : user wants to change an existing itinerary\n"
        "- summarize  : user wants a summary/recap of an existing itinerary\n"
        "             Examples: 'summarize', 'give me a summary', 'recap', 'give me summary for the X trip',\n"
        "             'remind me of the plan', 'what did we plan', 'overview of my trip'\n"
        "- off_topic  : nothing to do with travel\n\n"
        "Examples → plan: 'what is the budget?', 'how much will it cost?', 'yes what about the budget', "
        "'give me a plan', 'plan 3 days in Kenya', 'how much does it cost to go there'\n"
        f"Has active itinerary: {has_itinerary} (refine/summarize need this to be True)\n"
        + (f"\nRecent conversation:\n{recent}\n" if recent else "")
        + f"\nMessage: {state['query']}"
    )

    # Pre-check: detect summary/refine intent without LLM to avoid misclassification
    q_lower = state["query"].lower()
    summary_words = ["summary", "summarize", "recap", "remind me", "what did we plan",
                     "overview", "give me a summary", "what was the plan", "what we planned"]
    refine_words  = ["change", "modify", "update", "instead", "replace", "swap",
                     "make day", "day 1", "day 2", "day 3", "day 4", "day 5",
                     "first day", "second day", "third day", "last day"]

    if has_itinerary and any(w in q_lower for w in summary_words):
        state["intent"] = "summarize"
        return state
    if has_itinerary and any(w in q_lower for w in refine_words):
        state["intent"] = "refine"
        return state

    try:
        raw    = call_llm([{"role": "user", "content": prompt}], max_tokens=5).lower().split()[0]
        valid  = {"recommend", "plan", "refine", "summarize", "off_topic"}
        intent = raw if raw in valid else "recommend"
        if intent in ("refine", "summarize") and not has_itinerary:
            intent = "recommend"
    except Exception:
        intent = "recommend"

    state["intent"] = intent
    return state


# ── Node 2: extract plan params ────────────────────────────────────────────
_DAY_RE    = re.compile(r"\b(\d+)\s*(?:days?|nights?)\b", re.I)
_BUDGET_RE = re.compile(r"\$\s*(\d[\d,]*)|(\d[\d,]*)\s*(?:usd|dollars?)", re.I)
_DEST_RE   = re.compile(
    r"\b(?:to|for|in|visit(?:ing)?|travel(?:ing)?\s+to)\s+([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+)?)",
    re.I
)
DEST_ALIASES = {
    "algerie": "Algeria", "egypte": "Egypt", "tunisie": "Tunisia",
    "maroc": "Morocco",   "senegal": "Senegal", "ethiopie": "Ethiopia",
    "tanzanie": "Tanzania", "zambie": "Zambia", "ouganda": "Uganda",
    "namibie": "Namibia",  "botswana": "Botswana",
    "seychelle": "Seychelles", "seychelles": "Seychelles",
    "maurice": "Mauritius", "ile maurice": "Mauritius",
    "madagascar": "Madagascar", "cap vert": "Cape Verde",
    "afrique du sud": "South Africa", "maroc": "Morocco",
}

def extract_plan_params(state: AgentState) -> AgentState:
    q = state["query"]

    m = _DAY_RE.search(q)
    state["plan_days"] = int(m.group(1)) if m else 3

    m = _BUDGET_RE.search(q)
    state["plan_budget"] = (m.group(1) or m.group(2)).replace(",", "") if m else state.get("plan_budget") or "1500"

    # Known African countries — scanned directly from text (no preposition needed)
    KNOWN_COUNTRIES = [
        "Egypt", "Algeria", "Tunisia", "Morocco", "Senegal", "Ethiopia",
        "Tanzania", "Kenya", "Ghana", "Nigeria", "Zimbabwe", "Zambia",
        "Uganda", "Rwanda", "Mozambique", "Namibia", "Botswana",
        "Seychelles", "Mauritius", "Madagascar", "Cape Verde",
        "South Africa", "Ivory Coast", "Cameroon", "Angola",
    ]

    def _find_country(text: str) -> str:
        """Return first known country found in text (case-insensitive)."""
        text_lower = text.lower()
        # Check aliases first
        for alias, canonical in DEST_ALIASES.items():
            if alias in text_lower:
                return canonical
        # Then check canonical names
        for country in KNOWN_COUNTRIES:
            if country.lower() in text_lower:
                return country
        # Fallback: regex with prepositions
        m2 = _DEST_RE.search(text)
        return m2.group(1).strip().title() if m2 else ""

    # 1. Try current query first
    dest = _find_country(q)

    # 2. If not found, scan full conversation history (most recent first)
    if not dest:
        for msg in reversed(state.get("history", [])):
            dest = _find_country(msg["content"])
            if dest:
                break

    state["plan_destination"] = dest or state.get("plan_destination") or "Egypt"

    prefs = re.sub(r"\b(\d+\s*days?|\d+\s*nights?|\$\d[\d,]*|plan|itinerary|trip|travel|create|make|build)\b",
                   "", q, flags=re.I).strip(" ,.-")
    state["plan_preferences"] = prefs or "general sightseeing"
    return state


# ── Node 3: planner ────────────────────────────────────────────────────────
def run_planner(state: AgentState) -> AgentState:
    # If user asked about budget without specifying days, default to 3
    days   = state["plan_days"]   or 3
    budget = state["plan_budget"] or "1500"
    dest   = state["plan_destination"] or "Egypt"

    result = build_itinerary(
        destination=dest,
        days=days,
        preferences=state["plan_preferences"],
        budget=budget,
        passport="Moroccan",
    )
    state["plan_itinerary"] = state["response"] = result or f"Sorry, couldn't generate an itinerary for {dest}. Please try again."
    return state


# ── Node 4: refine ─────────────────────────────────────────────────────────
def run_refine(state: AgentState) -> AgentState:
    result = refine_itinerary(state["plan_itinerary"], state["query"])
    state["plan_itinerary"] = state["response"] = result
    return state


# ── Node 5: summarize ──────────────────────────────────────────────────────
def run_summarize(state: AgentState) -> AgentState:
    state["response"] = summarize_itinerary(state["plan_itinerary"])
    return state


# ── Node 6: recommend ──────────────────────────────────────────────────────
def get_recommendations(state: AgentState) -> AgentState:
    query = state["query"]

    # Enrich vague follow-ups with previous context
    if any(s in query.lower() for s in ["more", "other", "else", "another", "different", "also"]):
        for msg in reversed(state.get("history", [])):
            if msg["role"] == "assistant":
                query += " " + msg["content"][:200]
                break

    raw = retrieve(query)

    if not raw or len(raw) < 30:
        state["response"] = "I couldn't find matching destinations. Try: 'safe beach without visa' or 'budget safari East Africa'."
        return state

    # Split RAW DATA into recommendations vs context
    rec_lines, ctx_lines = [], []
    section = ""
    for line in raw.splitlines():
        if line.startswith("==="):
            section = line
        elif "RECOMMENDED" in section:
            rec_lines.append(line)
        else:
            ctx_lines.append(line)

    rec_text = "\n".join(rec_lines).strip()
    ctx_text = "\n".join(ctx_lines).strip()

    # Detect budget question early — give Gemini a focused prompt instead of a multi-rule prompt
    q_lower = state["query"].lower()
    is_budget_q = any(w in q_lower for w in ["budget", "cost", "price", "how much", "expensive", "afford", "money"])

    if is_budget_q:
        # Find the country being discussed from history
        country_ctx = ""
        for msg in reversed(state.get("history", [])):
            if msg["role"] == "assistant" and len(msg["content"]) > 30:
                country_ctx = msg["content"][:400]
                break

        budget_messages = [
            {"role": "system", "content": (
                "You are TravelMind AI. The user is asking about the budget for a destination "
                "already discussed in the conversation.\n"
                "Reply ONLY in this exact format, nothing else:\n\n"
                "Flights from Morocco: ~$X\n"
                "Accommodation: ~$X/night\n"
                "Food: ~$X/day\n"
                "Transport: ~$X total\n"
                "Activities: ~$X total\n"
                "Estimated total (X days): ~$X\n\n"
                "[One sentence: what affects the price most]\n\n"
                f"Context from conversation:\n{country_ctx}"
            )},
            {"role": "user", "content": state["query"]},
        ]
        state["response"] = call_llm(budget_messages, max_tokens=200) or "I couldn't estimate the budget right now. Please try again."
        return state

    messages = [
        {"role": "system", "content": (
            "You are TravelMind AI, a helpful travel assistant for Moroccan travelers exploring Africa.\n"
            "Never mention Morocco — that is where the traveler is FROM.\n\n"
            "RESPONSE RULES:\n"
            "- Greeting: introduce yourself warmly, ask what kind of trip. No destinations.\n"
            "- Recommendations: ONE line per destination: [Country]: why it fits. (visa status). Short closing question.\n"
            "- Interest in a specific country: one rich paragraph (max 80 words) — highlights, visa, best time, one tip. Ask if they want a full plan.\n"
            "- Follow-up: answer directly and concisely.\n"
            "- Never use markdown, bullets, or bold text. Plain sentences only.\n\n"
            "RECOMMENDED COUNTRIES:\n"
            f"{rec_text}\n\n"
            "ADDITIONAL CONTEXT:\n"
            f"{ctx_text}"
        )},
        *state.get("history", []),
        {"role": "user", "content": state["query"]},
    ]

    state["response"] = call_llm(messages, max_tokens=1500) or "I found some matches but couldn't format a response. Please try again."
    return state


# ── Node 8: qa — answer a direct question about a destination in context ────
def handle_qa(state: AgentState) -> AgentState:
    # Build context from last few assistant messages
    history  = state.get("history", [])
    ctx      = "\n".join(m["content"][:300] for m in history[-6:] if m["role"] == "assistant")

    messages = [
        {"role": "system", "content": (
            "You are TravelMind AI. The user is asking a specific question about a destination "
            "already discussed in the conversation.\n"
            "Answer directly and concisely using the conversation context below.\n"
            "Give concrete numbers where possible (prices in USD, scores, dates).\n"
            "Plain sentences only — no markdown, no bullets. Max 5 sentences.\n\n"
            f"CONVERSATION CONTEXT:\n{ctx}"
        )},
        {"role": "user", "content": state["query"]},
    ]
    state["response"] = call_llm(messages, max_tokens=300) or "Could you clarify what you'd like to know?"
    return state


# ── Node 9: off-topic ──────────────────────────────────────────────────────
def handle_off_topic(state: AgentState) -> AgentState:
    state["response"] = call_llm([
        {"role": "system", "content": "You are TravelMind AI. Only handle African travel for Moroccan travelers. Redirect off-topic warmly. Max 30 words."},
        {"role": "user",   "content": state["query"]},
    ], max_tokens=60) or "I only help with African travel! What destination are you curious about?"
    return state


# ── Graph ──────────────────────────────────────────────────────────────────
def route(state: AgentState) -> Literal["plan", "refine", "summarize", "recommend", "qa", "off_topic"]:
    return state["intent"]

def create_agent():
    g = StateGraph(AgentState)
    g.add_node("classify",  classify_query)
    g.add_node("extract",   extract_plan_params)
    g.add_node("plan",      run_planner)
    g.add_node("refine",    run_refine)
    g.add_node("summarize", run_summarize)
    g.add_node("recommend", get_recommendations)
    g.add_node("qa",        handle_qa)
    g.add_node("off_topic", handle_off_topic)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {
        "plan": "extract", "refine": "refine", "summarize": "summarize",
        "recommend": "recommend", "qa": "qa", "off_topic": "off_topic",
    })
    g.add_edge("extract",   "plan")
    g.add_edge("plan",      END)
    g.add_edge("refine",    END)
    g.add_edge("summarize", END)
    g.add_edge("recommend", END)
    g.add_edge("qa",        END)
    g.add_edge("off_topic", END)
    return g.compile()


# ── Chatbot class ──────────────────────────────────────────────────────────
class TravelMindAgent:
    def __init__(self):
        self.agent      = create_agent()
        self.history    = []
        self.plan_state = {
            "plan_destination": "",
            "plan_days":        3,
            "plan_budget":      "1500",
            "plan_preferences": "",
            "plan_itinerary":   "",
        }

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        result = self.agent.invoke({
            "query":    user_input,
            "response": "",
            "intent":   "",
            "history":  list(self.history[:-1]),
            **self.plan_state,
        })

        response = result.get("response") or "Sorry, I couldn't generate a response."

        for key in self.plan_state:
            if result.get(key):
                self.plan_state[key] = result[key]

        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        return response

    def reset(self):
        self.history    = []
        self.plan_state = {"plan_destination": "", "plan_days": 3,
                           "plan_budget": "1500", "plan_preferences": "", "plan_itinerary": ""}


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌍 TravelMind AI\n")
    bot = TravelMindAgent()
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input: continue
        if user_input.lower() in ["quit", "exit"]: break
        if user_input.lower() in ["reset", "new chat"]:
            bot.reset(); print("Reset."); continue
        print("TravelMind:", bot.chat(user_input), "\n")