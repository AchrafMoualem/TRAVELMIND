# 🌍 TravelMind AI
**Your Personal African Travel Assistant for Moroccan Travelers**

TravelMind AI is an intelligent conversational travel assistant that helps Moroccan travelers discover African destinations, plan detailed day-by-day itineraries, and get personalized visa and budget information — all through a simple chat interface.

---

## ✨ Features

- **Smart Destination Recommender** — semantic search with hard filters for safety, visa ease, and affordability
- **Day-by-Day Itinerary Planner** — full trip plans with real budget breakdowns tailored to your preferences
- **Itinerary Refinement** — modify any day of your plan through natural conversation
- **Trip Summarizer** — get a clean recap of any planned itinerary
- **RAG Knowledge Base** — ChromaDB vector store with destination guides and visa data for 50+ African countries
- **Context-Aware Conversations** — remembers what you discussed across turns

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent Framework | LangGraph |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
TRAVELMIND/
├── agent.py              # LangGraph agent — intent classification and routing
├── app.py                # Flask web server
├── config.py             # API keys and model configuration
├── tools/
│   ├── recommender.py    # Semantic destination ranking with hard filters
│   ├── retriever.py      # ChromaDB retrieval — no LLM call
│   └── planner.py        # Itinerary builder, refiner, and summarizer
├── data/                 # Raw destination documents and metadata CSV
├── vectorstore/          # ChromaDB persistent vector store
├── static/
│   └── index.html        # Chat UI — African jungle theme
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone git@github.com:AchrafMoualem/TRAVELMIND.git
cd TRAVELMIND
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your free Gemini API key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 4. Build the vector store
```bash
python ingest.py
```

### 5. Run the app
```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## 💬 Example Conversations

```
You: I like historical sites
TravelMind: Algeria: Ancient Roman ruins in an off-the-beaten-path setting. (visa-required)
            Ethiopia: Rock-hewn churches and ancient kingdoms. (e-visa)
            Egypt: Pyramids, temples, and pharaonic civilization. (visa-free)

You: Egypt sounds good
TravelMind: Egypt is a captivating destination for history lovers...

You: Plan a 3 day trip
TravelMind: TRIP OVERVIEW — Egypt, 3 days, $1155 USD...

You: Change day 2 to a Nile cruise
TravelMind: [Updated itinerary with changes made]

You: Give me a summary
TravelMind: Destination: Egypt | Duration: 3 days | Total: $1155...
```

---

## 🔧 Configuration

In `config.py`:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL      = "gemini-2.5-flash"
EMBED_MODEL    = "all-MiniLM-L6-v2"
MAX_TOKENS     = 2048
TOP_K_RESULTS  = 2
TOP_K_DESTINATIONS = 3
```

---

## ⚠️ Limitations

- Free Gemini API tier: 20 requests/min, 200 requests/day
- Destination data is static — visa rules and prices may change
- Scoped to African destinations only
- Flight prices are estimates, not live booking data
- No user accounts — history is session-based

---

## 👨‍💻 Author

**Achraf Moualem** — Capstone Project

---

## 📄 License

This project is for educational purposes as part of a capstone project.
