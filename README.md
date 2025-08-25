<div align="center">

# 🏡 Rihayish - RAG
An AI‑powered Real Estate Assistant for Bahria Town Phase 8 (extensible) with Retrieval Augmented Generation, semantic + rule filters, structured property enrichment, image handling, and Gemini LLM integration.

</div>

---

## ✨ Key Features
* Hybrid Retrieval: Title keyword filtering + dense embeddings fallback (ChromaDB)
* Intelligent Query Analysis: Detects price ranges (crores), bedrooms, property type (house/apartment), size (marla / sqft), statistical vs search intent, casual greetings, malformed queries
* Price Range Parsing: Handles forms like `2 to 3 crore`, `under 5 crore`, `<= 4 crores`, `budget 3 crore`
* Mild Relevance‑Preserving Shuffling: Slight randomized ordering inside same relevance tier (prevents monotony, keeps top highly relevant result stable)
* Image Support: Scraped listing image galleries with enlarged modal viewer (wide layout)
* Conversation Handling: Session chat with auto reset on page refresh (configurable)
* Gemini Integration: Uses `gemini-1.5-flash` for normal answers; auto‑upgrades to `gemini-1.5-pro` for analytical/statistical queries
* Robust Guardrails: Input validation, fallback paths, graceful handling of empty data / failed model calls
* Modular Scripts: Separate scrape → clean/enrich → embed → query pipeline

---

## 📁 Project Structure (Core)
```
scripts/
	scrape_listings.py        # Scrape raw listings & images
	clean_and_enrich.py       # (If used) Normalize / enrich raw JSON
	embed_and_store.py        # Build ChromaDB vector store
	query_rag.py              # Core RAG + query analysis pipeline
	utils.py                  # Shared helpers
web_ui/
	app.py                    # Flask web interface (chat + cards + images)
	static/                   # JS/CSS for gallery & chat
	templates/chat.html       # Chat + results layout
data/
	raw/                      # Raw scraped JSON (ignored by .gitignore)
	processed/                # Cleaned/enriched property JSON
chromadb_data/              # Persistent vector DB (ignored)
logs/                       # Rotating app logs
```

---

## 🚀 Quick Start

### 1. Clone & Environment
```powershell
git clone <your-repo-url>
cd PropertyGuru
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Set Environment Variables (PowerShell)
```powershell
$Env:GRAANA_GEMINI_API_KEY = "YOUR_GEMINI_KEY"   # or GEMINI_API_KEY
$Env:SECRET_KEY = "dev-secret-change"
```

### 3. Scrape (Example: 1 page, Phase 8)
```powershell
python scripts/scrape_listings.py --max-pages 1 --start-url "https://www.graana.com/sale/residential-properties-sale-bahria-town-phase-8-rawalpindi-3-258/?pageSize=5&page=1"
```

### 4. (Optional) Enrich / Clean
```powershell
python scripts/clean_and_enrich.py
```

### 5. Build Embeddings
```powershell
python scripts/embed_and_store.py --collection zameen_listings
```

### 6. Run Web UI
```powershell
python web_ui/app.py
```
Navigate to: http://127.0.0.1:5000

### 7. CLI Query Examples
```powershell
python scripts/query_rag.py --query "show me 10 marla houses under 5 crore" --collection zameen_listings
python scripts/query_rag.py --query "2 to 3 crore 5 marla houses" --collection zameen_listings
python scripts/query_rag.py --query "average price of 10 marla houses" --collection zameen_listings
```

---

## 🧠 Query Understanding Highlights
| Capability | Example | Parsed Action |
|------------|---------|---------------|
| Price range (crore) | `2 to 3 crore` | min=2, max=3 |
| Upper bound | `under 5 crore` | max=5 |
| Lower bound | `above 3 crore` | min=3 |
| Bedrooms | `3 bed` / `3 bedroom` | bedrooms=3 |
| Property type | `apartments` / `houses` | property_type=apartment/house |
| Size | `10 marla`, `2500 sqft` | size filter attempt |
| Statistical intent | `average`, `compare`, `how many` | Switch to analytical mode |
| Casual | `hello`, `hi`, `help` | Greeting response |

Graceful fallbacks ensure queries still attempt a retrieval even if partial parsing fails.

---

## 🔍 Retrieval Flow
```
User Query
	└─► Validation & Intent Detection
				├─ Casual? → Greeting reply
				├─ Statistical? → Dataset slice → Gemini Pro analysis
				└─ Search Flow
						 ├─ Title-based filter + scoring
						 ├─ (If empty) Embedding similarity via ChromaDB
						 ├─ Apply numeric & categorical filters
						 ├─ Mild tier-constrained shuffle
						 └─ Format listings + narrative answer
```

---

## 🖼 Image Gallery
* Property cards show thumbnails
* Clicking opens a wide modal (responsive) with grid layout
* Supports scrolling and high-resolution views

---

## 🧪 Testing / Sanity
Minimal test scaffold in `tests/`. Example quick programmatic check:
```powershell
python -c "from scripts.query_rag import rag_infer;import os;print(rag_infer(query='show me 10 marla houses under 5 crore')['mode'])"
```

Add more pytest cases for: price extraction, statistical classification, no-results suggestions.

---

## ⚙️ Configuration & Env Vars
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| GRAANA_GEMINI_API_KEY / GEMINI_API_KEY | Gemini API key | Yes (LLM) | None |
| SECRET_KEY | Flask session key | Yes (web) | dev-key | 
| COLLECTION_NAME (in config.py) | Chroma collection label | No | zameen_listings |
| EMBEDDING_MODEL | SentenceTransformers model | No | all-mpnet-base-v2 |

---

## 🛠 Developer Notes
* Logging: Rotating file logs in `logs/`. Adjust levels in `web_ui/app.py`.
* Conversation reset: Currently clears on page refresh (see index route). Remove that if persistence desired.
* Shuffling: Controlled inside `rag_infer` (mild tail adjustments) + tie-group shuffle in `title_based_search`.
* Performance: Simple in‑process embedding model cache + query cache stub (`_QUERY_CACHE`).

---

## ☁ Deployment (Render Example)
```text
gunicorn -c gunicorn.conf.py web_ui.app:app
```
Persistent disk: mount for `chromadb_data/` so embeddings survive redeploys.

Health endpoint: `/health` returns JSON (vector DB presence, conversation count, timestamp).

Scale carefully: each worker loads embedding model memory.

---

## 🔐 Security Practices
* Never commit `.env`, virtual env, raw data dumps (already in `.gitignore`).
* Rotate API keys periodically.
* Consider rate limiting (Flask middleware) if exposed publicly.
* Add input length caps to mitigate prompt abuse.



---

## 🙌 Contributing
Pull requests welcome. Please:
1. Add/adjust tests where behavior changes
2. Run lint / sanity query before submitting
3. Keep README section anchors intact

---

## ❓ Help
```powershell
python scripts/query_rag.py --help
```
If something fails: enable debug prints or raise logging level to DEBUG.

---

Happy building! 🏗
