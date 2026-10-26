# ATLAS Backend

Adaptive Trust Management for Long-term Agent Storage: Research Prototype API.

## Requirements
- Python 3.10+
- FastAPI, Uvicorn, Pydantic, Scikit-learn, NumPy, HTTPX, Python-dotenv

## Setup Instructions

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy `.env.example` to `.env` and optionally set your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   *Note: If no Gemini API key is provided, the backend runs in local deterministic research prototype mode with full 6-signal ATLAS defense simulation.*

5. **Start the API Server**:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```

## Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | System health check and store telemetry |
| `POST` | `/api/chat` | Conversational turn with real-time ATLAS trust evaluation |
| `GET` | `/api/memories` | List persistent accepted and quarantined memories |
| `GET` | `/api/memories/{id}` | Inspect detailed memory item with 6-signal breakdown |
| `POST` | `/api/memories/analyze` | Diagnostic memory text analysis |
| `POST` | `/api/atlas/analyze` | Deep-dive signal analysis with custom weights |
| `GET` | `/api/attacks/scenarios` | List available poisoning attack scenarios |
| `POST` | `/api/attacks/run` | Execute attack simulation (ATLAS ON vs OFF) |
| `POST` | `/api/attacks/reset` | Reset testbed to baseline clean state |
| `GET` | `/api/evaluation/summary` | Retrieve empirical performance metrics (primary protocol, with intervals) |
| `POST` | `/api/evaluation/run` | Trigger fresh benchmark evaluation (`?live=true` uses the Gemini embedder) |
| `GET` | `/api/evaluation/report` | Full report: protocols, extra datasets, baselines, seed sweep, Attack Lab |
| `POST` | `/api/evaluation/report/run` | Regenerate the full report (`?seeds=10&live=false`) |

## Memory stores

Set `VECTOR_STORE_BACKEND` in `.env`:

| Value | Behaviour |
|---|---|
| `memory` (default) | In-process, lost on restart |
| `sqlite` | Persists to `VECTOR_STORE_PATH` (default `data/atlas_memory.db`) |
| `faiss` | SQLite plus two FAISS inner-product indexes (accepted and other); needs `pip install -r requirements-optional.txt` |

There is no hosted (pgvector or Supabase) store.

## Optional NLI contradiction check

`USE_NLI=true` adds a cross-encoder contradiction probability to the historical
consistency signal (`NLI_MODEL`, `NLI_GATE`, `NLI_THRESHOLD`). It is off by default
because on this benchmark it adds no recall; see the root README for the numbers.

## Tests and reports

```bash
pip install -r requirements-dev.txt
python -m pytest -q                         # runs offline, no API key needed
python -m evaluation.run_report --seeds 10  # writes evaluation/results/latest.json
```
