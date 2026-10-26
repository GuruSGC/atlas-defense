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

## Learned components and the v4 benchmark

The rule-based signals only know the wordings someone thought of, so v4 adds learned components and
a benchmark that does not come from the people who wrote the rules.

| Part | Where | Notes |
|---|---|---|
| Local sentence embeddings | `core/embeddings.py` | `EMBEDDING_BACKEND=local` uses `all-MiniLM-L6-v2` (384 dimensions, CPU). `hash` and Gemini still work. |
| Poisoning classifier | `atlas/learned.py` | A fine-tuned MiniLM classifier (or a logistic regression on frozen embeddings), calibrated with Platt scaling. It sets an upper bound on behavioural consistency. `BEHAV_MODE=regex|learned|hybrid`. |
| Profile-fitted anomaly detector | `atlas/anomaly_detector.py` | The isolation forest or a kNN-distance detector can start from real benign embeddings (`set_profile`) instead of 50 random vectors. `ANOMALY_DETECTOR=iforest|knn`. |
| NLI historical consistency | `atlas/signals.py`, `atlas/nli.py` | Related memories are found by embedding similarity and an NLI cross-encoder scores contradiction. `HIST_MODE=clusters|nli|hybrid`. |

```bash
cd backend
pip install -r requirements-dev.txt -r requirements-optional.txt
python -m evaluation.run_v4 data          # downloads the public sources once (needs network)
python -m evaluation.run_v4 classifier    # trains both classifiers on the train split (about 8 minutes for the fine-tuned one)
python -m evaluation.run_v4 anomaly history ladder baselines sensitivity
python -m evaluation.run_v4 adaptive prb llm latency    # these use the live model or take longer
python -m evaluation.run_v4 compose       # writes evaluation/results/benchmark_v4.json for the dashboard
```

Trained models go to `models/` (git-ignored): `models/poison_clf` is the fine-tuned classifier and `models/poison_logreg` the lightweight one. The public sources are fetched by `evaluation/public_data.py` into `evaluation/data/public/`, which is also git-ignored. Without the trained models the behavioural signal uses the rules alone. The
leave-one-source-out run (`run_v4 loso`) retrains the fine-tuned classifier six times and takes about an hour on a
CPU. Every step writes its own file under `evaluation/results/v4/`, so one step can be rerun on its own.

## Tests and reports

```bash
pip install -r requirements-dev.txt
python -m pytest -q                         # runs offline, no API key needed
python -m evaluation.run_report --seeds 10  # writes evaluation/results/latest.json
```
