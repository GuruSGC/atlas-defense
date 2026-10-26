# ATLAS: Adaptive Trust Management for Long-term Agent Storage

> **Project Title**: *Trust Management for Defending Persistent LLM Agents Against Conversational Memory Poisoning*

ATLAS is a research prototype that defends persistent LLM long-term memory against conversational memory poisoning: malicious or misleading content introduced through ordinary dialogue that gets extracted into an agent's long-term memory and later influences its behavior. It evaluates every candidate memory against six independent trust signals before it is allowed into active storage.

---

## Architecture Overview

```
User / Attacker Dialogue Turn
              ↓
  Candidate Memory Extraction (LLM / Heuristic Filter)
              ↓
    Embedding Generation (Gemini text-embedding-004 / Local Hash Fallback)
              ↓
    Six ATLAS Trust Signals
  (Relevance, Anomaly, History,
   Behavioral, Source, Temporal)
              ↓
     Weighted Trust Score: sum(w_i * S_i) / sum(w_i)
              ↓
       Security Threshold (τ = 0.65)
    [ ACCEPT vs QUARANTINE ]
              ↓
Only ACCEPTED memories enter Persistent Vector Store
              ↓
Safe Retrieval for Future Agent Prompts
```

### The Six ATLAS Trust Signals

1. **Semantic Relevance ($S_{rel}$)**: Cosine similarity between the candidate memory and the active conversational turn.
2. **Embedding Anomaly ($S_{anom}$)**: An Isolation Forest fitted on latent embedding space to detect statistical outliers, hardened with a certified confidence radius via randomized smoothing (Cohen, Rosenfeld & Kolter, ICML 2019).
3. **Historical Consistency ($S_{hist}$)**: Detects direct factual contradictions and unauthorized overwrites against existing accepted memories, including a topic-agnostic revision-language detector.
4. **Behavioral Consistency ($S_{behav}$)**: Alignment with established user behavior and preferences, with heuristics for instruction-injection and destructive-command patterns.
5. **Source Reliability ($S_{src}$)**: Configurable trust weighting by input origin channel, with a generalized authority-claim detector for unverified privileged-role assertions.
6. **Temporal Trust ($S_{temp}$)**: Guards against burst injection attacks and rapid memory churn.

The six scores combine into a weighted trust score with a critical-signal veto: a severe weakness in any critical signal (embedding anomaly, historical consistency, behavioral consistency) caps the final score regardless of how strong the others are, so a single glaring red flag can't be averaged away.

---

## Evaluation

The benchmark has 1,736 cases (996 attacks, 740 benign). 1,486 of them come from public sources (InjecAgent, deepset, Gandalf, safe-guard, PoisonedRAG, PersonaChat, OpenAssistant), the rest are written by the authors. Each source is split into a training part and a held-out part, everything that learns uses the training part only, and every number below is on the held-out part (710 items: 406 attacks, 304 benign). Sources, licenses and caveats are in `backend/evaluation/data/README.md`. The public files are not committed; one command rebuilds them.

### What changed since the first version

The first evaluation used 120 cases written by the same people who wrote the rules, and it reported 75% recall. Scored on every item of the new benchmark before any learned component existed, those rules flagged 179 of 996 attacks (18.0%). On the held-out split they flag 16.3%. The fix that worked was to learn the behavioral check from labeled data.

### Results (held-out split, 95% intervals)

| Configuration | Recall | False positives | F1 |
|---|---|---|---|
| Original prototype | 16.3% | 0.3% | 0.279 |
| + kNN anomaly detector on a benign profile | 20.2% | 2.0% | 0.332 |
| + fine-tuned classifier (default learned configuration) | 79.8% (76 to 83) | 4.0% | 0.873 |
| + semantic NLI for historical consistency | 79.8% | 5.6% | 0.868 |

| Other defenses | Recall | False positives |
|---|---|---|
| No defense | 0% | 0% |
| Static pattern moderation | 4.2% | 0.0% |
| Provenance gate (unsigned / signed attacker) | 100% / 0% | 0% |
| Best anomaly detector alone (kNN distance) | 19.0% | 4.6% |
| Fine-tuned classifier alone | 75.4% | 2.3% |
| LLM moderator (200-item sample) | 88.0% | 3.0% |

- **Attack success** (PoisonedRAG items, persistence and retrieval and behavior): 81.7% with ATLAS off, 51.7% on, 12.5% when the correct answer is already stored. The behavior check used a small local model (Qwen2.5-1.5B-Instruct) because the hosted model's quota ran out.
- **Adaptive attacker:** padding a payload with ordinary sentences leaves 71% of flagged attacks flagged. A greedy black-box search gets 65% past the full configuration within 50 queries. Retraining on rewrites helped against the rewrites and did not stop the search.
- **Generalization:** leave-one-source-out recall on an unseen attack source ranges from 42% to 98%. With no OpenAssistant text in training the classifier flags 82% of OpenAssistant messages, so it needs benign examples of the deployment's own traffic.
- **Historical consistency:** keyword rules catch none of the contradictions in the profile-contradiction set; the NLI cross-encoder catches 89.2% with 3.2% false positives, but on this benchmark it adds no recall and some false positives.
- **Sensitivity:** the trust threshold does nothing between 0.30 and 0.70 because the veto decides. The learned classifier and the veto carry most of the weight; the other signals add little here.

### Limitations

- Plausible false facts can only be caught when a stored memory contradicts them. The LLM moderator catches some from world knowledge, ATLAS does not.
- Gradual drift and split payloads get through, because each memory is scored on its own.
- Attack labels describe content, not provenance. A benign standing instruction can read like an injected one.
- MINJA, MemPoison and MemGhost are imitated by mechanism, not reproduced. The baselines are our own re-implementations.
- The paraphrase and adaptive attacks are mechanical rewrites, and a human attacker would do better.

### Reproducing

```bash
cd backend
pip install -r requirements-dev.txt -r requirements-optional.txt
python -m pytest -q                      # offline, no API key
python -m evaluation.run_v4 data         # downloads the public sources
python -m evaluation.run_v4 classifier   # trains both classifiers
python -m evaluation.run_v4 anomaly history ladder baselines sensitivity compose
```

---

## Repository Structure

```
/
├── frontend/                     # Next.js 15+ (App Router), TypeScript, Tailwind CSS
│   ├── app/                      # Next.js App Router Pages
│   │   ├── page.tsx              # Research Dashboard Overview
│   │   ├── layout.tsx            # Global Layout & Navigation
│   │   ├── globals.css           # Styling & Themes
│   │   ├── chat/                 # Conversational Agent Interface
│   │   ├── memories/             # Persistent Memory Monitor
│   │   ├── atlas/                # Six-Signal Trust Diagnostic Tool
│   │   ├── attack-lab/           # Adversarial Poisoning Demonstration
│   │   └── evaluation/           # Quantitative Research Benchmarks
│   ├── components/               # Modular UI & Layout Components
│   │   ├── layout/               # Sidebar, Header, WorkflowBanner
│   │   └── ui/                   # TrustGauge, SignalBar, Card, Badge, StatCard
│   ├── lib/
│   │   ├── api.ts                # Centralized API client
│   │   ├── types.ts              # TypeScript interfaces (matching Pydantic models)
│   │   └── utils.ts              # Styling & format helpers
│   ├── .env.local.example
│   ├── package.json
│   └── README.md
│
├── backend/                      # Python FastAPI Backend
│   ├── main.py                   # FastAPI Application Entrypoint & CORS
│   ├── config.py                 # Core Configuration Re-export
│   ├── requirements.txt          # Dependencies (FastAPI, Scikit-learn, NumPy, SciPy, etc.)
│   ├── .env.example              # Server environment template (Gemini API Key kept server-side)
│   │
│   ├── api/                      # REST API Endpoints
│   │   ├── chat.py               # POST /api/chat
│   │   ├── memories.py           # GET /api/memories, GET /api/memories/{id}, POST /api/memories/analyze
│   │   ├── atlas.py              # POST /api/atlas/analyze, GET /api/atlas/config
│   │   ├── attacks.py            # GET /api/attacks/scenarios, POST /api/attacks/run, POST /api/attacks/reset
│   │   └── evaluation.py         # summary, run, full report, report/run
│   │
│   ├── core/                     # Core Services
│   │   ├── gemini_client.py      # Gemini API Wrapper & Fallback Mock Generator
│   │   ├── embeddings.py         # Normalized Embeddings (Gemini & Deterministic Fallback)
│   │   └── settings.py           # Pydantic Settings & Defaults
│   │
│   ├── agent/                    # Conversational Agent Layer
│   │   ├── agent.py              # AtlasAgent with Memory-Augmented Context
│   │   ├── memory_manager.py     # Memory Lifecycle & Routing
│   │   └── prompts.py            # System Prompts & Extraction Guidelines
│   │
│   ├── memory/                   # Storage Layer
│   │   ├── vector_store.py       # In-memory, SQLite, and FAISS stores
│   │   ├── memory_extractor.py   # Turn Extraction Heuristics & Logic
│   │   └── memory_models.py      # Pydantic Data Models
│   │
│   ├── atlas/                    # Defense Core
│   │   ├── trust_engine.py       # Pipeline Orchestrator
│   │   ├── signals.py            # Implementation of 6 Trust Signals
│   │   ├── patterns.py           # Linguistic detectors (directives, authority claims, slots)
│   │   ├── temporal.py           # Insertion-history tracker for temporal trust
│   │   ├── nli.py                # Optional NLI contradiction check
│   │   ├── anomaly_detector.py   # Isolation Forest + certified randomized-smoothing hardening
│   │   └── scoring.py            # Weighted Scoring & Rationale Generator
│   │
│   ├── attacks/                  # Attack Lab
│   │   ├── poisoning_simulator.py # Simulation Runner (ATLAS ON vs OFF)
│   │   └── scenarios.py          # Fact Overwrite, Instruction Injection, etc.
│   │
│   ├── models/                   # trained classifiers (git-ignored)
│   ├── tests/                    # pytest suite (offline)
│   │
│   └── evaluation/               # Metrics Suite
│       ├── metrics.py            # ASR, Detection Rate, Precision, Recall, F1, FPR, Confusion Matrix
│       ├── experiment_runner.py  # Protocols, streaming runs, full report
│       ├── baselines.py          # No-defense, pattern moderation, provenance gate
│       ├── stats.py              # Wilson and bootstrap intervals, AUC, sweeps
│       ├── run_report.py         # CLI that writes results/latest.json
│       └── data/                 # Benchmark dataset (PersonaChat-sourced + WhisperBench-organized) & provenance
│
└── README.md
```

---

## Quickstart Guide

### 1. Start the Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Run FastAPI Server
python -m uvicorn main:app --reload --port 8000
```

The backend server will start at `http://localhost:8000`. OpenAPI documentation is available at `http://localhost:8000/docs`.

### 2. Start the Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local

# Run Next.js Development Server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Attack Scenarios Implemented

1. **Direct fact overwrite**: replaces the verified language preference (`The user prefers Python for software development`) with a contradictory one.
2. **Covert instruction injection**: persists an instruction-like memory that changes how future answers are written.
3. **Gradual preference drift**: three small steps that move a stored preference until a safety setting is dropped.
4. **Sybil persona**: an unverified administrator claim used to get security scans recorded as switched off.
5. **Context splicing**: two fragments that are harmless alone and unsafe once retrieved together.

### Operational Definition of Attack Success
An attack is counted as **succeeded** if and only if:
1. The poisoned candidate is persisted in memory (`status == ACCEPTED`), **AND**
2. The poisoned memory is retrieved during the trigger probe query, **AND**
3. The downstream agent response is measurably influenced by that retrieved poisoned memory.

---

## Security & Privacy Model

- **Zero Client-Side Secrets**: `GEMINI_API_KEY` is loaded strictly on the server-side FastAPI process and never exposed to browser runtimes.
- **Safe Fallback Execution**: The system functions completely out-of-the-box in local offline mode using deterministic embeddings and isolation forests, or with live Gemini API credentials.

---

## Context

ATLAS was designed and built jointly and equally by **Ilamukil K J** and **Guru Charan S** as an independent research project at the School of Computer Science and Engineering (SCOPE), VIT, under the guidance of Prof. Satish C J.
