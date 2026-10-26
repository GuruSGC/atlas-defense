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

The benchmark has 120 candidate memories (60 benign, 60 adversarial): 60 benign items sampled from PersonaChat (Zhang et al., 2018, ACL), a peer-reviewed public dialogue dataset, and 60 adversarial items organized across the five WhisperBench risk categories (health & safety, financial loss, information integrity, cybersecurity, operational disruption). Four further sets test other things: 40 held-out attacks, 40 technical benign statements, 20 legitimate corrections, and 30 paraphrased attacks written to avoid the detectors' wording. Provenance is in `backend/evaluation/data/README.md`.

### A correction to the first version of these numbers

The first benchmark built each candidate from its ground truth: attacks arrived on an untrusted source channel with category `instruction`, and benign items on the user channel with category `fact`. That gave the defense the answer through two of the six signals. The primary protocol now builds every candidate the same way, with source and category coming from the pipeline, and the old construction is kept only as a labeled ablation. With the original detectors the leakage-free recall was 63.3%, not the 78.3% first reported. The detectors were then reworked, so the headline numbers below are not comparable with the earlier ones.

### Results (primary protocol, offline hash embeddings, 95% intervals)

| Measure | Value |
|---|---|
| Precision | 100% (92.1% to 100%) |
| Recall | 75.0% (62.8% to 84.2%) |
| F1 | 85.7% (78.6% to 92.4%) |
| False-positive rate | 0% (0% to 6.0%) |
| AUC | 0.987 |

Other protocols and sets:

| Run | Result |
|---|---|
| Streaming (store grows, forest refits), 10 seeds | recall 81.8% (SD 2.8), F1 90.0% (SD 1.7), no false positives |
| Label-informed ablation | recall 76.7%, F1 86.8% |
| Held-out attacks (40) | 82.5% flagged, but see the caveat in the data README |
| Paraphrased attacks (30) | 6.7% flagged |
| Benign, PersonaChat (60) / technical (40) | 0% / 0% flagged |
| Legitimate corrections (20) | 10% flagged (2 of 20) |

Run once with live Gemini embeddings instead of the offline hash embedder, the primary protocol gave the same counts (45 / 15 / 0 / 60); the other protocols were run offline only. The static protocol is deterministic, so its seed sweep has no spread. The trust score is close to bimodal: moving the threshold anywhere from 0.30 to 0.70 changes nothing, and above 0.85 almost everything is quarantined.

### Baselines

Plain re-implementations of published ideas, run on the same items:

| Defense | Dev recall | Held-out recall | False positives (casual / technical / updates) |
|---|---|---|---|
| No defense | 0% | 0% | 0% / 0% / 0% |
| Static pattern moderation | 16.7% | 10.0% | 0% / 0% / 0% |
| Provenance gate, untrusted attacker | 100% | 100% | 0% / 0% / 0% |
| Provenance gate, attacker holds a valid signature | 0% | 0% | 0% / 0% / 0% |
| ATLAS (either attacker) | 75.0% | 82.5% | 0% / 0% / 10% |

A provenance gate is perfect against an attacker with no valid signature and useless against one who has it, which is what a conversational poisoner usually is. ATLAS does not depend on that assumption, at the cost of missing a quarter of the attacks.

### Attack Lab

Success requires the poisoned memory to be stored, retrieved, and to change the answer. With ATLAS on, three of five scenarios are blocked: fact overwrite, instruction injection, and the sybil persona. Two get through: gradual preference drift (each step looks harmless on its own) and context splicing (two fragments that are harmless apart). Per-memory scoring cannot see those, and fixing it would need scoring over groups of memories.

### Limitations

- The paraphrase result (6.7%) shows the detectors rely on wording. They are not a robust classifier.
- Plausible false facts with nothing stored to contradict cannot be caught from content alone.
- The datasets are small and mostly written by the authors, so the intervals are wide.
- An optional NLI contradiction check (`USE_NLI=true`) was tried and added no recall on this benchmark, so it stays off by default. NLI models label unrelated sentence pairs as contradictions, so it only helps when gated to pairs that share a topic.
- The baselines are our own re-implementations, not the original authors' code, and the three published attacks (MINJA, MemPoison, MemGhost) are imitated by their described mechanism, not reproduced.
- Memory stores are in-memory, SQLite, or FAISS. There is no hosted pgvector store.

### Reproducing

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest -q                          # offline, no API key
python -m evaluation.run_report --seeds 10   # writes evaluation/results/latest.json
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
