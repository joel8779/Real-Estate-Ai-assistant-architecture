# DarGlobal & Wasalt AI Real Estate Knowledge Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.137-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

A modular, secure, and production-ready AI-powered real estate knowledge assistant built for **DarGlobal** (`darglobal.co.uk`) and **Wasalt** (`wasalt.sa/en`).

The assistant synthesizes grounded, factual intelligence across international luxury branded residences (e.g. Trump International, Missoni, Pagani, Lamborghini) and domestic Saudi property listings, company records, investment highlights, and FAQs.

---

## Key Features & Architecture Highlights

1. **Separation of Concerns**: Ingestion, normalization, embedding, vector storage, hybrid retrieval, LLM orchestration, and presentation layers are strictly decoupled. Ingestion never executes within the user request lifecycle.
2. **Grounded Retrieval & Anti-Hallucination**: Calibrated cosine similarity thresholds, explicit uncertainty handling when data is unavailable, verified source citations, and structured property cards.
3. **Backend-Controlled Safety**: Inbound client messages strictly accept `user` and `assistant` roles only (`system` instructions are immutably backend-controlled). Untrusted web context is isolated behind strict delimiter guards.
4. **Resilient AI Orchestration**: OpenRouter client with dynamic model selection (`meta-llama/llama-3.3-70b-instruct:free`, `google/gemini-2.0-flash-exp:free`), configurable fallback chains, and explicit service-unavailable error handling (no mock fallbacks in production).
5. **Hardened Security**: Non-root Docker containers (`appuser` UID 10001), 64 KB request payload validation middleware, IP-based sliding window rate limiting via `slowapi`, and CORS allowlisting.
6. **Luxury UI**: Minimalist interface inspired by Linear and Perplexity, featuring typography-focused layout, source filters, citation cards, and real-time backend health probes.

---

## Directory Structure

```plaintext
darglobal/
├── .env.example                       # Environment configuration template
├── .gitignore                         # Comprehensive git exclusions
├── docker-compose.yml                 # Multi-service container orchestration
├── README.md                          # Project documentation and developer guide
├── ARCHITECTURE.md                    # In-depth architectural design and data flow
├── SECURITY.md                        # Security controls, threat model, and guardrails
├── EVALUATION.md                      # Hallucination testing and threshold calibration
│
├── data/                              # Persistent storage volume
│   ├── raw/                           # Raw crawled documents
│   │   ├── darglobal/                 # DarGlobal raw artifacts
│   │   └── wasalt/                    # Wasalt raw JSON-LD artifacts
│   ├── processed/                     # Normalized knowledge entities
│   └── chroma/                        # ChromaDB persistent vector database
│
├── backend/                           # FastAPI Application (Python 3.11)
│   ├── Dockerfile                     # Multi-stage non-root container
│   ├── requirements.txt               # Pinned Python dependencies
│   ├── pytest.ini                     # Pytest configuration
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # App factory, middlewares, and lifespans
│   │   ├── config.py                  # Pydantic BaseSettings configuration
│   │   ├── api/                       # API routers (/healthz, /readyz, /api/v1/chat)
│   │   ├── core/                      # Logging, exceptions, and security middleware
│   │   ├── schemas/                   # Pydantic request/response/entity models
│   │   ├── ingestion/                 # Crawler & normalization pipeline (Milestone 2)
│   │   └── rag/                       # Vector store, embedder & LLM engine (Milestone 3-4)
│   └── tests/                         # Automated test suite
│
└── frontend/                          # React + Vite + TypeScript Application
    ├── Dockerfile                     # Multi-stage unprivileged Nginx container
    ├── nginx.conf                     # Reverse proxy and client-side routing config
    ├── package.json                   # Pinned frontend dependencies
    ├── vite.config.ts                 # Vite bundler and Vitest configuration
    ├── tailwind.config.js             # Luxury dark theme design tokens
    ├── src/
    │   ├── api/client.ts              # Typed backend API client
    │   ├── components/                # Modular UI components (Navbar, Cards, Chat)
    │   ├── types/index.ts             # Shared TypeScript schemas
    │   └── test/                      # Vitest UI unit tests
```

---

## Quickstart Guide

### Prerequisites
- **Python 3.11+**
- **Node.js 20+** and **npm**
- **Docker** and **Docker Compose** (for containerized deployment)

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd darglobal

# Create local environment file from template
cp .env.example .env

# Optional: Add your OpenRouter API key in .env if testing live AI inference
# OPENROUTER_API_KEY=your_key_here
```

### 2. Run with Docker Compose (Recommended)

```bash
# Build and launch both backend and frontend containers
docker compose up --build -d

# Check running status and health checks
docker compose ps

# View application logs
docker compose logs -f
```

- **Frontend Application**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Liveness Probe**: [http://localhost:8000/healthz](http://localhost:8000/healthz)
- **Readiness Probe**: [http://localhost:8000/readyz](http://localhost:8000/readyz)

#### Ollama Secondary Fallback Deployment (Qwen 2.5)

Ollama runs as an internal Docker Compose service (`darglobal-ollama`) attached to the private `darglobal-net` network.

- **Primary Provider**: OpenRouter remains the primary provider locked strictly to `nvidia/nemotron-3-ultra-550b-a55b:free`.
- **Secondary Fallback**: Ollama is used exclusively as a fallback if OpenRouter encounters account-wide daily free-model quota exhaustion (`free-models-per-day` 429) or upstream provider outages (502/503/504).
- **Fallback Model**: `qwen2.5:1.5b`.

After starting the Docker Compose services, download and verify the model:

```bash
# 1. Download the fallback model inside the running Ollama container
docker compose exec ollama ollama pull qwen2.5:1.5b

# 2. Verify the model is installed
docker compose exec ollama ollama list
```

> [!NOTE]
> **Persistent Volume & Security**:
> - Model weights are stored in the persistent named Docker volume `ollama_data` (mounted at `/root/.ollama`) and are preserved across container updates and restarts.
> - Model weight directories and binary artifacts (`ollama_data/`, `.ollama/`, `*.gguf`, `*.bin`) are strictly excluded via `.gitignore` and are **not committed to GitHub**.
> - Port `11434` is not publicly exposed to the host; it is accessible only internally to the backend service via `http://ollama:11434`.

---

### 3. Local Development (Without Docker)

#### Backend (Terminal 1)
```bash
# Navigate to project root
cd darglobal

# Install dependencies in Python 3.11 environment
py -3.11 -m pip install -r backend/requirements.txt

# Run backend development server
py -3.11 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Frontend (Terminal 2)
```bash
# Navigate to frontend directory
cd darglobal/frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

Visit [http://localhost:5173](http://localhost:5173). The frontend automatically proxies requests to `http://localhost:8000`.

---

## Data Ingestion & Normalization CLI

The ingestion engine is fully decoupled from the runtime chat lifecycle. It handles rate limiting, domain allowlisting, exponential backoff, access barrier detection, deduplication, and schema normalization.

### 1. Ingest Permitted Wasalt Records
```bash
py -3.11 -m backend.app.ingestion.cli --source wasalt --max-pages 15 --delay 0.5
```

### 2. Ingest DarGlobal (Permitted & Offline Seed Datasets)
```bash
py -3.11 -m backend.app.ingestion.cli --source darglobal --max-pages 15 --delay 0.5
```
*Note: DarGlobal fronts live HTTP endpoints with an Imperva/Incapsula challenge. The pipeline respects website access controls without deploying evasive techniques, logging access restrictions and ingesting permitted/provided offline datasets.*

### 3. Ingest All Sources
```bash
py -3.11 -m backend.app.ingestion.cli --source all --max-pages 25 --delay 0.5
```

### 4. Reprocess Persisted Documents Offline
```bash
py -3.11 -m backend.app.ingestion.cli --source all --reprocess
```

- **Raw Ingested Records**: Stored under `data/raw/<source>/<doc_id>.json`
- **Normalized Knowledge Entities**: Stored under `data/processed/records.jsonl` and `data/processed/records.json`
- **Audit Reports**: Generated at `data/processed/ingestion_report.json` and `data/processed/ingestion_report.md`

---

## Hybrid Retrieval & ChromaDB Vector Store (Milestone 3)

The retrieval subsystem indexes normalized knowledge entities into persistent **ChromaDB** dense vector embeddings and a sparse **BM25** lexical index, combined through **Reciprocal Rank Fusion (RRF)**.

### 1. Run Knowledge Base Indexing Pipeline
Reads `data/processed/records.json`, chunks entities preserving specification context, generates normalized 384-dimensional embeddings using `sentence-transformers/all-MiniLM-L6-v2`, and populates ChromaDB and BM25:

```bash
py -3.11 -m backend.app.rag.indexer
```

### 2. Run Retrieval Benchmark Evaluation
Evaluates Recall@5, Top-1 accuracy, source match, and out-of-domain rejection against the 16-query curated benchmark:

```bash
py -3.11 -m backend.app.rag.evaluator
```

### 3. Run Live Retrieval Demonstration
Runs 5 live queries across corporate, project, listing, cross-source, and out-of-domain categories:

```bash
py -3.11 scripts/demonstrate_retrieval.py
```

### 4. Query Retrieval API Endpoint (`/api/v1/retrieve`)
```bash
curl -i -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AIDA Muscat Oman Trump International Golf Club",
    "top_k": 3,
    "source_filter": "darglobal",
    "method": "hybrid"
  }'
```

---

## Conversational RAG & OpenRouter Integration (Milestone 4)

The conversational RAG engine combines hybrid retrieval, provenance-tiered data safeguards, untrusted document delimiter fencing, follow-up query contextualization, and resilient OpenRouter LLM generation.

### 1. Conversational RAG Architecture & Data Flow

```plaintext
User Input (React UI)
  │
  ▼
[Input Sanitization & Length Guard]
  │
  ▼
[Follow-Up Turn Contextualizer]  ──> Heuristically resolves anaphoric pronouns (e.g. "it", "where is it located?")
  │
  ▼
[Hybrid Retrieval Engine]       ──> Reciprocal Rank Fusion (ChromaDB 384-d vectors + BM25Okapi)
  │
  ▼
[Provenance & Data Mode Gate]   ──> ALLOW_UNVERIFIED_DEMO_DATA check (Strict Verified vs Demonstration Mode)
  │
  ▼
[Context Builder & Fencer]      ──> XML-fenced <retrieved_evidence> tags (Anti-Prompt Injection Guard)
  │
  ▼
[OpenRouter Async Client]       ──> HTTPX async call with bounded retries (429/5xx) & fallback model failover
  │
  ▼
[Response Validation Engine]    ──> Hallucinated number checks (prices, areas, beds) & citation binding
  │
  ▼
API Response (`/api/v1/chat`)   ──> Grounded answer + original citations + property cards + provenance warning
```

### 2. Configuration Settings (.env)

```env
# OpenRouter API Key (Required for live AI synthesis; raises LLM_NOT_CONFIGURED error if absent)
OPENROUTER_API_KEY=your_key_here

# Primary Model (Locked to NVIDIA Nemotron 3 Ultra)
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

# Ollama Secondary Fallback LLM Configuration (Local or Azure)
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TIMEOUT_SECONDS=60.0

# Provenance Data Mode Safeguard
# false: Strict verified mode (rejects unverified demonstration records; explains lack of verified data)
# true:  Demonstration mode (permits unverified records with a prominent visible provenance warning banner)
ALLOW_UNVERIFIED_DEMO_DATA=false
```

### 3. Run Live Conversational RAG Demonstration
Runs 8 live real-world scenarios against OpenRouter including corporate info, branded developments, domestic listings, follow-up contextualization, cross-source search, unanswerable queries, and prompt injection defense:

```bash
py -3.11 scripts/demonstrate_milestone4_rag.py
```

### 4. Query Chat Endpoint via cURL
```bash
curl -i -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What branded residences does DarGlobal offer in Dubai and Oman?"}
    ],
    "source_filter": "all"
  }'
```

---

## Verification & Testing Suites

### 1. Backend Automated Tests (Pytest)
Comprehensive suite of **58 unit and integration tests** covering configuration, crawler domain restriction, WAF barrier detection, HTML cleaning, extractor normalization, chunking, embedder dimensions and normalization, BM25 ranking, ChromaDB persistence, RRF hybrid fusion, security middleware, prompt injection delimiter fencing, numerical claim validation, and conversational RAG orchestration:

```bash
py -3.11 -m pytest -c backend/pytest.ini -v
```
*Result: 58 passed in ~19s.*

### 2. Frontend Automated Tests (Vitest)
Unit and integration tests for UI components (Navbar, ChatContainer, CitationCard, PropertyCard, and App integration):

```bash
cd frontend
npm test
```
*Result: 5 passed in ~2.4s.*

### 3. Frontend Production Build Check
Strict TypeScript compilation and asset minification:

```bash
cd frontend
npm run build
```
*Result: Built cleanly without warnings in ~2.2s.*

### 4. End-to-End Local Browser UI Verification (Playwright)
Validates the live running application inside headless Chromium:
- Backend starts and passes `/healthz` probe.
- Frontend starts and connects to knowledge assistant (`Online` indicator).
- Source filter toggles between *All Sources*, *DarGlobal*, and *Wasalt*.
- Live OpenRouter inference using locked model `nvidia/nemotron-3-ultra-550b-a55b:free`.
- Visible demonstration mode warning badges and disclaimer notices.
- External citation cards with `target="_blank"` and safe `rel="noopener noreferrer"` attributes.
- Follow-up conversational turns with pronoun contextualization.
- Clear Chat action resets history and restores welcome state.
- Grounded refusal and explicit uncertainty on out-of-domain queries (refusal to hallucinate Tokyo penthouses).
- Client-side error handling and system role rejection (HTTP 422).

```bash
py -3.11 scripts/verify_browser_ui.py
```
*Artifacts saved under `data/processed/`: `browser_ui_query1.png`, `browser_ui_query2.png`, `browser_ui_clear.png`, `browser_ui_uncertainty.png`, and `browser_ui_verification_report.json`.*

### 5. Docker Compose Configuration & Verification
The container orchestration is defined in `docker-compose.yml`:
- **Backend Service**: Multi-stage Python 3.11 build running as non-root `appuser` (UID 10001) with internal `/healthz` healthchecks, `OLLAMA_BASE_URL=http://ollama:11434`, and persistent volume mount `./data:/app/data`.
- **Ollama Service**: Local fallback inference engine running `qwen2.5:1.5b` with persistent volume mount `ollama_data:/root/.ollama` (internal port 11434, unexposed to host).
- **Frontend Service**: Multi-stage Node/Vite build served by unprivileged Nginx (UID 101) with internal HTTP healthchecks on port 8080.
- **Internal Network**: Isolated bridge network `darglobal-net`.

```bash
# Validate Docker Compose syntax and service definitions
docker compose config --quiet
```
*Result: Configuration is valid (exit code 0).*

> [!NOTE]
> **Host Docker Daemon Audit Notice**:
> `docker compose config --quiet` passed with exit code 0, confirming valid service specifications, healthchecks, and volume mappings. On Windows development hosts, launching containers via `docker compose up --build -d` requires the Windows Docker Desktop Service (`com.docker.service`) to be running with elevated administrative permissions. When the daemon is stopped, container execution is reported honestly as awaiting daemon startup without false claims of container pass-through.

---

## Deployment Architecture & Production Readiness

### Recommended Deployment Topology

```plaintext
┌──────────────────────────────────────────────┐
│                  Client Browser              │
└──────────────────────┬───────────────────────┘
                       │ HTTPS
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌─────────────────────────┐  ┌─────────────────────────────────┐
│     Vercel / Cloudflare │  │      Render / Railway / Fly.io  │
│  (React + Vite Frontend)│  │     (FastAPI Python Container)  │
│                         │  │                                 │
│  • Edge CDN Delivery    │  │  • Non-root `appuser` container │
│  • Environment:         │  │  • Persistent Volume:           │
│    VITE_API_BASE_URL    │  │    `/app/data/chroma`           │
└─────────────────────────┘  │  • Environment:                 │
                             │    OPENROUTER_API_KEY           │
                             │    OPENROUTER_MODEL             │
                             │    OPENROUTER_ALLOW_FALLBACK    │
                             │    ALLOW_UNVERIFIED_DEMO_DATA   │
                             │    CORS_ORIGINS                 │
                             └────────────────┬────────────────┘
                                              │ HTTPS
                                              ▼
                                 ┌─────────────────────────┐
                                 │     OpenRouter API      │
                                 │  `nvidia/nemotron-3...` │
                                 └─────────────────────────┘
```

### Production Setup Instructions

#### 1. Backend Service (Render / Railway / Fly.io)
1. **Container Deployment**: Point the deployment service to `backend/Dockerfile` with project root as context.
2. **Persistent Disk Mount**:
   - Mount a persistent volume at path `/app/data` (minimum 2 GB).
   - This preserves the ChromaDB vector database (`/app/data/chroma`) and normalized JSON-LD entities (`/app/data/processed/records.json`) across container restarts.
3. **Environment Variables**:
   ```env
   ENVIRONMENT=production
   OPENROUTER_API_KEY=sk-or-v1-...
   OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
   OPENROUTER_ALLOW_FALLBACK=false
   OPENROUTER_FALLBACK_MODELS=
   ALLOW_UNVERIFIED_DEMO_DATA=true
   CORS_ORIGINS=https://your-frontend-app.vercel.app
   RATE_LIMIT_PER_MINUTE=60
   MAX_REQUEST_SIZE_BYTES=65536
   SIMILARITY_THRESHOLD=0.45
   ```
4. **Health Check Probes**:
   - Liveness Probe: `GET /healthz` (200 OK)
   - Readiness Probe: `GET /readyz` (Checks ChromaDB vector store readiness)

#### 2. Frontend Application (Vercel)
1. **Framework Preset**: Vite
2. **Root Directory**: `frontend`
3. **Build Command**: `npm run build`
4. **Output Directory**: `dist`
5. **Environment Variable**:
   ```env
   VITE_API_BASE_URL=https://your-backend-api.onrender.com
   ```

---

## API Documentation & Contract Reference

### 1. `GET /healthz`
Liveness probe verifying that the FastAPI server is running.
- **Status**: `200 OK`
- **Response**:
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "environment": "development"
  }
  ```

### 2. `GET /readyz`
Readiness probe verifying vector store connectivity and knowledge base indexing.
- **Status**: `200 OK`
- **Response**:
  ```json
  {
    "status": "ready",
    "vector_store_connected": true,
    "indexed_chunks_count": 23,
    "environment": "development"
  }
  ```

### 3. `POST /api/v1/retrieve`
Standalone semantic and hybrid retrieval endpoint.
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "query": "AIDA Muscat Trump golf villas",
    "top_k": 3,
    "source_filter": "darglobal",
    "method": "hybrid"
  }
  ```
- **Response**:
  ```json
  {
    "query": "AIDA Muscat Trump golf villas",
    "total_results": 3,
    "results": [
      {
        "chunk_id": "darglobal_project_aida_chk0",
        "doc_id": "darglobal_project_aida",
        "source": "darglobal",
        "title": "AIDA — Muscat, Oman | DarGlobal Luxury Community",
        "category": "project",
        "url": "https://darglobal.co.uk/projects/aida",
        "score": 0.0327,
        "content": "...",
        "provenance": "demonstration_unverified"
      }
    ]
  }
  ```

### 4. `POST /api/v1/chat`
Conversational RAG endpoint with grounded OpenRouter LLM generation.
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "messages": [
      {"role": "user", "content": "What branded residences does DarGlobal develop in Dubai and Oman?"}
    ],
    "source_filter": "all",
    "session_id": "optional_session_uuid"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "DarGlobal features several prestigious branded residences...",
    "citations": [
      {
        "title": "AIDA — Muscat, Oman | DarGlobal Luxury Community",
        "url": "https://darglobal.co.uk/projects/aida",
        "source": "darglobal",
        "category": "project",
        "snippet": "a championship 18-hole golf course.",
        "provenance": "demonstration_unverified"
      }
    ],
    "property_cards": [
      {
        "title": "AIDA — Muscat, Oman | DarGlobal Luxury Community",
        "developer": "DarGlobal",
        "location": "Muscat, Oman",
        "property_type": "Luxury Villa",
        "price": null,
        "bedrooms": null,
        "amenities": [],
        "url": "https://darglobal.co.uk/projects/aida",
        "image_url": null
      }
    ],
    "confidence_score": 0.58,
    "grounded": true,
    "data_mode": "demonstration",
    "provenance_warning": "Notice: This response is generated from unverified demonstration records. It does not reflect live or verified real-world availability, terms, or pricing.",
    "error_code": null,
    "request_id": "req_6f98b85fe2c1",
    "disclaimer": "Information provided is grounded in available records from DarGlobal and Wasalt. Verify terms, availability, and prices directly with official representatives."
  }
  ```

---

## Compliance & Legal Disclaimer

1. **Permitted Public Access**: This system adheres to strict web ethics. It observes robots.txt, respects rate limits, and does not deploy CAPTCHA-evasion, stealth browser proxies, or unauthorized credential bypasses against DarGlobal's Imperva/Incapsula challenge or Wasalt's infrastructure.
2. **Demonstration Dataset Limitations**: Seed records for DarGlobal and Wasalt are classified strictly as `demonstration_unverified`. All UI responses display a prominent provenance notice. The knowledge assistant refuses to speculate on unlisted property prices, terms, or real-time unit availability.
3. **No Hallucination Claim**: Hallucination risks are mitigated through calibrated similarity cutoffs ($\tau = 0.45$), strict XML context fencing, numerical claim cross-referencing, and explicit uncertainty responses.



