<p align="center">
  <img src="https://content.umami.is/website/images/umami-logo.png" alt="Umami Logo" width="100">
</p>

<h1 align="center">Umami — AI-Enhanced Analytics</h1>

<p align="center">
  <i>Umami is a simple, fast, privacy-focused alternative to Google Analytics.</i>
  <br/>
  <b>This fork adds GPU-accelerated ML, graph-based visitor journey analysis, time-series optimization, and a public ML API with API key authentication.</b>
</p>

<p align="center">
  <a href="https://github.com/umami-software/umami/releases"><img src="https://img.shields.io/github/release/umami-software/umami.svg" alt="GitHub Release" /></a>
  <a href="https://github.com/umami-software/umami/blob/master/LICENSE"><img src="https://img.shields.io/github/license/umami-software/umami.svg" alt="MIT License" /></a>
  <a href="https://github.com/umami-software/umami/actions"><img src="https://img.shields.io/github/actions/workflow/status/umami-software/umami/ci.yml" alt="Build Status" /></a>
</p>

---

## ✨ What's New in This Fork

| Feature | Description |
|---|---|
| **PostgreSQL 18** | Upgraded from PG15 to PG18 with uuidv7(), AIO subsystem, skip scan |
| **TimescaleDB 2.27** | 7 hypertables with auto-partitioning, continuous aggregates, data retention |
| **Apache AGE 1.7** | Graph database for visitor journey analysis via openCypher |
| **pgvector + pgvectorscale** | 3 StreamingDiskANN indexes (256d GRU, 384d MiniLM, 4096d Qwen3) |
| **9 ML Models (GPU)** | Next-page, funnel drop-off, session intent, recommender, rage-click, journey clustering, contextual bandit, A/B testing, session replay analysis |
| **AI Insights Dashboard** | ML-powered recommendations and predictions with Train/Refresh/Sync buttons |
| **3 Recommendation Modes** | Token (GRU), Semantic (MiniLM/Qwen3), Hybrid — toggleable from dashboard |
| **2 Embedding Models** | MiniLM (384d, local) and Qwen3 (4096d, local or cloud API) |
| **BYOK Cloud Embedding** | Users bring their own API URL + key for cloud-based Qwen3 embedding |
| **Public ML API v1** | API-key authenticated endpoints for third-party access |
| **Self-Service API Keys** | Generate, list, and revoke API keys from Settings > ML API Keys |
| **User Preferences** | Per-user settings for rec mode, embedding model/mode, API URL/key, hybrid weight — all persisted server-side |
| **AGE Graph Recommendations** | Path-based recommendations via Apache AGE openCypher |
| **Auto-Indexing** | Semantic embeddings synced automatically every 15 min via cron |
| **Contextual Bandit** | Online learning for real-time content recommendations |
| **GPU Acceleration** | RTX 3060 (PyTorch 2.12) + Tesla P40 (ONNX/XGBoost) with CPU fallback |

---

## 🏗 Architecture

```
umami App (Next.js port 3000)          ML Service (FastAPI port 8001)
┌─────────────────────────────┐       ┌──────────────────────────────────────┐
│  /api/ml/* route handlers   │──────►│  /predict/next-page  (Markov+Trans)  │
│  (auth + proxy to 8001)     │       │  /predict/funnel-drop (XGBoost GPU)  │
│  /api/ml/v1/* public API   │       │  /predict/intent     (BERT+RF)       │
│  (API key auth)            │       │  /recommend          (GRU+ANN+AGE)   │
│  AI Insights dashboard     │       │  /predict/rage-click (Isolation)     │
│  User preferences API     │       │  /predict/cluster   (UMAP+HDBSCAN)   │
│  API Key management       │       │  /recommend/bandit  (LinUCB)          │
└─────────────────────────────┘       │  /ab-test/*         (Chi-squared)   │
                                      │  /predict/session-replay (rrweb)    │
                                      │  /train/*            (GPU loop)    │
                                      │  /embeddings/sync    (ONNX+ST)     │
                                      │  /health              (GPU info)   │
                                      └───────────┬──────────────────────────┘
                                                  │
                        ┌─────────────────────────┼──────────────────────┐
                        │                         │                      │
                   ┌────▼────┐           ┌────────▼───────┐   ┌─────────▼──┐
                   │ PG18    │           │ TimescaleDB    │   │ Apache AGE │
                   │ 18.4    │           │ 7 hypertables │   │ umami_     │
                   │ Users   │           │ Continuous     │   │ analytics  │
                   │ Teams   │           │ aggregates     │   │ graph      │
                   │ Reports │           │ Data retention │   │ openCypher │
                   └─────────┘           └────────────────┘   └────────────┘
                                                │
                                          ┌─────▼─────┐
                                          │ pgvector  │
                                          │ 3 x DiskANN │
                                          │ (256+384+4096)│
                                          └───────────┘
```

---

## 🚀 Getting Started

### Requirements

- A server with Node.js version 18.18+
- **PostgreSQL 18** with **TimescaleDB 2.27+**, **Apache AGE 1.7+**, **pgvector 0.8+**, **pgvectorscale 0.9+**
- NVIDIA GPU (optional, for ML acceleration): RTX 3060+ or Tesla P40+

### Quick Start with Docker

```bash
docker compose up -d
```

### Manual Setup

```bash
# 1. Install PostgreSQL 18 extensions
sudo apt-get install postgresql-18 postgresql-18-timescaledb postgresql-18-age postgresql-18-pgvector pgvectorscale-postgresql-18

# 2. Configure shared_preload_libraries
echo "shared_preload_libraries = 'timescaledb,age'" >> /etc/postgresql/18/main/postgresql.conf

# 3. Restart PG18 and create database
sudo pg_ctlcluster 18 main restart
sudo -u postgres psql -c "CREATE ROLE umami WITH LOGIN PASSWORD 'umami' CREATEDB;"
sudo -u postgres psql -c "CREATE DATABASE umami OWNER umami;"

# 4. Enable extensions
sudo -u postgres psql -d umami -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
sudo -u postgres psql -d umami -c "CREATE EXTENSION IF NOT EXISTS age CASCADE;"
sudo -u postgres psql -d umami -c "CREATE EXTENSION IF NOT EXISTS vector CASCADE;"
sudo -u postgres psql -d umami -c "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;"
sudo -u postgres psql -d umami -c "SELECT * FROM ag_catalog.create_graph('umami_analytics');"

# 5. Install umami
git clone https://github.com/iskandarsulaili/umami.git
cd umami
pnpm install
cp .env.example .env  # Edit DATABASE_URL
pnpm run build
pnpm run start
```

---

## 🤖 ML Service

The ML service runs as a separate FastAPI server on port 8001 with GPU acceleration.

### Start the ML Service

```bash
# Create virtual environment and install dependencies
python3 -m venv ml/.venv
source ml/.venv/bin/activate
pip install -r ml/requirements.txt

# Start the API server
python3 ml/run.py
```

### Recommended Modes

| Mode | Description | How It Works |
|---|---|---|
| **Token** | URL pattern matching | GRU neural network on page URL sequences. Fast, works with minimal data. |
| **Semantic** | Page meaning understanding | Embedding model (MiniLM or Qwen3) + pgvector DiskANN ANN search. |
| **Hybrid** | Best quality | Weighted fusion of Token + Semantic. Slider-adjustable weight per user. |

### Embedding Models

| Model | Dimensions | Modes | Requirements |
|---|---|---|---|
| **MiniLM** (default) | 384 (configurable) | Local only | 80MB, always available via SentenceTransformer |
| **Qwen3** | 4096 (configurable) | Local or Cloud | ~2GB for local, or API URL + key for cloud (BYOK) |

### Embedding Modes

| Mode | Description | Configuration |
|---|---|---|
| **Local** | Runs on GPU/CPU via SentenceTransformer | No additional setup needed for MiniLM. Qwen3 requires HuggingFace token. |
| **Cloud** | OpenAI-compatible API (Together AI, Alibaba Cloud, etc.) | Set API URL + Key in dashboard (per-user, BYOK). Falls back to local MiniLM on failure. |

### ML Models (9 total)

| # | Model | Algorithm | GPU | Training Data |
|---|---|---|---|---|
| 1 | **NextPagePredictor** | Markov chain + Transformer | PyTorch (RTX 3060) | Session page sequences |
| 2 | **FunnelPredictor** | XGBoost (hist tree) | CUDA (P40 via ONNX) | Session features + labels |
| 3 | **SessionIntentClassifier** | SentenceTransformer + RandomForest | PyTorch (RTX 3060) | Page URLs + session metadata |
| 4 | **Recommender** | GRU encoder + ANN (pgvectorscale) + AGE graph | PyTorch (RTX 3060) | Session page sequences |
| 5 | **RageClickDetector** | Isolation Forest + rule-based | CPU (sklearn) | Heatmap click coordinates |
| 6 | **JourneyClusterer** | UMAP + HDBSCAN | CPU (sklearn) | Session feature vectors |
| 7 | **ContextualBandit** | LinUCB (online learning) | CPU (numpy) | Real-time feedback |
| 8 | **ABTestFramework** | Chi-squared significance | CPU (scipy) | Experiment results |
| 9 | **SessionReplayAnalyzer** | rrweb event analysis | CPU | Session replay events |

### Training Pipeline

```bash
# Train all models for a specific website (last 30 days)
python3 ml/train_all.py --website <website_id> --days 30

# Or via the dashboard: click "Train All Models"
# Or via API:
curl -X POST http://localhost:8001/train/next-page \
  -H 'Content-Type: application/json' \
  -d '{"website_id":"<id>","start_date":"2026-06-01","end_date":"2026-07-01"}'
```

### Semantic Embedding Sync

```bash
# Sync embeddings via dashboard: click "Sync Embeddings"
# Or via API:
curl -X POST http://localhost:8001/embeddings/sync \
  -H 'Content-Type: application/json' \
  -d '{"website_id":"<id>","model":"minilm","mode":"local"}'
```

Auto-indexing runs every 15 min via the AGE sync cron job.

### Scheduled Jobs

| Job | Schedule | Function |
|---|---|---|
| `umami-age-sync` | Every 15 min | Sync TimescaleDB data to Apache AGE graph + auto-index semantic embeddings |
| `umami-ml-train` | Daily 3 AM | Retrain all ML models from latest data |

---

## 🔑 Public ML API (v1)

Third-party services can access ML predictions without a Umami login session.

See [`docs/ml-api.md`](docs/ml-api.md) for complete documentation.

**Quick example:**

```bash
# 1. Generate an API key (from Settings > ML API Keys)
curl -X POST https://your-umami.com/api/admin/ml/keys \
  -H "authorization: Bearer YOUR_ADMIN_TOKEN" \
  -d '{"websiteId":"YOUR_WEBSITE_ID","name":"My App"}'

# 2. Use the key
curl -X POST https://your-umami.com/api/ml/v1/recommend \
  -H "x-api-key: umami_ml_YOUR_KEY" \
  -d '{"sessionPages":["/","/pricing"],"topK":5,"mode":"hybrid"}'
```

---

## 🖥 AI Insights Dashboard

The AI Insights page is available in the website navigation under **Behavior > AI Insights**.

### Controls

| Control | Description |
|---|---|
| **Train All Models** | Retrain all 9 ML models from latest session data |
| **Sync Embeddings** | Generate semantic embeddings for all known pages (model/mode-aware) |
| **Refresh Insights** | Re-fetch all ML predictions |
| **Rec Mode** | Token / Semantic / Hybrid |
| **Weight Slider** | Adjust Token⇄Semantic blend (Hybrid mode only) |
| **Model** | MiniLM (384d) / Qwen3 (4096d) — switching warns about incompatible embeddings |
| **Mode** | Local / Cloud — Cloud shows API URL + Key inputs (BYOK) |

### Panels

- **Recommended Pages** — content suggestions based on session context
- **Predicted Next Pages** — most likely pages the visitor will view next
- **Session Intent** — automatic intent classification
- **Funnel Drop-off Risk** — real-time probability of abandonment
- **Frustration Detection** — rage clicks, dead clicks, mouse shaking
- **Visitor Archetype** — unsupervised journey clustering
- **Bandit Recommendations** — online learning for content recs
- **Session Replay Analysis** — UX signal detection from rrweb
- **Web Vitals Performance** — LCP, CLS, INP, FCP, TTFB
- **A/B Test Results** — statistical experiment outcomes
- **ML System Health** — GPU info, model status, embeddings status

---

## 🗄 Database Layer

### PostgreSQL 18 Extensions

| Extension | Version | Purpose |
|---|---|---|
| `timescaledb` | 2.27.1 | Time-series hypertables, continuous aggregates, retention |
| `age` | 1.7.0 | Graph database with openCypher query language |
| `vector` | 0.8.4 | Vector column type for embeddings |
| `vectorscale` | 0.9.0 | StreamingDiskANN index with SBQ compression |
| `pgcrypto` | 1.4 | Cryptographic functions (umami requirement) |
| `pg_stat_statements` | 1.12 | Query performance monitoring |

### TimescaleDB Hypertables

| Table | Chunk Interval | Retention | Space Partition |
|---|---|---|---|
| `website_event` | 1 day | 2 years | website_id (4) |
| `event_data` | 1 day | 2 years | website_id (4) |
| `session_data` | 1 day | 2 years | website_id (4) |
| `session` | 1 day | 2 years | website_id (4) |
| `session_replay` | 7 days | 1 year | website_id (2) |
| `heatmap_event` | 1 day | 1 year | website_id (4) |
| `revenue` | 7 days | 2 years | — |

### pgvectorscale Indexes

```sql
-- GRU token embeddings (256d)
CREATE INDEX idx_page_embeddings_ann ON page_embeddings 
  USING diskann (embedding vector_cosine_ops);

-- MiniLM semantic embeddings (384d)
CREATE INDEX idx_page_embeddings_semantic_ann ON page_embeddings 
  USING diskann (semantic_embedding vector_cosine_ops);

-- Qwen3 embeddings (4096d)
CREATE INDEX idx_page_embeddings_qwen3_ann ON page_embeddings 
  USING diskann (qwen3_embedding vector_cosine_ops);
```

### Apache AGE Graph

The `umami_analytics` graph stores:
- **Page vertices** — each unique URL path
- **LEADS_TO edges** — weighted page transitions (count)
- **openCypher queries** — path traversal, centrality, recommendations

```cypher
-- Most common page transitions
MATCH (p1:Page)-[r:LEADS_TO]->(p2:Page)
RETURN p1.url, p2.url, r.count
ORDER BY r.count DESC LIMIT 10
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string (port 5433) |
| `ML_API_URL` | `http://localhost:8001` | ML service endpoint |
| `ML_USE_GPU` | `true` | Enable GPU acceleration |
| `ML_GPU_ID` | auto | Specific GPU to use |
| `ML_MODEL_DIR` | `ml/models/saved` | Model storage path |
| `ML_API_PORT` | `8001` | ML API server port |
| `AGE_ENABLED` | `true` | Enable Apache AGE graph |
| `EMBEDDING_MODEL` | `minilm` | Default embedding model (`minilm` or `qwen3`) |
| `EMBEDDING_MODE` | `local` | Default embedding mode (`local` or `cloud`) |
| `EMBEDDING_DIM_MINILM` | `384` | MiniLM embedding dimension |
| `EMBEDDING_DIM_QWEN3` | `4096` | Qwen3 embedding dimension |
| `EMBEDDING_API_URL` | — | Fallback cloud embedding API URL (env-level) |
| `EMBEDDING_API_KEY` | — | Fallback cloud embedding API key (env-level) |

---

## 📦 Library Versions

| Library | Version | Notes |
|---|---|---|
| PyTorch | 2.12.1 | CUDA 13.0, latest stable |
| XGBoost | 3.3.0 | GPU support via CUDA |
| scikit-learn | 1.9.0 | Latest stable |
| umap-learn | 0.5.12 | Dimensionality reduction |
| hdbscan | 0.8.44 | Hierarchical clustering |
| ONNX Runtime | 1.27.0 | GPU via CUDA + TensorRT |
| Transformers | 5.12.1 | HuggingFace ecosystem |
| Sentence-Transformers | 5.6.0 | Text embeddings |
| FastAPI | 0.139.0 | API framework |
| Pandas | 3.0.3 | Data processing |

---

## 🛟 Support

<p align="center">
  <a href="https://github.com/iskandarsulaili/umami"><img src="https://img.shields.io/badge/GitHub--blue?style=social&logo=github" alt="GitHub" /></a>
  <a href="https://umami.is/discord"><img src="https://img.shields.io/badge/Discord--blue?style=social&logo=discord" alt="Discord" /></a>
</p>
