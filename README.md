<p align="center">
  <img src="https://content.umami.is/website/images/umami-logo.png" alt="Umami Logo" width="100">
</p>

<h1 align="center">Umami — AI-Enhanced Analytics</h1>

<p align="center">
  <i>Umami is a simple, fast, privacy-focused alternative to Google Analytics.</i>
  <br/>
  <b>This fork adds GPU-accelerated ML, graph-based visitor journey analysis, and time-series optimization.</b>
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
| **pgvector + pgvectorscale** | StreamingDiskANN index with SBQ compression for fast ANN search |
| **5 ML Models (GPU)** | Next-page predictor, funnel drop-off, session intent, recommender, rage-click detection |
| **AI Insights Dashboard** | ML-powered recommendations and predictions in the umami UI |
| **GPU Acceleration** | RTX 3060 (PyTorch) + Tesla P40 (ONNX/XGBoost) with CPU fallback |

---

## 🏗 Architecture

```
umami App (Next.js port 3000)          ML Service (FastAPI port 8001)
┌─────────────────────────────┐       ┌──────────────────────────────────┐
│  /api/ml/* route handlers   │──────►│  /predict/next-page  (Markov+XGB)│
│  (auth + proxy to 8001)     │       │  /predict/funnel-drop (XGBoost)  │
│  AI Insights dashboard      │       │  /predict/intent     (BERT+RF)   │
│  Journey/Funnel reports     │       │  /recommend          (GRU+ANN)   │
└─────────────────────────────┘       │  /predict/rage-click (Isolation)  │
                                      │  /train/*            (GPU loop)  │
                                      │  /health              (GPU info) │
                                      └───────────┬──────────────────────┘
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
                                          │ + vectors │
                                          │ Streaming │
                                          │ DiskANN   │
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
# Build and start with PG18 + TimescaleDB + AGE
docker compose up -d

# The db service builds a custom image with all extensions
# The umami service runs on port 3000
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

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | GPU status, model loaded state |
| `POST` | `/predict/next-page` | Predict next page(s) a visitor will view |
| `POST` | `/predict/funnel-drop` | Predict drop-off probability at funnel steps |
| `POST` | `/predict/intent` | Classify session intent (8 categories) |
| `POST` | `/recommend` | Session-based content recommendations |
| `POST` | `/predict/rage-click` | Detect user frustration from click patterns |
| `POST` | `/train/next-page` | Train next-page predictor from session data |
| `POST` | `/train/funnel` | Train funnel drop-off predictor |
| `POST` | `/train/intent` | Train session intent classifier |
| `POST` | `/train/recommender` | Train session-based recommender |
| `POST` | `/train/rage-click` | Train rage-click detector from heatmap data |
| `POST` | `/models/save` | Save all trained models to disk |
| `POST` | `/models/load` | Load all saved models from disk |

### ML Models

| Model | Algorithm | GPU | Training Data |
|---|---|---|---|
| **NextPagePredictor** | Markov chain + Transformer | PyTorch (RTX 3060) | Session page sequences |
| **FunnelPredictor** | XGBoost (hist tree) | CUDA (P40 via ONNX) | Session features + labels |
| **SessionIntentClassifier** | SentenceTransformer + RandomForest | PyTorch (RTX 3060) | Page URLs + session metadata |
| **Recommender** | GRU encoder + ANN (pgvectorscale) | PyTorch (RTX 3060) | Session page sequences |
| **RageClickDetector** | Isolation Forest + rule-based | CPU (sklearn) | Heatmap click coordinates |

### Training Pipeline

```bash
# Train all models for a specific website (last 30 days)
python3 ml/train_all.py --website <website_id> --days 30

# Or via API
curl -X POST http://localhost:8001/train/next-page \
  -H 'Content-Type: application/json' \
  -d '{"website_id":"<id>","start_date":"2026-06-01","end_date":"2026-07-01"}'
```

### Scheduled Jobs

| Job | Schedule | Function |
|---|---|---|
| `umami-age-sync` | Every 15 min | Sync TimescaleDB data to Apache AGE graph |
| `umami-ml-train` | Daily 3 AM | Retrain all ML models from latest data |

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

### pgvectorscale Index

```sql
CREATE INDEX idx_page_embeddings_ann 
    ON page_embeddings 
    USING diskann (embedding vector_cosine_ops)
    WITH (num_neighbors=50, search_list_size=200, max_alpha=1.2);
```

---

## 🖥 AI Insights Dashboard

The AI Insights page is available in the website navigation under **Behavior > AI Insights**.

It displays:
- **Recommended Pages** — content suggestions based on session context
- **Predicted Next Pages** — most likely pages the visitor will view next
- **Session Intent** — automatic intent classification
- **Funnel Drop-off Risk** — real-time probability of abandonment

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

---

## 🛟 Support

<p align="center">
  <a href="https://github.com/iskandarsulaili/umami"><img src="https://img.shields.io/badge/GitHub--blue?style=social&logo=github" alt="GitHub" /></a>
  <a href="https://umami.is/discord"><img src="https://img.shields.io/badge/Discord--blue?style=social&logo=discord" alt="Discord" /></a>
</p>
