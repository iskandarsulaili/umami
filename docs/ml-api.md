# Umami ML API v1 — Third-Party Integration Guide

## Overview

The ML API v1 allows third-party services to access trained ML predictions for a website without requiring a Umami admin login session. Authentication is via API key (Bring Your Own Key — BYOK).

**Base URL:** `https://your-umami-instance.com/api/ml/v1`

---

## Authentication

All v1 endpoints require an `x-api-key` header with a valid API key.

### Generating an API Key

**From the dashboard:** Settings > ML API Keys > Generate New Key

**Via API (admin only):**

```bash
curl -X POST https://your-umami.com/api/admin/ml/keys \
  -H "Content-Type: application/json" \
  -H "authorization: Bearer YOUR_ADMIN_TOKEN" \
  -d '{"websiteId":"YOUR_WEBSITE_ID","name":"My App"}'
```

Response:
```json
{
  "key": "umami_ml_3d205a8203fdc7a6f0f8da94d769e8f2f02c3881cf1ae95753968357e7344a11",
  "prefix": "3d205a82",
  "name": "My App"
}
```

> ⚠️ The full key is only shown once. Save it securely.

### Listing API Keys

```bash
curl -X GET https://your-umami.com/api/admin/ml/keys?websiteId=YOUR_WEBSITE_ID \
  -H "authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Revoking an API Key

```bash
curl -X DELETE https://your-umami.com/api/admin/ml/keys \
  -H "Content-Type: application/json" \
  -H "authorization: Bearer YOUR_ADMIN_TOKEN" \
  -d '{"websiteId":"YOUR_WEBSITE_ID","keyId":"KEY_UUID"}'
```

---

## Endpoints

### Health Check

```http
GET /api/ml/v1/health
```

No authentication required.

### Get Recommendations

```http
POST /api/ml/v1/recommend
Content-Type: application/json
x-api-key: umami_ml_YOUR_KEY

{
  "sessionPages": ["/", "/pricing", "/signup"],
  "sessionFeatures": {},
  "topK": 10,
  "mode": "hybrid",
  "semanticWeight": 0.6,
  "embeddingModel": "minilm",
  "embeddingMode": "local",
  "embeddingApiUrl": null,
  "embeddingApiKey": null
}
```

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `sessionPages` | array | `[]` | Pages visited in the current session (URL paths) |
| `sessionFeatures` | object | `{}` | Optional session metadata (browser, device, country, etc.) |
| `topK` | int | `20` | Number of recommendations to return |
| `mode` | string | `"token"` | `"token"` — GRU URL matching, `"semantic"` — embedding-based, `"hybrid"` — fused |
| `semanticWeight` | float | `0.6` | Weight for semantic in hybrid mode (0.0 = pure token, 1.0 = pure semantic) |
| `embeddingModel` | string | `"minilm"` | `"minilm"` (384d) or `"qwen3"` (4096d) |
| `embeddingMode` | string | `"local"` | `"local"` (GPU/CPU) or `"cloud"` (API) |
| `embeddingApiUrl` | string | null | Required for cloud mode. Your API endpoint URL |
| `embeddingApiKey` | string | null | Required for cloud mode. Your API key |

**Field names are case-insensitive** — snake_case (`session_pages`) and camelCase (`sessionPages`) both work.

**Response:**

```json
{
  "website_id": "...",
  "session_pages": ["/", "/pricing", "/signup"],
  "recommendations": [
    {"page": "/enterprise", "score": 0.8234, "base_similarity": 0.7801},
    {"page": "/docs", "score": 0.6543, "base_similarity": 0.6123}
  ],
  "age_recommendations": [
    {"target": "/enterprise", "weight": 0.75}
  ],
  "count": 2,
  "is_cold_start": false
}
```

> Recommendations will be empty `[]` if models aren't trained yet. Click "Train All Models" on the dashboard first.

---

### Next Page Prediction

```http
POST /api/ml/v1/predict/next-page
Content-Type: application/json
x-api-key: umami_ml_YOUR_KEY

{
  "sessionPages": ["/", "/pricing"],
  "topK": 5
}
```

**Response:**

```json
{
  "website_id": "...",
  "session_pages": ["/", "/pricing"],
  "predictions": [
    {"page": "/signup", "probability": 0.42},
    {"page": "/docs", "probability": 0.28}
  ],
  "model_type": "markov",
  "device": "cuda:0"
}
```

---

### Session Intent Classification

```http
POST /api/ml/v1/predict/intent
Content-Type: application/json
x-api-key: umami_ml_YOUR_KEY

{
  "sessionPages": ["/", "/pricing", "/enterprise"],
  "sessionFeatures": {"browser": "Chrome", "country": "US"}
}
```

**Response:**

```json
{
  "intent": "price_comparing",
  "confidence": 0.87,
  "probabilities": {
    "researching": 0.05,
    "price_comparing": 0.87,
    "ready_to_buy": 0.08
  },
  "is_cold_start": false
}
```

**Intent categories:** `researching`, `ready_to_buy`, `price_comparing`, `support_seeking`, `just_browsing`, `churning`, `account_management`, `content_consumption`

---

### Funnel Drop-off Risk

```http
POST /api/ml/v1/predict/funnel-drop
Content-Type: application/json
x-api-key: umami_ml_YOUR_KEY

{
  "session": {
    "browser": "Chrome",
    "device": "desktop",
    "country": "US",
    "avg_lcp": 1234,
    "avg_cls": 0.05,
    "page_views": 3
  },
  "threshold": 0.5
}
```

**Response:**

```json
{
  "will_continue": true,
  "probability": 0.72,
  "confidence": 0.44,
  "threshold": 0.5
}
```

---

### Rage Click / Frustration Detection

```http
POST /api/ml/v1/predict/rage-click
Content-Type: application/json
x-api-key: umami_ml_YOUR_KEY

{
  "clicks": [
    {"x": 100, "y": 200, "created_at": "2026-07-01T12:00:00Z"},
    {"x": 102, "y": 198, "created_at": "2026-07-01T12:00:01Z"}
  ]
}
```

Or provide a `sessionId` to fetch click data from the database:

```json
{
  "sessionId": "SESSION_UUID"
}
```

**Response:**

```json
{
  "frustration_type": "rage_click",
  "confidence": 0.85,
  "severity": "high",
  "ml_anomaly_score": 0.72,
  "features": {"n_clicks": 5, "rage_clusters": 3}
}
```

**Types:** `rage_click`, `dead_click`, `mouse_shake`, `form_abandonment`, `scroll_rage`, `error_loop`, `none`

---

### Visitor Archetype (Journey Clustering)

```http
POST /api/ml/v1/predict/cluster
Content-Type: application/json
x-api-key: umami_ml_YOUR_KEY

{
  "session": {
    "pages": ["/", "/blog", "/pricing"],
    "duration_seconds": 180,
    "device": "desktop",
    "hour": 14,
    "is_weekend": false
  }
}
```

**Response:**

```json
{
  "archetype": "comparison_shopper",
  "description": "Product pages, back-and-forth navigation, price checking",
  "probability": 0.91,
  "cluster_id": 1
}
```

**Archetypes:** `power_user`, `comparison_shopper`, `single_page_scanner`, `lost_user`, `conversion_ready`, `content_consumer`, `returning_visitor`, `deal_seeker`, `support_seeker`, `browser`, `unclassified`

---

### Session Replay Analysis

```http
POST /api/ml/v1/predict/session-replay
Content-Type: application/json
x-api-key: umami_ml_YOUR_KEY

{
  "sessionId": "SESSION_UUID"
}
```

**Response:**

```json
{
  "error_loops": 3,
  "dead_clicks": 2,
  "form_struggles": 0,
  "rapid_navigation": 1,
  "severity": "medium",
  "total_events": 245
}
```

---

## Error Handling

All errors return a consistent structure:

```json
{
  "error": {
    "message": "Description of what went wrong",
    "code": "error-code",
    "status": 400
  }
}
```

| HTTP Status | Code | Meaning |
|---|---|---|
| 401 | `unauthorized` | Missing or invalid `x-api-key` header |
| 422 | `validation-error` | Invalid request parameters |
| 500 | `server-error` | ML service unavailable or error |
| 500 | `server-error` | Embedding model failed (falls back internally) |

---

## Rate Limiting

Currently no rate limiting is enforced. This may change in future versions.

---

## Embedding Configuration

### Local Mode (default)

- **MiniLM**: Works out of the box (80MB model, auto-downloaded on first use)
- **Qwen3**: Requires HuggingFace token (`HF_TOKEN` env var) to download the gated model (~2GB)

### Cloud Mode

Requires `embeddingApiUrl` and `embeddingApiKey` parameters (or the equivalent env vars):

| Provider | API URL |
|---|---|
| **Together AI** | `https://api.together.xyz/v1` |
| **Alibaba Cloud DashScope** | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| **Any OpenAI-compatible** | Your API endpoint |

If cloud embedding fails, the system **automatically falls back** to local MiniLM — no errors, no crashes.

---

## Example: cURL

```bash
# Get recommendations (token mode)
curl -X POST https://your-umami.com/api/ml/v1/recommend \
  -H "Content-Type: application/json" \
  -H "x-api-key: umami_ml_YOUR_KEY" \
  -d '{"sessionPages":["/","/pricing"],"topK":5,"mode":"token"}'

# Get recommendations (hybrid mode, Qwen3 cloud)
curl -X POST https://your-umami.com/api/ml/v1/recommend \
  -H "Content-Type: application/json" \
  -H "x-api-key: umami_ml_YOUR_KEY" \
  -d '{
    "sessionPages":["/","/pricing"],
    "topK":10,
    "mode":"hybrid",
    "semanticWeight":0.7,
    "embeddingModel":"qwen3",
    "embeddingMode":"cloud",
    "embeddingApiUrl":"https://api.together.xyz/v1",
    "embeddingApiKey":"YOUR_TOGETHER_KEY"
  }'

# Predict next page
curl -X POST https://your-umami.com/api/ml/v1/predict/next-page \
  -H "Content-Type: application/json" \
  -H "x-api-key: umami_ml_YOUR_KEY" \
  -d '{"sessionPages":["/","/blog"],"topK":3}'
```

## Example: Python

```python
import requests

API_KEY = "umami_ml_YOUR_KEY"
BASE_URL = "https://your-umami.com/api/ml/v1"

# Get recommendations
resp = requests.post(f"{BASE_URL}/recommend", json={
    "sessionPages": ["/", "/pricing"],
    "topK": 5,
    "mode": "hybrid",
}, headers={"x-api-key": API_KEY})

print(resp.json())
```

## Example: JavaScript

```javascript
const API_KEY = "umami_ml_YOUR_KEY";

// Get recommendations
const resp = await fetch("https://your-umami.com/api/ml/v1/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json", "x-api-key": API_KEY },
  body: JSON.stringify({
    sessionPages: ["/", "/pricing"],
    topK: 5,
    mode: "hybrid",
  }),
});
const data = await resp.json();
console.log(data.recommendations);
```
