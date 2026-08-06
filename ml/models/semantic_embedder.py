"""
Semantic Embedder for Umami ML Service
Generates text embeddings using any SentenceTransformer model locally or any OpenAI-compatible API.

Supports:
- Local mode: any SentenceTransformer model from HuggingFace
- OpenAI mode: any OpenAI embedding model (text-embedding-3-small, text-embedding-3-large, etc.)
- Gemini mode: any Google Generative AI embedding model
- Cloud mode: any OpenAI-compatible API (user-provided URL + key)

Configuration (env vars):
  EMBEDDING_MODEL: any HF model name (default: intfloat/multilingual-e5-small)
  EMBEDDING_MODE: 'local', 'openai', 'gemini', or 'cloud'
  EMBEDDING_API_URL: e.g. https://api.openai.com/v1 or user URL
  EMBEDDING_API_KEY: API key for cloud provider
  EMBEDDING_DIM: output dimension (default varies by model)
  EMBEDDING_MODEL_QWEN3: backward compat, if 'qwen3' still used
"""
import os
import json
import logging
import threading
import numpy as np
from typing import Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from ..config import CONFIG
from .. import gpu_utils

# Cache for local encoder instances
_encoder_cache: dict[str, SentenceTransformer] = {}

# Guard for lazy encoder loading — SentenceTransformer init is expensive and NOT
# thread-safe for concurrent double-load (TOCTOU race: two cold-cache requests
# would both load the model). (audit: umami encoder cache race fix)
_encoder_cache_lock = threading.Lock()

# Default dimension if not configured
DEFAULT_DIM = int(os.getenv('EMBEDDING_DIM', '384'))

# Provider-specific default models
PROVIDER_DEFAULT_MODELS = {
    'local': 'intfloat/multilingual-e5-small',
    'openai': 'text-embedding-3-small',
    'gemini': 'models/embedding-001',
    'cloud': '',  # Must be provided by user
}


def get_config() -> tuple[str, str, str, str]:
    """Get embedding configuration from environment."""
    model_raw = os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')

    # Backward compat: if model is 'minilm' or 'qwen3', resolve
    model = model_raw.lower()
    if model == 'minilm':
        model = 'intfloat/multilingual-e5-small'
    elif model == 'qwen3':
        model = os.getenv('EMBEDDING_MODEL_QWEN3', 'Qwen/Qwen3-Embedding-0.6B')

    mode = os.getenv('EMBEDDING_MODE', 'local').lower()
    api_url = os.getenv('EMBEDDING_API_URL', '')
    api_key = os.getenv('EMBEDDING_API_KEY', '')

    # Backward compat: if no explicit mode but has API URL/Key, infer 'cloud'
    if mode == 'local' and api_url and api_key:
        mode = 'cloud'

    return model, mode, api_url, api_key


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize embedding array."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)


# ---------------------------------------------------------------------------
# Provider dispatchers
# ---------------------------------------------------------------------------

def _embed_local(texts: list[str], model: str, device: Optional[str] = None) -> np.ndarray:
    """Embed via SentenceTransformer (any HF model supported)."""
    global _encoder_cache

    if not ST_AVAILABLE:
        raise ImportError("sentence-transformers not installed: pip install sentence-transformers")

    if model not in _encoder_cache:
        with _encoder_cache_lock:
            # Double-checked locking: another thread may have loaded it while we waited.
            if model not in _encoder_cache:
                dev = device or str(gpu_utils.DEVICE)
                logger.info(f"Loading model '{model}' on {dev} (this may take a moment)")
                try:
                    _encoder_cache[model] = SentenceTransformer(model, device=dev)
                except Exception as e:
                    logger.warning(f"Failed to load '{model}': {e}, falling back to intfloat/multilingual-e5-small")
                    _encoder_cache[model] = SentenceTransformer("intfloat/multilingual-e5-small", device=dev)
                    model = "intfloat/multilingual-e5-small"

    encoder = _encoder_cache[model]
    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def _embed_openai(texts: list[str], model: str, api_key: str, api_url: Optional[str] = None) -> np.ndarray:
    """Embed via OpenAI-compatible API."""
    url = (api_url or 'https://api.openai.com/v1').rstrip('/') + '/embeddings'
    api_key = api_key or os.getenv('OPENAI_API_KEY', '')

    if not api_key:
        raise ValueError("OpenAI API key required (set EMBEDDING_API_KEY or OPENAI_API_KEY)")

    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library required for API embedding")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        'model': model,
        'input': texts,
        'encoding_format': 'float',
    }

    # Add dimension if configured (supported by OpenAI and compatible APIs)
    dim = os.getenv('EMBEDDING_DIM', '')
    if dim:
        payload['dimensions'] = int(dim)

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    embeddings = [d['embedding'] for d in sorted(data['data'], key=lambda x: x['index'])]
    arr = np.array(embeddings, dtype=np.float32)
    return _normalize(arr)


def _embed_gemini(texts: list[str], model: str, api_key: str) -> np.ndarray:
    """Embed via Google Gemini/Generative AI API."""
    api_key = api_key or os.getenv('GEMINI_API_KEY', '')

    if not api_key:
        raise ValueError("Gemini API key required (set EMBEDDING_API_KEY or GEMINI_API_KEY)")

    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library required for API embedding")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"

    results = []
    # Gemini API processes one text at a time
    for text in texts:
        payload = {
            'model': f'models/{model}' if not model.startswith('models/') else model,
            'content': {'parts': [{'text': text}]},
        }
        resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results.append(data['embedding']['values'])

    arr = np.array(results, dtype=np.float32)
    return _normalize(arr)


def _embed_cloud(texts: list[str], model: str, api_url: str, api_key: str) -> np.ndarray:
    """Embed via any OpenAI-compatible API (generic cloud)."""
    if not api_url or not api_key:
        raise ValueError("Cloud embedding requires EMBEDDING_API_URL and EMBEDDING_API_KEY")

    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library required for API embedding")

    url = api_url.rstrip('/') + '/embeddings'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        'model': model,
        'input': texts,
        'encoding_format': 'float',
    }

    dim = os.getenv('EMBEDDING_DIM', '')
    if dim:
        payload['dimensions'] = int(dim)

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    embeddings = [d['embedding'] for d in sorted(data['data'], key=lambda x: x['index'])]
    arr = np.array(embeddings, dtype=np.float32)
    return _normalize(arr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(texts: list[str], model: Optional[str] = None, mode: Optional[str] = None,
               device: Optional[str] = None, api_url: Optional[str] = None,
               api_key: Optional[str] = None) -> np.ndarray:
    """Embed text using configured model, mode, and optional overrides.

    Provider resolution:
      mode='local'   → SentenceTransformer with model name (any HF model)
      mode='openai'  → OpenAI API with model name
      mode='gemini'  → Google Generative AI API
      mode='cloud'   → User-provided API URL (OpenAI-compatible)

    Args:
        texts: List of text strings
        model: Model name (default from env EMBEDDING_MODEL)
        mode: Provider mode ('local', 'openai', 'gemini', 'cloud')
        device: Torch device string (local mode only)
        api_url: Override API URL (cloud modes only)
        api_key: Override API key (cloud modes only)

    Returns:
        numpy array of shape (len(texts), dim) with normalized embeddings
    """
    cfg_model, cfg_mode, cfg_url, cfg_key = get_config()
    model = model or cfg_model
    mode = (mode or cfg_mode).lower()
    api_url = api_url or cfg_url
    api_key = api_key or cfg_key

    try:
        if mode == 'local':
            return _embed_local(texts, model, device)
        elif mode == 'openai':
            return _embed_openai(texts, model, api_key, api_url)
        elif mode == 'gemini':
            return _embed_gemini(texts, model, api_key)
        elif mode == 'cloud':
            return _embed_cloud(texts, model, api_url, api_key)
        else:
            logger.warning(f"Unknown mode '{mode}', falling back to local")
            return _embed_local(texts, model, device)
    except Exception as e:
        logger.warning(f"{mode}/{model} embedding failed: {e}, falling back to local")
        try:
            return _embed_local(texts, 'intfloat/multilingual-e5-small', device)
        except Exception as e2:
            logger.error(f"Fallback embedding failed: {e2}")
            raise


def embed_page_url(url: str) -> str:
    """Convert a URL path into a descriptive text string for embedding.

    E.g. '/products/running-shoes' -> 'products running shoes'
    """
    text = unquote(url.strip("/"))
    text = text.replace("/", " ").replace("-", " ").replace("_", " ")
    if "?" in text:
        text = text.split("?")[0]
    return text.strip() or "/"


def get_embedding_dim(model_override: Optional[str] = None) -> int:
    """Get configured embedding dimension."""
    dim = os.getenv('EMBEDDING_DIM', str(DEFAULT_DIM))
    return int(dim)


def get_modelled_column(model: str) -> str:
    """Return a deterministic column name for any model."""
    # Sanitize model name to a valid Postgres column identifier
    safe = model.replace('/', '_').replace('-', '_').replace('.', '_').lower()
    # Keep known short alias for common models
    alias_map = {
        'all_minilm_l6_v2': 'semantic_embedding',
        'all-minilm-l6-v2': 'semantic_embedding',
        'intfloat_multilingual_e5_small': 'semantic_embedding',
        'intfloat/multilingual-e5-small': 'semantic_embedding',
        'text_embedding_3_small': 'embedding',
        'text_embedding_3_large': 'embedding',
    }
    return alias_map.get(safe, f"emb_{safe[:48]}")


def sync_semantic_embeddings(
    website_id: str,
    page_urls: list[str],
    db_url: Optional[str] = None,
    model: Optional[str] = None,
    mode: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> int:
    """Generate and store semantic embeddings for a list of pages.

    Supports any embedding model via any provider. Stores all embeddings
    in the unified `embedding` column (vector(4096)) with model_name tracking.

    Args:
        website_id: Umami website UUID
        page_urls: List of page URL paths to embed
        db_url: Database connection string
        model: Model name (any HuggingFace model, OpenAI model, etc.)
        mode: 'local', 'openai', 'gemini', or 'cloud'
        api_url: API URL override (for cloud/openai/gemini modes)
        api_key: API key override

    Returns:
        Number of embeddings synced
    """
    if not page_urls:
        return 0

    cfg_model, cfg_mode, _, _ = get_config()
    model = model or cfg_model
    mode = mode or cfg_mode

    texts = [embed_page_url(p) for p in page_urls]
    embeddings = embed_text(texts, model=model, mode=mode, api_url=api_url, api_key=api_key)

    if not HAS_PSYCOPG2:
        logger.warning("psycopg2 not available, cannot store embeddings")
        return 0

    db_url = db_url or CONFIG.db.url
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        count = 0
        # Pick the storage column based on the model (384-dim models -> semantic_embedding,
        # 4096-dim / cloud models -> embedding). Keeps e5/MiniLM compatible with existing rows.
        col = get_modelled_column(model)
        for page_url, emb in zip(page_urls, embeddings):
            emb_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
            cur.execute(f"""
                INSERT INTO page_embeddings (website_id, page_url, {col}, model_name, updated_at)
                VALUES (%s, %s, %s::vector, %s, NOW())
                ON CONFLICT (website_id, page_url)
                DO UPDATE SET {col} = %s::vector, model_name = %s, updated_at = NOW()
            """, (website_id, page_url, emb_str, model, emb_str, model))
            count += 1
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Synced {count} embeddings for model '{model}' ({mode})")
        return count
    except Exception as e:
        logger.warning(f"Semantic embedding sync failed: {e}")
        return 0


def retrieve_semantic_candidates(
    query_text: str,
    website_id: str,
    top_k: int = 20,
    exclude: Optional[set] = None,
    db_url: Optional[str] = None,
    model: Optional[str] = None,
    mode: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[tuple[str, float]]:
    """Retrieve candidate pages via semantic similarity search.

    Uses the unified `embedding` column with DiskANN index.
    Filters by model_name to match only embeddings from the current model.

    Args:
        query_text: Text to search by
        website_id: Umami website UUID
        top_k: Number of candidates
        exclude: Set of page URLs to exclude
        db_url: Database connection string
        model: Model name (default from config)
        mode: Provider mode
        api_url: API URL override
        api_key: API key override

    Returns:
        List of (page_url, similarity_score) tuples
    """
    cfg_model, cfg_mode, _, _ = get_config()
    model_resolved = model or cfg_model
    mode_resolved = mode or cfg_mode

    embed = embed_text([query_text], model=model_resolved, mode=mode_resolved,
                       api_url=api_url, api_key=api_key)[0]
    emb_str = "[" + ",".join(f"{v:.6f}" for v in embed) + "]"

    if not HAS_PSYCOPG2:
        return []

    db_url = db_url or CONFIG.db.url
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        exclude_clause = ""
        col = get_modelled_column(model_resolved)
        params = [emb_str, website_id, model_resolved, emb_str, top_k]
        if exclude:
            placeholders = ", ".join(f"%s" for _ in exclude)
            exclude_clause = f"AND page_url NOT IN ({placeholders})"
            params = [emb_str, website_id, model_resolved, emb_str] + list(exclude) + [top_k]

        query = f"""
            SELECT page_url, 1 - ({col} <=> %s::vector) AS similarity
            FROM page_embeddings
            WHERE website_id = %s
              AND model_name = %s
              AND {col} IS NOT NULL
              {exclude_clause}
            ORDER BY {col} <=> %s::vector
            LIMIT %s
        """
        cur.execute(query, params)
        results = [(row[0], float(row[1])) for row in cur.fetchall()]
        cur.close()
        conn.close()
        return results
    except Exception as e:
        logger.warning(f"Semantic search ({model_resolved}) failed: {e}")
        return []
