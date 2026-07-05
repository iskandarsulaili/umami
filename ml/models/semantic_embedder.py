"""
Semantic Embedder for Umami ML Service
Generates text embeddings using configurable models for semantic page recommendations.

Supports:
- Models: 'minilm' (384d, all-MiniLM-L6-v2, local only) or 'qwen3' (2048d, local or cloud)
- Modes: 'local' (SentenceTransformer on GPU/CPU) or 'cloud' (OpenAI-compatible API)
- Each mode stores results in its own pgvector column: semantic_embedding (384d) or qwen3_embedding (2048d)
- Graceful fallback: if model not available, falls back to token mode

Configuration (env vars):
  EMBEDDING_MODEL: 'minilm' (default) or 'qwen3'
  EMBEDDING_MODE: 'local' (default) or 'cloud'
  EMBEDDING_API_URL: e.g. https://api.together.xyz/v1 or https://dashscope.aliyuncs.com/compatible-mode/v1
  EMBEDDING_API_KEY: API key for cloud provider
"""
import os
import json
import logging
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

# Cached local model instances
_minilm_encoder = None
_qwen3_encoder = None

# Embedding dimensions per model (configurable via env vars)
# Qwen3-Embedding-0.6B: 32-1024 (default 1024)
# Qwen3-Embedding-4B:   32-2560
# Qwen3-Embedding-8B:   32-4096
EMBEDDING_DIMS = {
    'minilm': int(os.getenv('EMBEDDING_DIM_MINILM', '384')),
    'qwen3': int(os.getenv('EMBEDDING_DIM_QWEN3', '1024')),
}

# Column names per model
EMBEDDING_COLUMNS = {
    'minilm': 'semantic_embedding',
    'qwen3': 'qwen3_embedding',
}

def get_config() -> tuple[str, str, str, str]:
    """Get embedding configuration from environment."""
    model = os.getenv('EMBEDDING_MODEL', 'minilm').lower()
    mode = os.getenv('EMBEDDING_MODE', 'local').lower()
    api_url = os.getenv('EMBEDDING_API_URL', '')
    api_key = os.getenv('EMBEDDING_API_KEY', '')
    return model, mode, api_url, api_key


def get_local_encoder(model: str, device: Optional[str] = None):
    """Get or create cached SentenceTransformer instance for the given model."""
    global _minilm_encoder, _qwen3_encoder

    if model == 'qwen3':
        if _qwen3_encoder is None:
            if not ST_AVAILABLE:
                raise ImportError("sentence-transformers not installed: pip install sentence-transformers")
            dev = device or str(gpu_utils.DEVICE)
            logger.info(f"Loading Qwen3-embedding on {dev} (this may take a moment for first download)")
            try:
                model_name = os.getenv("EMBEDDING_MODEL_QWEN3", "Qwen/Qwen3-Embedding-0.6B")
                logger.info(f"Loading {model_name} on {dev} (this may take a moment)")
                _qwen3_encoder = SentenceTransformer(model_name, device=dev)
            except Exception as e:
                logger.warning(f"Failed to load Qwen3-embedding locally: {e}. Falling back to MiniLM.")
                logger.info("Loading all-MiniLM-L6-v2 as fallback")
                _qwen3_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
        return _qwen3_encoder
    else:
        # Default: all-MiniLM-L6-v2
        if _minilm_encoder is None:
            if not ST_AVAILABLE:
                raise ImportError("sentence-transformers not installed: pip install sentence-transformers")
            dev = device or str(gpu_utils.DEVICE)
            logger.info(f"Loading all-MiniLM-L6-v2 on {dev}")
            _minilm_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
        return _minilm_encoder


def embed_text_local(texts: list[str], model: str = 'minilm', device: Optional[str] = None) -> np.ndarray:
    """Embed text using local SentenceTransformer model.

    Args:
        texts: List of text strings
        model: 'minilm' (384d) or 'qwen3' (2048d)
        device: Torch device string

    Returns:
        numpy array of shape (len(texts), dim) with normalized embeddings
    """
    encoder = get_local_encoder(model, device)
    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def embed_text_cloud(texts: list[str], model: str = 'qwen3') -> np.ndarray:
    """Embed text using OpenAI-compatible cloud API (e.g. Together AI, Alibaba Cloud).

    Args:
        texts: List of text strings
        model: API model name (default 'Qwen/Qwen3-embedding')

    Returns:
        numpy array of shape (len(texts), dim) with normalized embeddings
    """
    _, _, api_url, api_key = get_config()

    if not api_url or not api_key:
        logger.warning("Cloud embedding not configured (set EMBEDDING_API_URL and EMBEDDING_API_KEY)")
        # Fallback to local
        return embed_text_local(texts, 'minilm')

    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not available, falling back to local")
        return embed_text_local(texts, model if model == 'qwen3' else 'minilm')

    api_model = "Qwen/Qwen3-embedding" if model == 'qwen3' else "all-MiniLM-L6-v2"
    url = api_url.rstrip('/') + '/embeddings'

    try:
        response = requests.post(
            url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': api_model,
                'input': texts,
                'encoding_format': 'float',
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        # Extract embeddings, ordered by index
        embeddings = [d['embedding'] for d in sorted(data['data'], key=lambda x: x['index'])]
        arr = np.array(embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-8)
        return arr
    except Exception as e:
        logger.warning(f"Cloud embedding failed: {e}, falling back to local")
        return embed_text_local(texts, 'minilm')


def embed_text(texts: list[str], model: Optional[str] = None, mode: Optional[str] = None, device: Optional[str] = None) -> np.ndarray:
    """Embed text using the configured model and mode.

    Args:
        texts: List of text strings
        model: Override config model ('minilm' or 'qwen3')
        mode: Override config mode ('local' or 'cloud')
        device: Torch device string

    Returns:
        numpy array of shape (len(texts), dim) with normalized embeddings
    """
    cfg_model, cfg_mode, _, _ = get_config()
    model = model or cfg_model
    mode = mode or cfg_mode

    if mode == 'cloud' and (model == 'qwen3' or _has_cloud_config()):
        try:
            return embed_text_cloud(texts, model)
        except Exception as e:
            logger.warning(f"Cloud embedding failed: {e}, falling back to local")
            return embed_text_local(texts, model, device)
    else:
        return embed_text_local(texts, model, device)


def _has_cloud_config() -> bool:
    """Check if cloud embedding is configured."""
    _, _, api_url, api_key = get_config()
    return bool(api_url and api_key)


def embed_page_url(url: str) -> str:
    """Convert a URL path into a descriptive text string for embedding.

    E.g. '/products/running-shoes' -> 'products running shoes'
    """
    text = unquote(url.strip("/"))
    text = text.replace("/", " ").replace("-", " ").replace("_", " ")
    if "?" in text:
        text = text.split("?")[0]
    return text.strip() or "/"


def get_embedding_dim(model: Optional[str] = None) -> int:
    """Get the embedding dimension for the given model."""
    cfg_model, _, _, _ = get_config()
    return EMBEDDING_DIMS.get(model or cfg_model, 384)


def get_embedding_column(model: Optional[str] = None) -> str:
    """Get the pgvector column name for the given model."""
    cfg_model, _, _, _ = get_config()
    return EMBEDDING_COLUMNS.get(model or cfg_model, 'semantic_embedding')


def sync_semantic_embeddings(
    website_id: str,
    page_urls: list[str],
    db_url: Optional[str] = None,
    model: Optional[str] = None,
    mode: Optional[str] = None,
) -> int:
    """Generate and store semantic embeddings for a list of pages.

    Supports both 'minilm' and 'qwen3' models, both 'local' and 'cloud' modes.
    Stores in the corresponding pgvector column and rebuilds the DiskANN index.

    Args:
        website_id: Umami website UUID
        page_urls: List of page URL paths to embed
        db_url: Database connection string
        model: 'minilm' or 'qwen3' (default from config)
        mode: 'local' or 'cloud' (default from config)

    Returns:
        Number of embeddings synced
    """
    if not page_urls:
        return 0

    cfg_model, cfg_mode, _, _ = get_config()
    model = model or cfg_model
    mode = mode or cfg_mode

    texts = [embed_page_url(p) for p in page_urls]
    embeddings = embed_text(texts, model=model, mode=mode)
    column = get_embedding_column(model)

    if not HAS_PSYCOPG2:
        logger.warning("psycopg2 not available, cannot store embeddings")
        return 0

    db_url = db_url or CONFIG.db.url
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        count = 0
        for page_url, emb in zip(page_urls, embeddings):
            emb_str = "[" + ",".join(f"{v:.6f}" for v in emb) + "]"
            cur.execute(f"""
                INSERT INTO page_embeddings (website_id, page_url, {column}, updated_at)
                VALUES (%s, %s, %s::vector, NOW())
                ON CONFLICT (website_id, page_url)
                DO UPDATE SET {column} = %s::vector, updated_at = NOW()
            """, (website_id, page_url, emb_str, emb_str))
            count += 1
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Synced {count} {model} embeddings ({mode}) to pgvector column '{column}'")
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
) -> list[tuple[str, float]]:
    """Retrieve candidate pages via semantic similarity search.

    Uses the configured model's pgvector column and DiskANN index.

    Args:
        query_text: Text to search by
        website_id: Umami website UUID
        top_k: Number of candidates to return
        exclude: Set of page URLs to exclude
        db_url: Database connection string
        model: 'minilm' or 'qwen3' (default from config)
        mode: 'local' or 'cloud' (default from config)

    Returns:
        List of (page_url, similarity_score) tuples
    """
    exclude = exclude or set()
    column = get_embedding_column(model)

    cfg_model, cfg_mode, _, _ = get_config()
    model_resolved = model or cfg_model
    mode_resolved = mode or cfg_mode

    query_emb = embed_text([query_text], model=model_resolved, mode=mode_resolved)[0]
    emb_str = "[" + ",".join(f"{v:.6f}" for v in query_emb) + "]"

    if not HAS_PSYCOPG2 or not db_url:
        return []

    db_url = db_url or CONFIG.db.url
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        exclude_clause = ""
        if exclude:
            esc = ", ".join(f"'{e.replace(chr(39), chr(39)+chr(39))}'" for e in exclude)
            exclude_clause = f"AND page_url NOT IN ({esc})"

        query = f"""
            SELECT page_url, 1 - ({column} <=> %s::vector) AS similarity
            FROM page_embeddings
            WHERE website_id = %s
              AND {column} IS NOT NULL
              {exclude_clause}
            ORDER BY {column} <=> %s::vector
            LIMIT %s
        """
        cur.execute(query, (emb_str, website_id, emb_str, top_k))
        results = [(row[0], float(row[1])) for row in cur.fetchall()]
        cur.close()
        conn.close()
        return results
    except Exception as e:
        logger.warning(f"Semantic search ({model_resolved}, {column}) failed: {e}")
        return []
