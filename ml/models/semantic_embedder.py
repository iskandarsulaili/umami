"""
Semantic Embedder for Umami ML Service
Generates text embeddings using SentenceTransformer for semantic page recommendations.

Allows toggling between:
- 'token' mode: GRU-based session encoder (current default, URL tokens only)
- 'semantic' mode: SentenceTransformer embeddings (page content meaning)

GPU: Auto-detects CUDA, falls back to CPU. Model is cached in memory after first load.
"""
import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

# Cached model instance
_text_encoder = None
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def get_text_encoder(device: Optional[str] = None):
    """Get or create cached SentenceTransformer instance"""
    global _text_encoder
    if _text_encoder is None:
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers not installed: pip install sentence-transformers")
        dev = device or str(gpu_utils.DEVICE)
        logger.info(f"Loading semantic embedder (all-MiniLM-L6-v2) on {dev}")
        _text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
    return _text_encoder


def embed_text(texts: list[str], device: Optional[str] = None) -> np.ndarray:
    """Embed a list of text strings into 384-dim vectors.

    Args:
        texts: List of text strings (page titles, descriptions, or URL + content)
        device: Torch device string (cuda:0 or cpu)

    Returns:
        numpy array of shape (len(texts), 384) with normalized embeddings
    """
    encoder = get_text_encoder(device)
    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings


def embed_page_url(url: str) -> str:
    """Convert a URL path into a descriptive text string for embedding.

    Extracts meaningful words from URL paths to create a searchable text.
    E.g. '/products/running-shoes' -> 'products running shoes'
    """
    from urllib.parse import unquote
    # Strip leading slash and decode percent-encoding
    text = unquote(url.strip("/"))
    # Replace separators with spaces
    text = text.replace("/", " ").replace("-", " ").replace("_", " ")
    # Remove query parameters
    if "?" in text:
        text = text.split("?")[0]
    return text.strip() or "/"


def sync_semantic_embeddings(
    website_id: str,
    page_urls: list[str],
    db_url: Optional[str] = None,
) -> int:
    """Generate and store semantic embeddings for a list of pages.

    Each page URL is converted to a descriptive text string, embedded via
    SentenceTransformer, and stored in the page_embeddings table's
    semantic_embedding column.

    Args:
        website_id: Umami website UUID
        page_urls: List of page URL paths to embed
        db_url: Database connection string (default: from config)

    Returns:
        Number of embeddings synced
    """
    if not page_urls:
        return 0

    # Generate texts from URLs
    texts = [embed_page_url(p) for p in page_urls]
    embeddings = embed_text(texts)

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
            cur.execute("""
                INSERT INTO page_embeddings (website_id, page_url, semantic_embedding, updated_at)
                VALUES (%s, %s, %s::vector, NOW())
                ON CONFLICT (website_id, page_url)
                DO UPDATE SET semantic_embedding = %s::vector, updated_at = NOW()
            """, (website_id, page_url, emb_str, emb_str))
            count += 1
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Synced {count} semantic embeddings to pgvector")
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
) -> list[tuple[str, float]]:
    """Retrieve candidate pages via semantic similarity search.

    Embeds the query text and uses pgvector DiskANN index for fast
    approximate nearest neighbor search on the semantic_embedding column.

    Args:
        query_text: Text to search by (e.g. combined page context)
        website_id: Umami website UUID
        top_k: Number of candidates to return
        exclude: Set of page URLs to exclude
        db_url: Database connection string

    Returns:
        List of (page_url, similarity_score) tuples
    """
    exclude = exclude or set()
    query_emb = embed_text([query_text])[0]
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
            SELECT page_url, 1 - (semantic_embedding <=> %s::vector) AS similarity
            FROM page_embeddings
            WHERE website_id = %s
              AND semantic_embedding IS NOT NULL
              {exclude_clause}
            ORDER BY semantic_embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(query, (emb_str, website_id, emb_str, top_k))
        results = [(row[0], float(row[1])) for row in cur.fetchall()]
        cur.close()
        conn.close()
        return results
    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return []
