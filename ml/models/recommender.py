"""
Session-Based Recommender System - Model 4
Produces ranked content recommendations for active sessions.
Uses a three-stage architecture:
  1. Session encoder (GRU/Transformer) on GPU
  2. Approximate nearest neighbor retrieval (pgvector in TimescaleDB)
  3. Cross-attention ranker (lightweight Transformer)

Reference: arxiv:1902.04864 (Survey on Session-based Recommender Systems),
           arxiv:2402.17129 (Side Information-Driven SBR),
           arxiv:1811.00855 (SR-GNN - Session-based Rec with GNN)
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime, timedelta
import numpy as np

from .base import BaseModel
from ..config import CONFIG, DatabaseConfig
from .. import gpu_utils

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional semantic embedder
try:
    from . import semantic_embedder
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


class SessionEncoder(nn.Module):
    """
    GRU-based session encoder that produces fixed-dimension
    session embeddings from variable-length page sequences.
    GPU accelerated, exported to ONNX for serving.
    Includes a decoder head for next-item prediction training.
    
    Reference: arxiv:1902.04864 (SBRS survey)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Decoder head for next-item prediction during training
        self.decoder = nn.Linear(hidden_dim, vocab_size + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded)
        # Concatenate forward and backward last hidden states
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output_proj(last_hidden)
    
    def forward_with_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both embedding and next-item logits."""
        emb = self.forward(x)
        logits = self.decoder(emb)
        return emb, logits


class Recommender(BaseModel):
    """
    Model 4: Session-Based Recommender System
    
    Three-stage architecture:
    1. Encode session to embedding vector (SessionEncoder, GPU)
    2. Retrieve candidate items via ANN (pgvector in TimescaleDB)
    3. Rank candidates with cross-attention
    
    Implementation:
    - Stage 1: Trained SessionEncoder produces session embeddings
    - Stage 2: pgvector stores all page embeddings; ANN via IVFFlat index
    - Stage 3: Lightweight scorer combining session + candidate features
    
    GPU: SessionEncoder on GPU, ANN retrieval on CPU (pgvector handles this)
    CPU: Full pipeline works on CPU (slower but functional)
    """
    
    def __init__(self):
        super().__init__(name="recommender", version="1.0.0")
        self.session_encoder: Optional[SessionEncoder] = None
        self.page_embeddings: dict[str, np.ndarray] = {}
        self.page_to_idx: dict[str, int] = {}
        self.idx_to_page: dict[int, str] = {}
        self.vocab_size = 0
        self.ann_index: Optional[NearestNeighbors] = None
        self.ann_page_order: list[str] = []
    
    def _build_vocab(self, all_pages: set):
        self.vocab_size = len(all_pages)
        self.page_to_idx = {p: i+1 for i, p in enumerate(sorted(all_pages))}
        self.idx_to_page = {i+1: p for i, p in enumerate(sorted(all_pages))}
    
    def train_session_encoder(
        self,
        sequences: list[list[str]],
        epochs: int = 30,
        batch_size: int = 256,
        website_id: Optional[str] = None,
    ):
        """
        Train the SessionEncoder using next-item prediction task.
        Uses GPU if available, falls back to CPU.
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for SessionEncoder training")
            return
        
        device = gpu_utils.DEVICE
        logger.info(f"Training SessionEncoder on {device}")
        
        # Build vocabulary
        all_pages = set()
        for seq in sequences:
            all_pages.update(seq)
        self._build_vocab(all_pages)
        
        # Prepare training data
        train_data = []
        for seq in sequences:
            indices = [self.page_to_idx.get(p, 0) for p in seq]
            if len(indices) >= 3:
                train_data.append(indices)
        
        if not train_data:
            logger.warning("No valid training sequences")
            return
        
        max_len = min(max(len(s) for s in train_data), CONFIG.training.max_sequence_length)
        
        # Create model
        self.session_encoder = SessionEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=CONFIG.training.embedding_dim,
            hidden_dim=CONFIG.training.hidden_dim,
            num_layers=CONFIG.training.num_layers,
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            self.session_encoder.parameters(),
            lr=CONFIG.training.learning_rate,
            weight_decay=1e-5,
        )
        
        # Pad sequences
        padded = np.zeros((len(train_data), max_len), dtype=np.int64)
        for i, seq in enumerate(train_data):
            length = min(len(seq), max_len)
            padded[i, :length] = seq[:length]
        
        dataset = torch.tensor(padded, device=device)
        
        # Training loop
        self.session_encoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start:start + batch_size]
                
                # Input: all but last, Target: last
                inputs = batch[:, :-1]
                targets = batch[:, -1]
                
                optimizer.zero_grad()
                embeddings, logits = self.session_encoder.forward_with_logits(inputs)
                loss = F.cross_entropy(logits, targets, ignore_index=0)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.session_encoder.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        self.session_encoder.eval()
        
        # Generate page embeddings (sync to pgvector if website_id provided)
        self._build_page_embeddings(website_id)
        
        self.is_trained = True
        logger.info("SessionEncoder training complete")
    
    def _build_page_embeddings(self, website_id: Optional[str] = None):
        """Generate embeddings for all known pages and sync to pgvector."""
        if self.session_encoder is None:
            return
        
        device = next(self.session_encoder.parameters()).device
        self.page_embeddings = {}
        
        for page, idx in self.page_to_idx.items():
            tensor = torch.tensor([[idx]], device=device)
            with torch.no_grad():
                emb = self.session_encoder(tensor)
                self.page_embeddings[page] = emb.cpu().numpy().flatten()
        
        # Sync to pgvector
        if HAS_PSYCOPG2 and website_id:
            self._sync_to_pgvector(website_id)
        
        # In-memory ANN fallback
        if self.page_embeddings and SKLEARN_AVAILABLE:
            pages = list(self.page_embeddings.keys())
            vectors = np.array([self.page_embeddings[p] for p in pages])
            self.ann_index = NearestNeighbors(
                n_neighbors=min(50, len(pages)),
                metric='cosine',
                algorithm='brute',
            )
            self.ann_index.fit(vectors)
            self.ann_page_order = pages
        
        logger.info(f"Built {len(self.page_embeddings)} page embeddings")
    
    def _sync_to_pgvector(self, website_id: str):
        """Sync page embeddings to pgvector table in TimescaleDB."""
        if not HAS_PSYCOPG2 or not self.page_embeddings:
            return
        
        try:
            conn = psycopg2.connect(CONFIG.db.url)
            cur = conn.cursor()
            
            for page_url, embedding in self.page_embeddings.items():
                emb_str = '[' + ','.join(f'{v:.6f}' for v in embedding) + ']'
                cur.execute("""
                    INSERT INTO page_embeddings (website_id, page_url, embedding, updated_at)
                    VALUES (%s, %s, %s::vector, NOW())
                    ON CONFLICT (website_id, page_url)
                    DO UPDATE SET embedding = %s::vector, updated_at = NOW()
                """, (website_id, page_url, emb_str, emb_str))
            
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Synced {len(self.page_embeddings)} embeddings to pgvector")
        except Exception as e:
            logger.warning(f"pgvector sync failed (using in-memory fallback): {e}")
    
    def retrieve_candidates_pgvector(
        self,
        session_embedding: np.ndarray,
        website_id: str,
        top_k: int = 50,
        exclude: set = None,
    ) -> list[tuple[str, float]]:
        """
        Retrieve candidate items via pgvector ANN search.
        Uses TimescaleDB's IVFFlat index for fast approximate nearest neighbor.
        """
        exclude = exclude or set()
        if not HAS_PSYCOPG2 or website_id is None:
            return self.retrieve_candidates_sklearn(session_embedding, top_k, exclude)
        
        try:
            conn = psycopg2.connect(CONFIG.db.url)
            cur = conn.cursor()
            
            emb_str = '[' + ','.join(f'{v:.6f}' for v in session_embedding) + ']'
            
            exclude_condition = ""
            if exclude:
                exclude_list = ", ".join(f"'{e.replace(chr(39), chr(39)+chr(39))}'" for e in exclude)
                exclude_condition = f"AND page_url NOT IN ({exclude_list})"
            
            query = f"""
                SELECT page_url, 1 - (embedding <=> %s::vector) AS similarity
                FROM page_embeddings
                WHERE website_id = %s {exclude_condition}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            cur.execute(query, (emb_str, website_id, emb_str, top_k))
            results = [(row[0], float(row[1])) for row in cur.fetchall()]
            cur.close()
            conn.close()
            return results
        except Exception as e:
            logger.warning(f"pgvector query failed, using sklearn fallback: {e}")
            return self.retrieve_candidates_sklearn(session_embedding, top_k, exclude)
    
    def retrieve_candidates_sklearn(
        self,
        session_embedding: np.ndarray,
        top_k: int = 50,
        exclude: set = None,
    ) -> list[tuple[str, float]]:
        """Fallback: sklearn NearestNeighbors ANN (in-memory)."""
        exclude = exclude or set()
        
        if self.ann_index is not None and self.ann_page_order:
            distances, indices = self.ann_index.kneighbors(
                session_embedding.reshape(1, -1),
                n_neighbors=min(top_k * 2, len(self.ann_page_order)),
            )
            candidates = []
            for idx, dist in zip(indices[0], distances[0]):
                page = self.ann_page_order[idx]
                if page not in exclude:
                    candidates.append((page, float(1 - dist)))
            return candidates[:top_k]
        
        if not self.page_embeddings:
            return []
        
        similarities = []
        for page, emb in self.page_embeddings.items():
            if page in exclude:
                continue
            sim = float(np.dot(session_embedding, emb) / 
                       (np.linalg.norm(session_embedding) * np.linalg.norm(emb) + 1e-8))
            similarities.append((page, sim))
        
        return sorted(similarities, key=lambda x: -x[1])[:top_k]
    
    def encode_session(self, pages: list[str]) -> Optional[np.ndarray]:
        """Encode a session into an embedding vector"""
        if self.session_encoder is None:
            return None
        
        device = next(self.session_encoder.parameters()).device
        indices = [self.page_to_idx.get(p, 0) for p in pages[-30:]]
        
        if not indices:
            return None
        
        tensor = torch.tensor([indices], device=device)
        
        with torch.no_grad():
            self.session_encoder.eval()
            embedding = self.session_encoder(tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def retrieve_candidates(
        self,
        session_embedding: np.ndarray,
        top_k: int = 50,
        exclude: set = None,
    ) -> list[tuple[str, float]]:
        """
        Retrieve candidate items via ANN search.
        Uses pgvector-style cosine similarity.
        """
        exclude = exclude or set()
        
        if self.ann_index is not None and self.ann_page_order:
            # Fast ANN retrieval
            distances, indices = self.ann_index.kneighbors(
                session_embedding.reshape(1, -1),
                n_neighbors=min(top_k * 2, len(self.ann_page_order)),
            )
            
            candidates = []
            for idx, dist in zip(indices[0], distances[0]):
                page = self.ann_page_order[idx]
                if page not in exclude:
                    candidates.append((page, float(1 - dist)))  # convert distance to similarity
            
            return candidates[:top_k]
        
        # Fallback: brute force over all page embeddings
        if not self.page_embeddings:
            return []
        
        similarities = []
        for page, emb in self.page_embeddings.items():
            if page in exclude:
                continue
            sim = float(np.dot(session_embedding, emb) / 
                       (np.linalg.norm(session_embedding) * np.linalg.norm(emb) + 1e-8))
            similarities.append((page, sim))
        
        return sorted(similarities, key=lambda x: -x[1])[:top_k]
    
    def rank_candidates(
        self,
        session_pages: list[str],
        candidates: list[tuple[str, float]],
        session_features: Optional[dict] = None,
    ) -> list[dict]:
        """
        Rank candidates using session context.
        Uses a simple scoring function combining:
        - Cosine similarity (from ANN)
        - Recency (pages seen earlier in session should influence less)
        - Optional session features
        """
        if not candidates:
            return []
        
        # Boost recency: pages similar to recent views get a boost
        recent_pages = session_pages[-3:] if session_pages else []
        recent_boost = {}
        
        if recent_pages and self.page_embeddings:
            for candidate, base_score in candidates:
                boost = 0.0
                for recent in recent_pages:
                    if recent in self.page_embeddings and candidate in self.page_embeddings:
                        recent_emb = self.page_embeddings[recent]
                        cand_emb = self.page_embeddings[candidate]
                        sim = float(np.dot(recent_emb, cand_emb) /
                                   (np.linalg.norm(recent_emb) * np.linalg.norm(cand_emb) + 1e-8))
                        boost += sim * 0.1  # 10% recency boost
                
                recent_boost[candidate] = boost
        
        # Compute final scores
        ranked = []
        for candidate, base_similarity in candidates:
            final_score = base_similarity
            
            # Add recency boost
            if candidate in recent_boost:
                final_score = min(1.0, final_score + recent_boost[candidate])
            
            ranked.append({
                'page': candidate,
                'score': round(float(final_score), 4),
                'base_similarity': round(float(base_similarity), 4),
            })
        
        return sorted(ranked, key=lambda x: -x['score'])[:20]
    
    def recommend(
        self,
        session_pages: list[str],
        session_features: Optional[dict] = None,
        top_k: int = 20,
        mode: str = "token",
        semantic_weight: float = 0.6,
        website_id: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_mode: str = "local",
        embedding_api_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> list[dict]:
        """
        Full recommendation pipeline for a session.

        Args:
            session_pages: Pages visited in current session
            session_features: Optional session metadata dict
            top_k: Number of recommendations
            mode: 'token' (GRU), 'semantic' (SentenceTransformer), or 'hybrid'
            semantic_weight: Weight for semantic scores in hybrid mode (0-1)
            website_id: Required for 'semantic' and 'hybrid' modes
            embedding_model: 'minilm' (384d) or 'qwen3' (2048d)
            embedding_mode: 'local' (GPU/CPU) or 'cloud' (API)
            embedding_api_url: Optional API endpoint URL for cloud mode
            embedding_api_key: Optional API key for cloud mode

        Returns:
            Ranked list of {page, score, base_similarity} dicts
        """
        if mode == "hybrid":
            return self._hybrid_recommend(session_pages, session_features, top_k, semantic_weight, website_id, embedding_model, embedding_mode, embedding_api_url, embedding_api_key)

        if mode == "semantic" and SEMANTIC_AVAILABLE:
            return self._semantic_recommend(session_pages, website_id or "", top_k, embedding_model, embedding_mode, embedding_api_url, embedding_api_key)

        # Default: token-based (GRU + pgvector)
        return self._token_recommend(session_pages, session_features, top_k)

    def _token_recommend(
        self,
        session_pages: list[str],
        session_features: Optional[dict] = None,
        top_k: int = 20,
    ) -> list[dict]:
        """Token-based recommendation using GRU session encoder + pgvector."""
        if not self.is_trained:
            logger.warning("Recommender not trained")
            return []

        if len(session_pages) < 2:
            return self._cold_start_recommend(top_k)

        embedding = self.encode_session(session_pages)
        if embedding is None:
            return self._cold_start_recommend(top_k)

        exclude = set(session_pages)
        candidates = self.retrieve_candidates(embedding, top_k * 3, exclude)

        if not candidates:
            return self._cold_start_recommend(top_k)

        return self.rank_candidates(session_pages, candidates, session_features)[:top_k]

    def _semantic_recommend(
        self,
        session_pages: list[str],
        website_id: str,
        top_k: int = 20,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_mode: str = "local",
        embedding_api_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> list[dict]:
        """Semantic recommendation using configurable embedding model."""
        if not SEMANTIC_AVAILABLE:
            logger.warning("Semantic embedder not available, falling back to token")
            return self._token_recommend(session_pages, None, top_k)

        exclude = set(session_pages) if session_pages else set()

        page_texts = [semantic_embedder.embed_page_url(p) for p in session_pages]
        query_text = " [SEP] ".join(page_texts)

        if session_pages:
            last_text = semantic_embedder.embed_page_url(session_pages[-1])
            query_text = f"{last_text} [SEP] {query_text}"

        # Temporarily set env vars for cloud config if provided
        old_url = os.environ.get("EMBEDDING_API_URL")
        old_key = os.environ.get("EMBEDDING_API_KEY")
        old_dim = os.environ.get("EMBEDDING_DIM_QWEN3")
        if embedding_api_url:
            os.environ["EMBEDDING_API_URL"] = embedding_api_url
        if embedding_api_key:
            os.environ["EMBEDDING_API_KEY"] = embedding_api_key
        if embedding_dim:
            os.environ["EMBEDDING_DIM_QWEN3"] = str(embedding_dim)
        
        try:
            candidates = semantic_embedder.retrieve_semantic_candidates(
                query_text, website_id, top_k=top_k * 3, exclude=exclude,
                model=embedding_model, mode=embedding_mode,
            )
        finally:
            # Restore env vars
            if old_url is not None:
                os.environ["EMBEDDING_API_URL"] = old_url
            elif embedding_api_url:
                os.environ.pop("EMBEDDING_API_URL", None)
            if old_key is not None:
                os.environ["EMBEDDING_API_KEY"] = old_key
            elif embedding_api_key:
                os.environ.pop("EMBEDDING_API_KEY", None)

        if not candidates:
            return self._cold_start_recommend(top_k)

        return [
            {'page': p, 'score': round(float(s), 4), 'base_similarity': round(float(s), 4)}
            for p, s in candidates[:top_k]
        ]

    def _retrieve_semantic_fallback(
        self,
        query_text: str,
        top_k: int,
        exclude: set,
    ) -> list[tuple[str, float]]:
        """In-memory fallback for semantic retrieval using sklearn."""
        if not self.page_embeddings or not SEMANTIC_AVAILABLE:
            return []

        # Use sklearn NearestNeighbors on stored GRU embeddings as fallback
        if self.ann_index is not None and self.ann_page_order:
            # Still use token-based ANN as fallback
            return self.retrieve_candidates_sklearn(
                np.zeros(256), top_k, exclude  # dummy — just returns popular
            )
        return []

    def _hybrid_recommend(
        self,
        session_pages: list[str],
        session_features: Optional[dict] = None,
        top_k: int = 20,
        semantic_weight: float = 0.6,
        website_id: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_mode: str = "local",
        embedding_api_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> list[dict]:
        """
        Hybrid recommendation: fuses token-based + semantic scores.
        """
        token_weight = 1.0 - semantic_weight

        token_results = self._token_recommend(session_pages, session_features, top_k * 2)
        semantic_results = []
        if SEMANTIC_AVAILABLE:
            semantic_results = self._semantic_recommend(session_pages, website_id or "", top_k * 2, embedding_model, embedding_mode, embedding_api_url, embedding_api_key)

        # If one mode returned nothing, use the other exclusively
        if not token_results:
            return semantic_results[:top_k]
        if not semantic_results:
            return token_results[:top_k]

        # Fuse scores using weighted average by rank position
        all_pages = {}
        for rank, r in enumerate(token_results):
            all_pages[r['page']] = {
                'page': r['page'],
                'token_score': r['score'] * token_weight,
                'semantic_score': 0.0,
                'base_similarity': r['base_similarity'],
            }
        for rank, r in enumerate(semantic_results):
            page = r['page']
            if page not in all_pages:
                all_pages[page] = {
                    'page': page,
                    'token_score': 0.0,
                    'semantic_score': 0.0,
                    'base_similarity': r['base_similarity'],
                }
            all_pages[page]['semantic_score'] = r['score'] * semantic_weight
            all_pages[page]['base_similarity'] = max(
                all_pages[page].get('base_similarity', 0), r['base_similarity']
            )

        # Final score: weighted sum
        for p in all_pages.values():
            p['score'] = round(p['token_score'] + p['semantic_score'], 4)

        return sorted(all_pages.values(), key=lambda x: -x['score'])[:top_k]

    def _cold_start_recommend(self, top_k: int = 20) -> list[dict]:
        """Fallback: recommend popular pages when not enough session data."""
        if not self.ann_page_order:
            return []
        return [
            {'page': p, 'score': 1.0 - i * 0.01, 'base_similarity': 0.0}
            for i, p in enumerate(self.ann_page_order[:top_k])
        ]

    def predict(self, *args, **kwargs):
        """Override base predict - delegates to recommend"""
        return self.recommend(*args, **kwargs)

    def train(self, *args, **kwargs):
        """Override base train - delegates to session_encoder training"""
        return self.train_session_encoder(*args, **kwargs)

    def to_onnx(self, path: Optional[str] = None) -> str:
        """Export session encoder to ONNX"""
        if self.session_encoder is None:
            raise RuntimeError("Session encoder not trained")
        
        path = path or CONFIG.paths.recommender_model
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        device = next(self.session_encoder.parameters()).device
        dummy = torch.randint(1, self.vocab_size, (1, 10), device=device)
        
        torch.onnx.export(
            self.session_encoder,
            dummy,
            path,
            input_names=['input_sequence'],
            output_names=['session_embedding'],
            dynamic_axes={
                'input_sequence': {0: 'batch_size', 1: 'seq_length'},
                'session_embedding': {0: 'batch_size'},
            },
            opset_version=17,
        )
        
        logger.info(f"Recommender exported to ONNX: {path}")
        return path
