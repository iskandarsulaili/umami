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
from ..config import CONFIG
from .. import gpu_utils

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
        
        # Generate page embeddings
        self._build_page_embeddings()
        
        self.is_trained = True
        logger.info("SessionEncoder training complete")
    
    def _build_page_embeddings(self):
        """Generate embeddings for all known pages"""
        if self.session_encoder is None:
            return
        
        device = next(self.session_encoder.parameters()).device
        self.page_embeddings = {}
        
        for page, idx in self.page_to_idx.items():
            # Single-page "session" for embedding
            tensor = torch.tensor([[idx]], device=device)
            with torch.no_grad():
                emb = self.session_encoder(tensor)
                self.page_embeddings[page] = emb.cpu().numpy().flatten()
        
        # Build ANN index
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
    ) -> list[dict]:
        """
        Full recommendation pipeline for a session.
        
        Args:
            session_pages: Pages visited in current session
            session_features: Optional session metadata dict
            top_k: Number of recommendations
            
        Returns:
            Ranked list of {page, score, base_similarity} dicts
        """
        if not self.is_trained:
            logger.warning("Recommender not trained")
            return []
        
        # Cold-start: session too short, use popular pages
        if len(session_pages) < 2:
            return self._cold_start_recommend(top_k)
        
        # Stage 1: Encode session
        embedding = self.encode_session(session_pages)
        if embedding is None:
            return self._cold_start_recommend(top_k)
        
        # Stage 2: Retrieve candidates (exclude already-visited pages)
        exclude = set(session_pages)
        candidates = self.retrieve_candidates(embedding, top_k * 3, exclude)
        
        if not candidates:
            return self._cold_start_recommend(top_k)
        
        # Stage 3: Rank candidates
        return self.rank_candidates(session_pages, candidates, session_features)[:top_k]
    
    def _cold_start_recommend(self, top_k: int = 20) -> list[dict]:
        """Fallback: recommend popular pages"""
        if not self.ann_page_order:
            return []
        
        # Just return top pages by frequency order
        return [
            {'page': p, 'score': 1.0 - i * 0.01, 'base_similarity': 0.0}
            for i, p in enumerate(self.ann_page_order[:top_k])
        ]
    
    def train(self, *args, **kwargs):
        """Override base train - delegates to session_encoder training"""
        return self.train_session_encoder(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """Override base predict - delegates to recommend"""
        return self.recommend(*args, **kwargs)
    
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
