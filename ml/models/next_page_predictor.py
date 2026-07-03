"""
Next Page Predictor - Model 1
Predicts the most likely next page(s) a visitor will view.
Uses a two-tier approach:
  1. Markov chain (fast, lightweight, CPU) for real-time predictions
  2. Transformer (GPU) for batch training offline, fallback ONNX inference

Technique: arxiv:1405.7868 (Markov model), arxiv:1811.00855 (SR-GNN),
           arxiv:2102.01922 (Self-Attention for SBR)
"""

import os
import json
import pickle
import logging
from typing import Optional
from collections import defaultdict
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


class MarkovChain:
    """
    Variable-order Markov chain for next page prediction.
    Fast, interpretable, runs entirely on CPU.
    Supports up to order-5 for context-aware predictions.
    
    Reference: arxiv:1405.7868 - "A Vague Improved Markov Model 
               Approach for Web Page Prediction"
    """
    
    def __init__(self, max_order: int = 3, laplace_smoothing: float = 0.01):
        self.max_order = max_order
        self.laplacian = laplace_smoothing
        # transitions[order][(page_1, ..., page_N)] = {next_page: count}
        self.transitions: list[dict] = [defaultdict(lambda: defaultdict(float))]
        self.page_counts: dict[str, float] = defaultdict(float)
        self.total_transitions = 0
    
    def fit(self, sequences: list[list[str]]):
        """Train Markov chain from page view sequences"""
        for seq in sequences:
            if len(seq) < 2:
                continue
            
            for i in range(1, len(seq)):
                current = seq[i]
                prev = seq[i-1]
                self.page_counts[current] += 1.0
                self.page_counts[prev] += 1.0
                self.total_transitions += 1
                
                # order-1 transitions
                self.transitions[0][(prev,)][current] += 1.0
                
                # higher-order transitions
                for order in range(2, min(self.max_order, i) + 1):
                    context = tuple(seq[i-order:i])
                    if context not in self.transitions[0]:
                        # Extend transitions list if needed
                        while len(self.transitions) < order:
                            self.transitions.append(defaultdict(lambda: defaultdict(float)))
                    self.transitions[order-1][context][current] += 1.0
        
        logger.info(f"Markov chain trained: {len(self.page_counts)} pages, "
                    f"{self.total_transitions} transitions")
    
    def predict(self, context: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """
        Predict next pages given context sequence.
        Uses Kneser-Ney-like backoff for higher-order smoothing.
        """
        if not context:
            # Fall back to most popular pages
            total = sum(self.page_counts.values()) or 1.0
            probs = [(p, c / total) for p, c in self.page_counts.most_common(top_k)]
            return probs[:top_k]
        
        scores = defaultdict(float)
        
        # Try from highest order to lowest (backoff)
        for order in range(min(len(context), self.max_order), 0, -1):
            ctx = tuple(context[-order:])
            if ctx in self.transitions[order-1]:
                trans = self.transitions[order-1][ctx]
                total = sum(trans.values()) + self.laplacian * len(self.page_counts)
                
                for page, count in trans.items():
                    # Higher weight for higher-order matches
                    scores[page] += (count / total) * (order / self.max_order)
                
                # Backoff weight decreases with order
                break  # Only use highest matching order
        
        # Sort by score and return top-k
        sorted_pages = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_pages[:top_k]


class PageTransformer(nn.Module):
    """
    Transformer-based page sequence model.
    Uses self-attention to capture long-range page view patterns.
    GPU-accelerated training with ONNX export for inference.
    
    Reference: arxiv:2102.01922 - "Session-based Recommendation
               with Self-Attention Networks"
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=0)
        self.pos_encoder = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(embedding_dim, vocab_size + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        embedded = self.embedding(x) + self.pos_encoder(positions)
        embedded = self.dropout(embedded)
        
        # Causal masking for autoregressive prediction
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        
        output = self.transformer(embedded, mask=mask)
        # Predict next page from last position
        return self.output(output[:, -1, :])


class NextPagePredictor(BaseModel):
    """
    Model 1: Next Page Predictor
    Combines Markov chain (fast, real-time) with Transformer (batch, GPU).
    
    Architecture:
    - Real-time inference: Markov chain (sub-millisecond, CPU)
    - Batch training: Transformer on GPU -> exported to ONNX -> CPU/GPU inference
    - Fallback: Markov chain always available
    """
    
    def __init__(self):
        super().__init__(name="next_page_predictor", version="1.0.0")
        self.markov = MarkovChain(max_order=3)
        self.transformer: Optional[PageTransformer] = None
        self.page_to_idx: dict[str, int] = {}
        self.idx_to_page: dict[int, str] = {}
        self.vocab_size = 0
    
    def _build_vocabulary(self, sequences: list[list[str]]):
        """Build page vocabulary from training data"""
        all_pages = set()
        for seq in sequences:
            all_pages.update(seq)
        
        self.page_to_idx = {p: i+1 for i, p in enumerate(sorted(all_pages))}
        self.idx_to_page = {i+1: p for i, p in enumerate(sorted(all_pages))}
        self.vocab_size = len(all_pages)
        self.metadata['vocab_size'] = self.vocab_size
        logger.info(f"Vocabulary: {self.vocab_size} unique pages")
    
    def train(
        self,
        sequences: list[list[str]],
        use_transformer: bool = True,
        epochs: int = 20,
        batch_size: int = 256,
    ):
        """
        Train both Markov chain and optionally Transformer.
        
        Args:
            sequences: List of page view sequences
            use_transformer: Whether to train the Transformer (GPU)
            epochs: Training epochs for Transformer
            batch_size: Batch size for Transformer training
        """
        logger.info(f"Training NextPagePredictor on {len(sequences)} sequences")
        
        # Always train Markov chain (fast, always available)
        self.markov.fit(sequences)
        self._build_vocabulary(sequences)
        
        # Optionally train Transformer (GPU) for better accuracy
        if use_transformer and TORCH_AVAILABLE and len(sequences) > 100:
            self._train_transformer(sequences, epochs, batch_size)
        else:
            if not TORCH_AVAILABLE:
                logger.info("PyTorch not available, skipping Transformer training")
            elif len(sequences) <= 100:
                logger.info(f"Too few sequences ({len(sequences)}), "
                           "skipping Transformer training")
        
        self.is_trained = True
        logger.info("NextPagePredictor training complete")
    
    def _train_transformer(
        self,
        sequences: list[list[str]],
        epochs: int,
        batch_size: int
    ):
        """Train Transformer on GPU"""
        device = gpu_utils.DEVICE
        logger.info(f"Training Transformer on {device}")
        
        # Prepare training data
        train_seqs = []
        for seq in sequences:
            indices = [self.page_to_idx.get(p, 0) for p in seq]
            if len(indices) >= 2:
                train_seqs.append(indices)
        
        if not train_seqs:
            logger.warning("No valid training sequences for Transformer")
            return
        
        # Create model
        max_seq_len = min(max(len(s) for s in train_seqs), CONFIG.training.max_sequence_length)
        self.transformer = PageTransformer(
            vocab_size=self.vocab_size,
            embedding_dim=CONFIG.training.embedding_dim,
            hidden_dim=CONFIG.training.hidden_dim,
            num_layers=CONFIG.training.num_layers,
            max_seq_len=max_seq_len,
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=CONFIG.training.learning_rate
        )
        
        # Pad sequences to same length
        padded = np.zeros((len(train_seqs), max_seq_len), dtype=np.int64)
        for i, seq in enumerate(train_seqs):
            length = min(len(seq), max_seq_len)
            padded[i, :length] = seq[:length]
        
        dataset = torch.tensor(padded, device=device)
        
        # Training loop
        self.transformer.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start:start + batch_size]
                
                # Input: first N-1 pages, Target: last N-1 pages shifted
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                optimizer.zero_grad()
                output = self.transformer(inputs)
                
                loss = F.cross_entropy(
                    output.reshape(-1, self.vocab_size + 1),
                    targets.reshape(-1),
                    ignore_index=0
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        self.transformer.eval()
        self.metadata['transformer_trained'] = True
        self.metadata['transformer_epochs'] = epochs
        logger.info("Transformer training complete")
    
    def predict(
        self,
        context: list[str],
        top_k: int = 10,
        use_transformer: bool = False
    ) -> list[dict]:
        """
        Predict next pages given current session context.
        
        Args:
            context: Current page sequence in session
            top_k: Number of predictions to return
            use_transformer: Use Transformer instead of Markov
            
        Returns:
            List of {page, probability} dicts
        """
        if use_transformer and self.transformer is not None and TORCH_AVAILABLE:
            return self._predict_transformer(context, top_k)
        
        return self._predict_markov(context, top_k)
    
    def _predict_markov(self, context: list[str], top_k: int) -> list[dict]:
        """Fast Markov chain prediction"""
        predictions = self.markov.predict(context, top_k)
        return [
            {'page': page, 'probability': round(prob, 4)}
            for page, prob in predictions
        ]
    
    def _predict_transformer(self, context: list[str], top_k: int) -> list[dict]:
        """GPU-accelerated Transformer prediction"""
        self.transformer.eval()
        device = next(self.transformer.parameters()).device
        
        indices = [self.page_to_idx.get(p, 0) for p in context[-30:]]
        if not indices:
            return self._predict_markov(context, top_k)
        
        tensor = torch.tensor([indices], device=device)
        
        with torch.no_grad():
            output = self.transformer(tensor)
            probs = F.softmax(output[0], dim=0)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probs, min(top_k, self.vocab_size + 1))
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            if idx == 0:
                continue  # Skip padding
            page = self.idx_to_page.get(idx, f"<UNKNOWN_{idx}>")
            results.append({'page': page, 'probability': round(float(prob), 4)})
        
        return results[:top_k]
    
    def get_popular_pages(self, top_k: int = 20) -> list[dict]:
        """Get most popular pages for cold-start recommendations"""
        total = sum(self.markov.page_counts.values())
        sorted_pages = sorted(
            self.markov.page_counts.items(),
            key=lambda x: -x[1]
        )[:top_k]
        
        return [
            {'page': page, 'probability': round(count / total, 4)}
            for page, count in sorted_pages
        ]
    
    def to_onnx(self, path: Optional[str] = None) -> str:
        """Export trained Transformer to ONNX"""
        if self.transformer is None:
            raise RuntimeError("Transformer not trained, cannot export to ONNX")
        
        path = path or CONFIG.paths.next_page_model
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        device = next(self.transformer.parameters()).device
        dummy_input = torch.randint(1, self.vocab_size, (1, 10), device=device)
        
        torch.onnx.export(
            self.transformer,
            dummy_input,
            path,
            input_names=['input_ids'],
            output_names=['next_page_logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'next_page_logits': {0: 'batch_size'},
            },
            opset_version=17,
        )
        
        logger.info(f"Transformer exported to ONNX: {path}")
        return path
