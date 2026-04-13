"""
SVAGFormer: Modular framework for Spatio-temporal Video Action Grounding (SVAG).
Combines spatial grounding (TempRMOT-style) and temporal grounding (FlashVTG-style).

Architecture:
  - Text Encoder   : LLaMA-2 (frozen) → language features
  - Visual Encoder : InternVideo2 (frozen) → video features
  - Spatial Head   : Transformer-based multi-object tracker (TempRMOT)
  - Temporal Head  : FlashVTG with temporal feature layering + adaptive score refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Text Encoder (LLaMA-2 wrapper – frozen)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Wraps a HuggingFace LLM for query feature extraction."""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", hidden_dim: int = 256):
        super().__init__()
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            llm_dim = self.backbone.config.hidden_size
        except Exception:
            # Fallback: lightweight text encoder for development / CI
            self.tokenizer = None
            self.backbone = nn.Embedding(30522, 768)  # BERT-vocab size
            llm_dim = 768

        self.proj = nn.Linear(llm_dim, hidden_dim)
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, queries: List[str]) -> torch.Tensor:
        """
        Args:
            queries: list of natural language action queries
        Returns:
            text_feats: (B, hidden_dim)
        """
        if self.tokenizer is not None:
            tokens = self.tokenizer(queries, return_tensors="pt",
                                    padding=True, truncation=True, max_length=64)
            tokens = {k: v.to(next(self.parameters()).device) for k, v in tokens.items()}
            with torch.no_grad():
                out = self.backbone(**tokens)
            feats = out.last_hidden_state[:, 0]          # CLS token
        else:
            # Dummy path
            device = next(self.parameters()).device
            ids = torch.zeros(len(queries), 1, dtype=torch.long, device=device)
            feats = self.backbone(ids).squeeze(1)

        return self.proj(feats)                          # (B, D)


# ---------------------------------------------------------------------------
# Visual Encoder (InternVideo2 wrapper – frozen)
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """Wraps InternVideo2 for per-frame / temporal feature extraction."""

    def __init__(self, hidden_dim: int = 256, pretrained: bool = False):
        super().__init__()
        # Placeholder backbone – replace with InternVideo2 when available
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 4, 4)),
        )
        self.proj = nn.Linear(64 * 4 * 4, hidden_dim)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, T, C, H, W)  – raw frames
        Returns:
            visual_feats: (B, T, hidden_dim)
        """
        B, T, C, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4)          # (B, C, T, H, W)
        x = self.backbone(x)                        # (B, 64, T, 4, 4)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 64, 4, 4)
        x = x.view(B, T, -1)
        return self.proj(x)                          # (B, T, D)


# ---------------------------------------------------------------------------
# Spatial Grounding Head  (TempRMOT-style)
# ---------------------------------------------------------------------------

class QueryMemory(nn.Module):
    """Lightweight query memory that stores track embeddings across frames."""

    def __init__(self, hidden_dim: int, memory_length: int = 5):
        super().__init__()
        self.memory_length = memory_length
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def update(self, prev_memory: Optional[torch.Tensor],
               curr_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_memory: (num_queries, D) or None
            curr_query : (num_queries, D)
        Returns:
            new_memory : (num_queries, D)
        """
        if prev_memory is None:
            prev_memory = torch.zeros_like(curr_query)
        return self.gru(curr_query, prev_memory)


class SpatialGroundingHead(nn.Module):
    """
    Referring Multi-Object Tracking head (TempRMOT-inspired).
    Detects and tracks all referent objects per query across frames.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 100,
                 num_heads: int = 8, num_layers: int = 6, memory_length: int = 5):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable object queries
        self.object_queries = nn.Embedding(num_queries, hidden_dim)

        # Cross-attention between queries and visual features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Text-conditioned gating
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Prediction heads
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)   # (cx, cy, w, h) normalized
        )
        self.cls_head = nn.Linear(hidden_dim, 1)  # foreground / background

        # Query memory for temporal consistency
        self.memory = QueryMemory(hidden_dim, memory_length)

    def forward(self, visual_feats: torch.Tensor,
                text_feats: torch.Tensor,
                prev_memory: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_feats : (B, T, D)  – per-frame features
            text_feats   : (B, D)     – per-query text embedding
            prev_memory  : (B * num_queries, D) or None
        Returns:
            bboxes       : (B, T, num_queries, 4)
            scores       : (B, T, num_queries, 1)
            memory       : (B * num_queries, D)
        """
        B, T, D = visual_feats.shape
        Nq = self.num_queries

        queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Nq, D)

        all_bboxes, all_scores = [], []
        memory = prev_memory

        for t in range(T):
            frame_feat = visual_feats[:, t:t+1, :]         # (B, 1, D)

            # Cross-attend
            out = self.decoder(queries, frame_feat)          # (B, Nq, D)

            # Text gating
            txt = text_feats.unsqueeze(1).expand(-1, Nq, -1)
            gate = self.text_gate(torch.cat([out, txt], dim=-1))
            out = out * gate                                  # (B, Nq, D)

            # Update memory
            out_flat = out.reshape(B * Nq, D)
            memory_flat = memory.reshape(B * Nq, D) if memory is not None else None
            memory_flat = self.memory.update(memory_flat, out_flat)
            memory = memory_flat.reshape(B, Nq, D)
            queries = memory

            bboxes = torch.sigmoid(self.bbox_head(out))     # (B, Nq, 4)
            scores = self.cls_head(out)                      # (B, Nq, 1)
            all_bboxes.append(bboxes)
            all_scores.append(scores)

        bboxes = torch.stack(all_bboxes, dim=1)             # (B, T, Nq, 4)
        scores = torch.stack(all_scores, dim=1)             # (B, T, Nq, 1)

        return bboxes, scores, memory_flat


# ---------------------------------------------------------------------------
# Temporal Feature Layering  (FlashVTG-inspired)
# ---------------------------------------------------------------------------

class TemporalFeatureLayering(nn.Module):
    """Multi-scale temporal feature pyramid with self-attention at each scale."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 5, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            pyramid: list of feature maps at different temporal scales
        """
        pyramid = []
        cur = x
        for layer in self.layers:
            cur = layer(cur)
            pyramid.append(cur)
            if cur.size(1) > 1:
                cur = self.pool(cur.permute(0, 2, 1)).permute(0, 2, 1)  # down-sample T
        return pyramid


class AdaptiveScoreRefinement(nn.Module):
    """Refines moment scores using cross-scale attention."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_feat: torch.Tensor, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            query_feat : (B, 1, D)
            pyramid    : list of (B, T_i, D)
        Returns:
            refined    : (B, 1, D)
        """
        out = query_feat
        for level in pyramid:
            attn_out, _ = self.attn(out, level, level)
            out = self.norm(out + attn_out)
        return out


# ---------------------------------------------------------------------------
# Temporal Grounding Head (FlashVTG-inspired)
# ---------------------------------------------------------------------------

class TemporalGroundingHead(nn.Module):
    """
    Text-guided video temporal grounding head.
    Predicts (start, end) frame indices and confidence score.
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 5, num_heads: int = 8):
        super().__init__()
        self.layering = TemporalFeatureLayering(hidden_dim, num_layers, num_heads)
        self.refinement = AdaptiveScoreRefinement(hidden_dim)

        # Cross-attention: text query attends to video
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Prediction
        self.moment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)    # (start, end) in [0,1]
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, visual_feats: torch.Tensor,
                text_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_feats : (B, T, D)
            text_feats   : (B, D)
        Returns:
            moments      : (B, 2)  – normalized (start, end)
            tscores      : (B, 1)  – confidence
        """
        # Build temporal pyramid
        pyramid = self.layering(visual_feats)              # list of (B, T_i, D)

        # Text-guided cross attention on finest scale
        txt_q = text_feats.unsqueeze(1)                    # (B, 1, D)
        ctx, _ = self.cross_attn(txt_q, visual_feats, visual_feats)
        ctx = self.norm(ctx + txt_q)

        # Adaptive refinement across scales
        refined = self.refinement(ctx, pyramid)            # (B, 1, D)
        refined = refined.squeeze(1)                       # (B, D)

        moments = torch.sigmoid(self.moment_head(refined))
        tscores = self.score_head(refined)
        return moments, tscores


# ---------------------------------------------------------------------------
# Full SVAGFormer
# ---------------------------------------------------------------------------

class SVAGFormer(nn.Module):
    """
    Full SVAG pipeline:
      1. Encode text query → text_feats
      2. Encode video frames → visual_feats
      3. Temporal Grounding Head → predicted temporal window
      4. Spatial  Grounding Head → per-frame bounding boxes & IDs
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D = cfg.get("hidden_dim", 256)

        self.text_encoder = TextEncoder(
            model_name=cfg.get("text_model", "meta-llama/Llama-2-7b-hf"),
            hidden_dim=D
        )
        self.visual_encoder = VisualEncoder(hidden_dim=D)

        self.temporal_head = TemporalGroundingHead(
            hidden_dim=D,
            num_layers=cfg.get("tfl_layers", 5),
            num_heads=cfg.get("num_heads", 8)
        )
        self.spatial_head = SpatialGroundingHead(
            hidden_dim=D,
            num_queries=cfg.get("num_queries", 100),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("spatial_layers", 6),
            memory_length=cfg.get("memory_length", 5)
        )

    def forward(self, videos: torch.Tensor, queries: List[str],
                prev_memory: Optional[torch.Tensor] = None):
        """
        Args:
            videos  : (B, T, C, H, W)
            queries : list of B natural-language strings
        Returns:
            dict with keys: moments, tscores, bboxes, scores, memory
        """
        text_feats = self.text_encoder(queries)             # (B, D)
        visual_feats = self.visual_encoder(videos)          # (B, T, D)

        moments, tscores = self.temporal_head(visual_feats, text_feats)
        bboxes, scores, memory = self.spatial_head(visual_feats, text_feats, prev_memory)

        return {
            "moments": moments,       # (B, 2) – normalized start/end
            "tscores": tscores,       # (B, 1)
            "bboxes": bboxes,         # (B, T, Nq, 4)
            "scores": scores,         # (B, T, Nq, 1)
            "memory": memory          # track memory for next clip
        }
