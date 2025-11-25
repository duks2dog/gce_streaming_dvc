"""Utilities for constructing Gemini payload segments from Streaming DVC features."""
from __future__ import annotations

import base64
import importlib
import io
from typing import Any, Dict, List, Optional

np = importlib.import_module("numpy")
_image_module = importlib.import_module("PIL.Image")
Image = getattr(_image_module, "Image")


def encode_frame(frame: Any) -> str:
    """Convert a numpy frame to base64 JPEG."""
    pil_image = _image_module.fromarray(frame.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def select_top_segments(
    importance_scores: Any,
    token_indices: Any,
    frame_indices: Any,
    frames: Optional[List[Any]] = None,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Select top-K segments and optionally attach encoded frames."""
    scores = importance_scores[0]
    token_indices = token_indices[:top_k]
    segments: List[Dict[str, Any]] = []
    for rank, token_idx in enumerate(token_indices):
        frame_idx = int(frame_indices[min(rank, len(frame_indices) - 1)])
        if frames:
            frame_idx = max(0, min(frame_idx, len(frames) - 1))
            frame = frames[frame_idx]
        else:
            frame = None
        segments.append(
            {
                "segment_index": rank,
                "token_index": int(token_idx),
                "importance": float(scores[token_idx]),
                "frame_index": frame_idx,
            }
        )
        if frame is not None:
            segments[-1]["image_base64"] = encode_frame(frame)
    return segments


def build_memory_summary(
    contextualized_features: Any,
    token_indices: Any,
    importance_scores: Any,
    top_k: int = 4,
    pool_dim: int = 32,
) -> List[Dict[str, Any]]:
    """Convert contextualized memory tokens into JSON-friendly embeddings."""
    if contextualized_features is None:
        return []
    feats = np.array(contextualized_features)[0]
    scores = np.array(importance_scores)[0]
    summary = []
    k = min(top_k, len(token_indices))
    for rank in range(k):
        token_idx = int(token_indices[rank])
        embedding = feats[token_idx]
        if pool_dim and pool_dim < embedding.shape[-1]:
            stride = embedding.shape[-1] // pool_dim
            pooled = embedding[: stride * pool_dim].reshape(pool_dim, stride).mean(axis=1)
        else:
            pooled = embedding
        summary.append(
            {
                "segment_index": rank,
                "token_index": token_idx,
                "importance": float(scores[token_idx]),
                "embedding": pooled.astype(float).tolist(),
            }
        )
    return summary
