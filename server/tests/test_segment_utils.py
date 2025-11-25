from __future__ import annotations

import base64
import importlib

import numpy as np

segment_utils = importlib.import_module("gce_streaming.server.segment_utils")
select_top_segments = getattr(segment_utils, "select_top_segments")
build_memory_summary = getattr(segment_utils, "build_memory_summary")


def test_select_top_segments_returns_top_importance() -> None:
    scores = np.array([[0.1, 0.9, 0.3, 0.7]], dtype=np.float32)
    token_indices = np.array([1, 3, 0, 2])
    frame_indices = np.array([0, 1, 2, 3])
    frames = [(np.ones((2, 2, 3), dtype=np.uint8) * (i * 30)) for i in range(4)]

    segments = select_top_segments(scores, token_indices, frame_indices, frames, top_k=2)

    assert len(segments) == 2
    assert segments[0]["token_index"] == 1
    assert segments[0]["frame_index"] == 0
    assert segments[1]["token_index"] == 3
    assert segments[1]["frame_index"] == 1
    assert all("image_base64" in seg for seg in segments)
    decoded = base64.b64decode(segments[0]["image_base64"])  # ensure it's valid base64
    assert len(decoded) > 0


def test_build_memory_summary_pools_embeddings() -> None:
    contextualized = np.arange(1 * 4 * 8).reshape(1, 4, 8).astype(np.float32)
    token_indices = np.array([3, 2, 1, 0])
    scores = np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float32)

    summary = build_memory_summary(contextualized, token_indices, scores, top_k=2, pool_dim=4)

    assert len(summary) == 2
    assert summary[0]["token_index"] == 3
    assert len(summary[0]["embedding"]) == 4
    # pooled embedding should be deterministic
    assert summary[0]["embedding"][0] == contextualized[0, 3, :4].mean()
