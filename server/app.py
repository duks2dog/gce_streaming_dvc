"""FastAPI server that exposes Streaming DVC + Gemini as a streaming caption service."""
from __future__ import annotations

import asyncio
import base64
import importlib
import json
import logging
from functools import lru_cache
from typing import Any, Dict, List

np = importlib.import_module("numpy")
cv2 = importlib.import_module("cv2")

FastAPI = getattr(importlib.import_module("fastapi"), "FastAPI")
WebSocket = getattr(importlib.import_module("fastapi"), "WebSocket")
WebSocketDisconnect = getattr(importlib.import_module("fastapi"), "WebSocketDisconnect")
CORSMiddleware = getattr(importlib.import_module("fastapi.middleware.cors"), "CORSMiddleware")
WebSocketState = getattr(importlib.import_module("starlette.websockets"), "WebSocketState")

from .config import settings
from .dvc_pipeline import get_feature_extractor
from .gemini_client import build_llm_client
from .segment_utils import select_top_segments, build_memory_summary

logger = logging.getLogger("streaming_dvc_server")
logging.basicConfig(level=getattr(logging, settings.server.log_level.upper(), logging.INFO))

app = FastAPI(title="Streaming DVC + Gemini Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _get_llm_client():
    return build_llm_client(settings.llm)


async def _process_frames(frames: List[Any]) -> Dict[str, Any]:
    extractor = get_feature_extractor(
        config_module=settings.dvc.config_module,
        checkpoint_path=settings.dvc.checkpoint_path,
        batch_size=settings.dvc.batch_size,
        frame_batch_size=settings.dvc.frame_batch_size,
        vertex_bert_endpoint=settings.dvc.vertex_bert_endpoint,
        vertex_project=settings.dvc.vertex_project,
        vertex_location=settings.dvc.vertex_location,
        vertex_batch_size=settings.dvc.vertex_batch_size,
        segment_top_k=settings.dvc.segment_top_k,
        memory_summary_dim=settings.dvc.memory_summary_dim,
    )
    features = await extractor.extract_features(frames)
    top_segments = select_top_segments(
        features.importance_scores,
        features.token_indices,
        features.frame_indices,
        frames,
        top_k=settings.dvc.segment_top_k,
    )
    memory_summary = build_memory_summary(
        features.contextualized_features,
        features.token_indices,
        features.importance_scores,
        top_k=settings.dvc.segment_top_k,
        pool_dim=settings.dvc.memory_summary_dim,
    )
    llm_client = _get_llm_client()
    captions = await llm_client.generate_captions(top_segments, memory_summary=memory_summary)
    captions["memory_summary"] = memory_summary
    return {"type": "caption", "data": captions}


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: Any) -> None:
    await websocket.accept()
    frames: List[Any] = []
    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)
            msg_type = payload.get("type")

            if msg_type == "frame":
                frame = _decode_frame(payload)
                frames.append(frame)
                await websocket.send_json({"type": "ack", "frame_count": len(frames)})
                if len(frames) >= settings.dvc.frame_batch_size:
                    response = await _process_frames(frames)
                    frames.clear()
                    await websocket.send_json(response)
            elif msg_type in {"flush", "end"}:
                if frames:
                    response = await _process_frames(frames)
                    frames.clear()
                    await websocket.send_json(response)
                if msg_type == "end":
                    break
            else:
                await websocket.send_json({"type": "error", "detail": f"Unknown message type: {msg_type}"})
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:  # pragma: no cover - runtime error handling
        logger.exception("Processing error: %s", exc)
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()


def _decode_frame(payload: Dict[str, Any]) -> Any:
    image_b64 = payload.get("image_base64")
    if not image_b64:
        raise ValueError("Payload missing image_base64")
    raw = base64.b64decode(image_b64)
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode frame")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":  # pragma: no cover
    uvicorn = importlib.import_module("uvicorn")

    uvicorn.run(
        "gce_streaming.server.app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
        log_level=settings.server.log_level,
    )
