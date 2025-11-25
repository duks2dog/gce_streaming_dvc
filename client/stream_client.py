"""CLI client that streams video frames to the Streaming DVC server."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import AsyncIterator, Optional

import click
import cv2
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streaming_dvc_client")


async def _frame_iterator(video_source: str, target_fps: Optional[float]) -> AsyncIterator[bytes]:
    if video_source == "camera":
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    nominal_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    delay = 1.0 / (target_fps or nominal_fps)

    try:
        while True:
            start = time.perf_counter()
            success, frame = capture.read()
            if not success:
                break
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning("Failed to encode frame, skipping")
                continue
            yield encoded.tobytes()
            elapsed = time.perf_counter() - start
            await asyncio.sleep(max(0.0, delay - elapsed))
    finally:
        capture.release()


async def stream_video(
    server_url: str, video_source: str, target_fps: Optional[float], only_gemini: bool
) -> None:
    async with websockets.connect(server_url) as websocket:
        logger.info("Connected to %s", server_url)
        frame_index = 0
        async for frame_bytes in _frame_iterator(video_source, target_fps):
            message = {
                "type": "frame",
                "frame_index": frame_index,
                "timestamp": time.time(),
                "image_base64": base64.b64encode(frame_bytes).decode("utf-8"),
            }
            await websocket.send(json.dumps(message))
            frame_index += 1

            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                if not only_gemini:
                    logger.info("Server: %s", response)
            except asyncio.TimeoutError:
                pass

        await websocket.send(json.dumps({"type": "end"}))
        logger.info("Sent end signal, awaiting captions...")

        async for response in websocket:
            if (not only_gemini) or "caption" in response:
                logger.info("Server: %s", response)


@click.command()
@click.option("--server", default="ws://localhost:8000/ws/stream", help="WebSocket endpoint of the server")
@click.option("--video", default="camera", help="Path to a video file or 'camera' for webcam")
@click.option("--fps", type=float, default=None, help="Target streaming FPS (defaults to source FPS)")
@click.option("--loop", is_flag=True, help="Loop the video until interrupted")
@click.option("--gemini", is_flag=True, help="Print Gemini responses only")
def main(server: str, video: str, fps: Optional[float], loop: bool, gemini: bool) -> None:
    """Start streaming video to the remote Streaming DVC server."""

    async def runner() -> None:
        try:
            await stream_video(server, video, fps, gemini)
        except Exception as exc:  # pragma: no cover
            logger.error("Streaming error: %s", exc)

    if loop:
        while True:
            asyncio.run(runner())
            if video == "camera":
                break
    else:
        asyncio.run(runner())


if __name__ == "__main__":  # pragma: no cover
    main()
