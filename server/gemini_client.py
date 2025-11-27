"""Gemini API wrapper for converting Streaming DVC features to captions."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional during static analysis
    from google import generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "google-generativeai is required to contact Gemini. Install google-generativeai>=0.7.2."
    ) from exc

from . import segment_utils


def _build_caption_prompt(
    segments: List[Dict[str, Any]],
    memory_summary: Optional[List[Dict[str, Any]]] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    """Shared prompt template for Gemini / Vertex LLM providers."""
    narrative = [
        "You are assisting a streaming dense video captioning pipeline.",
        "Each segment comes from an importance-ranked memory token produced by a vision model.",
        "Produce concise Japanese captions (<= 25 tokens) for each segment and keep chronological order.",
        "Return output as JSON with fields: segment_index, caption, confidence (0-1).",
        "Use double quotes for keys and values and avoid trailing comments.",
    ]
    if extra_instructions:
        narrative.append(extra_instructions.strip())
    narrative.append("Here are the segments:")
    payload = json.dumps(segments, ensure_ascii=False, indent=2)
    narrative.append(payload)
    if memory_summary:
        narrative.append("Vertex BERT memory summary embeddings:")
        narrative.append(json.dumps(memory_summary, ensure_ascii=False, indent=2))
    narrative.append("Respond with a valid JSON array only.")
    return "\n".join(narrative)


class GeminiCaptioner:
    """Helper class to communicate with Gemini multimodal endpoints."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.2,
        prompt_template: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY in the environment.")
        self.model_name = model
        self.temperature = temperature
        self._prompt_template = prompt_template
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    async def generate_captions(
        self,
        timeline_segments: List[Dict[str, Any]],
        memory_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        prompt = _build_caption_prompt(
            timeline_segments,
            memory_summary,
            extra_instructions=self._prompt_template,
        )
        logger.debug("Sending prompt to Gemini: %s", prompt[:4000])
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(
                prompt,
                request_options={"temperature": self.temperature},
            ),
        )
        text = response.text if hasattr(response, "text") else str(response)
        return {
            "model": self.model_name,
            "raw_response": text,
            "segments": timeline_segments,
            "memory_summary": memory_summary or [],
        }


class VertexAIGenerativeCaptioner:
    """Captioner that calls Vertex AI Generative Models (Gemini / text-bison)."""

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model: str = "gemini-1.5-pro",
        temperature: float = 0.2,
        prompt_template: Optional[str] = None,
    ) -> None:
        if not project:
            raise ValueError("Vertex project is required when provider=vertex.")
        try:
            import vertexai  # type: ignore
            from vertexai.preview.generative_models import GenerativeModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "google-cloud-aiplatform is required for provider=vertex. "
                "Install google-cloud-aiplatform>=1.70.0."
            ) from exc
        vertexai.init(project=project, location=location)
        self._model = GenerativeModel(model)
        self.model_name = model
        self.temperature = temperature
        self._generation_config = {"temperature": self.temperature}
        self._prompt_template = prompt_template

    async def generate_captions(
        self,
        timeline_segments: List[Dict[str, Any]],
        memory_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        prompt = _build_caption_prompt(
            timeline_segments,
            memory_summary,
            extra_instructions=self._prompt_template,
        )
        logger.debug("Sending prompt to Vertex AI (%s)", self.model_name)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._model.generate_content(
                prompt,
                generation_config=self._generation_config,
            ),
        )
        text = response.text if hasattr(response, "text") else str(response)
        return {
            "model": self.model_name,
            "raw_response": text,
            "segments": timeline_segments,
            "memory_summary": memory_summary or [],
        }


class ExternalLLMDecoderHead:
    """Adapter providing an async generate() interface that accepts feature
    payloads (or representative frames) and returns structured captions.

    This class keeps the networking out of the model code and provides a
    minimal, testable contract for experiment code to call an external LLM
    decoder.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.2,
        memory_summary_dim: int = 32,
        provider: str = "gemini",
        vertex_project: Optional[str] = None,
        vertex_location: str = "us-central1",
        vertex_model: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ):
        provider_key = (provider or "gemini").lower()
        if provider_key in {"vertex", "vertex-gemini", "vertexai"}:
            self._captioner = VertexAIGenerativeCaptioner(
                project=vertex_project or "",
                location=vertex_location,
                model=vertex_model or model,
                temperature=temperature,
                prompt_template=prompt_template,
            )
        else:
            self._captioner = GeminiCaptioner(
                api_key=api_key,
                model=model,
                temperature=temperature,
                prompt_template=prompt_template,
            )
        self._memory_summary_dim = memory_summary_dim

    async def generate_from_feature_payload(
        self,
        feature_payload: Dict[str, Any],
        frames: Optional[List[Any]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Build timeline segments from a feature payload and call Gemini.

        Args:
          feature_payload: mapping with keys 'importance_scores', 'token_indices', 'frame_indices'
          frames: optional list of original frames to attach to segments
          top_k: how many segments to request (defaults to 3)

        Returns:
          dict with keys ['model','raw_response','segments','captions']
        """
        # Build a minimal segments list: choose top_k tokens by importance
        importance = feature_payload.get("importance_scores")
        token_indices = feature_payload.get("token_indices")
        frame_indices = feature_payload.get("frame_indices")

        if importance is None or token_indices is None or frame_indices is None:
            raise ValueError("feature_payload missing required fields")

        segments = segment_utils.select_top_segments(
            importance_scores=importance,
            token_indices=token_indices,
            frame_indices=frame_indices,
            frames=frames,
            top_k=top_k,
        )
        contextualized = feature_payload.get("contextualized_features")
        memory_summary = segment_utils.build_memory_summary(
            contextualized,
            token_indices,
            importance,
            top_k=top_k,
            pool_dim=self._memory_summary_dim,
        )
        return await self.generate_from_segments(segments, memory_summary)

    async def generate_from_segments(
        self,
        segments: List[Dict[str, Any]],
        memory_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send precomputed segments (and optional memory summary) to Gemini."""
        result = await self._captioner.generate_captions(segments, memory_summary)
        raw_text = result.get("raw_response", "")
        parsed = self._parse_response(raw_text)
        result["captions"] = parsed
        return result

    @staticmethod
    def _parse_response(raw_text: str) -> Any:
        try:
            return json.loads(raw_text)
        except Exception:
            return raw_text


def build_llm_client(llm_settings: Any) -> Any:
    """Factory used by FastAPI server to lazily create an LLM client instance."""
    provider = (getattr(llm_settings, "provider", "") or "gemini").lower()
    model = getattr(llm_settings, "model", "gemini-1.5-pro")
    temperature = float(getattr(llm_settings, "temperature", 0.2))
    prompt_override = None
    prompt_file = getattr(llm_settings, "prompt_file", None)
    if prompt_file:
        try:
            prompt_override = Path(prompt_file).read_text(encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"Failed to read prompt file {prompt_file}: {exc}") from exc

    if provider in {"vertex", "vertex-gemini", "vertexai"}:
        project = getattr(llm_settings, "vertex_project", None)
        location = getattr(llm_settings, "vertex_location", "us-central1")
        if not project:
            raise ValueError("vertex_project must be set when provider=vertex.")
        return VertexAIGenerativeCaptioner(
            project=project,
            location=location or "us-central1",
            model=model,
            temperature=temperature,
            prompt_template=prompt_override,
        )

    api_key = getattr(llm_settings, "api_key", "")
    if not api_key:
        raise ValueError("LLM API key is missing; set LLM_API_KEY or GEMINI_API_KEY.")
    return GeminiCaptioner(
        api_key=api_key,
        model=model,
        temperature=temperature,
        prompt_template=prompt_override,
    )
