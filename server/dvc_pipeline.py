"""Streaming DVC feature extraction utilities for the GPU server."""
from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional

np = importlib.import_module("numpy")
_pil_image_module = importlib.import_module("PIL.Image")
Image = getattr(_pil_image_module, "Image")

from gce_streaming.server import segment_utils

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency at development time
    _default_config_module = importlib.import_module(
        "scenic.projects.streaming_dvc.configs.git_anet_streaming_input_output_local"
    )
except ImportError:  # pragma: no cover - handled lazily during runtime on GPU VM
    _default_config_module = None


@dataclass
class FeaturePayload:
    visual_features: Any
    raw_streaming_feature: Any
    importance_scores: Any
    token_indices: Any
    frame_indices: Any
    contextualized_features: Any | None = None


class DvcFeatureExtractor:
    """Wraps Streaming DVC feature extraction for streaming workloads."""

    def __init__(
        self,
        config_module: str = "scenic.projects.streaming_dvc.configs.git_anet_streaming_input_output_local",
        checkpoint_path: Optional[str] = None,
        batch_size: int = 1,
        frame_batch_size: int = 32,
        vertex_bert_endpoint: Optional[str] = None,
        vertex_project: Optional[str] = None,
        vertex_location: str = "us-central1",
        vertex_batch_size: int = 4,
        segment_top_k: int = 4,
        memory_summary_dim: int = 32,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-1.5-pro",
        llm_temperature: float = 0.2,
        llm_vertex_project: Optional[str] = None,
        llm_vertex_location: str = "us-central1",
        llm_prompt_template: Optional[str] = None,
    ) -> None:
        self.config_module = config_module
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.frame_batch_size = frame_batch_size
        self.vertex_bert_endpoint = vertex_bert_endpoint
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        self.vertex_batch_size = vertex_batch_size
        self.segment_top_k = segment_top_k
        self.memory_summary_dim = memory_summary_dim
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_vertex_project = llm_vertex_project
        self.llm_vertex_location = llm_vertex_location
        self.llm_prompt_template = llm_prompt_template

        self._load_config()
        self._build_model()
        self._vertex_client = None

    def _load_config(self) -> None:
        if (
            _default_config_module is not None
            and self.config_module == _default_config_module.__name__
        ):
            self.cfg = _default_config_module.get_config()
        else:
            module = __import__(self.config_module, fromlist=["get_config"])
            self.cfg = module.get_config()
        self.model_cfg = dict(self.cfg.model)
        self.dataset_cfg = self.cfg.dataset
        self.num_frames = self.model_cfg.get("num_frames", self.dataset_cfg.num_frames)
        self.image_size = self.dataset_cfg.image_size
        logger.info(
            "Loaded Streaming DVC config %s (num_frames=%d, image_size=%s)",
            self.config_module,
            self.num_frames,
            self.image_size,
        )

    def _build_model(self) -> None:
        jax_module = importlib.import_module("jax")
        jnp_module = importlib.import_module("jax.numpy")
        streaming_module = importlib.import_module(
            "scenic.projects.streaming_dvc.modeling.streaming_model"
        )

        model_cls = getattr(streaming_module, "StreamingCaptioningFlaxModel")
        self.model = model_cls(**self.model_cfg)
        self._jax = jax_module
        self._jnp = jnp_module

        rng = self._jax.random.PRNGKey(0)
        dummy = self._jnp.zeros(
            (self.batch_size, self.num_frames, self.image_size, self.image_size, 3),
            dtype=self._jnp.float32,
        )
        logger.info("Initializing Streaming DVC parameters (this may take a while)...")
        params = self.model.init(rng, dummy, preprocess=True, train=False)

        if self.checkpoint_path:
            try:
                train_utils = importlib.import_module("scenic.train_lib_deprecated.train_utils")

                params = train_utils.restore_checkpoint(self.checkpoint_path, target=params)
                logger.info("Loaded checkpoint from %s", self.checkpoint_path)
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("Failed to load checkpoint from %s: %s", self.checkpoint_path, exc)
        self.params = params

    @staticmethod
    def _resize_and_normalize(frame: Any, target_size: int) -> Any:
        image = Image.fromarray(frame.astype(np.uint8))
        image = image.resize((target_size, target_size))
        array = np.asarray(image).astype(np.float32) / 255.0
        return array

    def _prepare_batch(self, frames: List[Any]) -> Any:
        if len(frames) < self.num_frames:
            last = frames[-1]
            frames = frames + [last] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            stride = len(frames) / self.num_frames
            frames = [frames[int(i * stride)] for i in range(self.num_frames)]
        processed = [self._resize_and_normalize(frame, self.image_size) for frame in frames[: self.num_frames]]
        batch = np.stack(processed, axis=0)
        batch = np.expand_dims(batch, axis=0)
        return batch

    def _get_vertex_client(self):
        if not self.vertex_bert_endpoint:
            return None
        if self._vertex_client is None:
            try:
                vertex_module = importlib.import_module(
                    "scenic.projects.streaming_dvc.modeling.vertex_ai_client"
                )
                client_cls = getattr(vertex_module, "VertexAIBertClient")
            except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dep
                logger.warning("Vertex AI client unavailable: %s", exc)
                return None
            if not self.vertex_project:
                logger.warning("vertex_project is not set; skipping Vertex BERT calls")
                return None
            self._vertex_client = client_cls(
                endpoint_id=self.vertex_bert_endpoint,
                project=self.vertex_project,
                location=self.vertex_location,
            )
        return self._vertex_client

    async def _contextualize_with_vertex(self, visual_features: Any) -> Any | None:
        client = self._get_vertex_client()
        if client is None:
            return None

        def _call_vertex():
            return client.encode_features(
                np.array(visual_features),
                batch_size=self.vertex_batch_size,
            )

        loop = asyncio.get_event_loop()
        try:
            contextualized = await loop.run_in_executor(None, _call_vertex)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Vertex BERT call failed: %s", exc)
            return None
        return contextualized

    async def extract_features(self, frames: List[Any]) -> FeaturePayload:
        batch = await asyncio.get_event_loop().run_in_executor(None, self._prepare_batch, frames)
        outputs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.apply(
                self.params,
                self._jnp.asarray(batch),
                preprocess=True,
                train=False,
                return_features_only=True,
            ),
        )
        visual_features = np.array(outputs["visual_features"])
        raw_streaming_feature = np.array(outputs["raw_streaming_feature"])
        importance_scores = raw_streaming_feature.mean(axis=-1)
        token_indices = np.argsort(importance_scores[0])[::-1]
        max_frame = max(len(frames) - 1, 0)
        frame_indices = np.linspace(0, max_frame, num=len(token_indices), dtype=int)
        contextualized_features = await self._contextualize_with_vertex(visual_features)
        return FeaturePayload(
            visual_features=visual_features,
            raw_streaming_feature=raw_streaming_feature,
            importance_scores=importance_scores,
            token_indices=token_indices,
            frame_indices=frame_indices,
            contextualized_features=contextualized_features,
        )

    def _build_segments(
        self,
        payload: FeaturePayload,
        frames: List[Any],
        top_k: int,
    ) -> List[dict[str, Any]]:
        return segment_utils.select_top_segments(
            importance_scores=payload.importance_scores,
            token_indices=payload.token_indices,
            frame_indices=payload.frame_indices,
            frames=frames,
            top_k=top_k,
        )

    def _build_memory_summary(
        self,
        payload: FeaturePayload,
        top_k: int,
    ) -> List[dict[str, Any]]:
        return segment_utils.build_memory_summary(
            payload.contextualized_features,
            payload.token_indices,
            payload.importance_scores,
            top_k=top_k,
            pool_dim=self.memory_summary_dim,
        )

    async def infer_and_caption(
        self,
        frames: List[Any],
        use_external_llm: bool = True,
        api_key: Optional[str] = None,
        top_k: Optional[int] = None,
        llm_provider: Optional[str] = None,
    ) -> Any:
        """End-to-end: extract features for given frames and (optionally)
        call an external LLM decoder to produce captions.

        Returns the raw caption result from the external decoder or the
        FeaturePayload when `use_external_llm` is False.
        """
        payload = await self.extract_features(frames)

        if not use_external_llm:
            return payload

        provider = (llm_provider or self.llm_provider or "gemini").lower()
        if provider == "gemini" and not api_key:
            raise ValueError("api_key is required for Gemini provider when use_external_llm is True")

        # Lazy import the adapter to avoid importing Gemini libs when not used.
        from gce_streaming.server.gemini_client import ExternalLLMDecoderHead

        decoder = ExternalLLMDecoderHead(
            api_key=api_key or "",
            model=self.llm_model,
            temperature=self.llm_temperature,
            memory_summary_dim=self.memory_summary_dim,
            provider=provider,
            vertex_project=self.llm_vertex_project,
            vertex_location=self.llm_vertex_location,
            vertex_model=self.llm_model,
            prompt_template=self.llm_prompt_template,
        )
        segment_budget = int(top_k) if top_k is not None else int(self.segment_top_k)
        segments = self._build_segments(payload, frames, segment_budget)
        memory_summary = self._build_memory_summary(payload, segment_budget)
        captions = await decoder.generate_from_segments(
            segments=segments,
            memory_summary=memory_summary,
        )
        return captions


@lru_cache(maxsize=1)
def get_feature_extractor(
    config_module: str,
    checkpoint_path: Optional[str],
    batch_size: int,
    frame_batch_size: int,
    vertex_bert_endpoint: Optional[str],
    vertex_project: Optional[str],
    vertex_location: str,
    vertex_batch_size: int,
    segment_top_k: int,
    memory_summary_dim: int,
) -> DvcFeatureExtractor:
    return DvcFeatureExtractor(
        config_module=config_module,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        batch_size=batch_size,
        frame_batch_size=frame_batch_size,
        vertex_bert_endpoint=vertex_bert_endpoint,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_batch_size=vertex_batch_size,
        segment_top_k=segment_top_k,
        memory_summary_dim=memory_summary_dim,
    )
