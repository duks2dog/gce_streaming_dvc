"""Configuration helpers for the streaming DVC server."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """Runtime configuration for the FastAPI server."""

    host: str = Field(default=os.environ.get("STREAMING_DVC_SERVER_HOST", "0.0.0.0"))
    port: int = Field(default=int(os.environ.get("STREAMING_DVC_SERVER_PORT", 8000)))
    log_level: str = Field(default=os.environ.get("STREAMING_DVC_LOG_LEVEL", "info"))
    allow_origins: list[str] = Field(
        default_factory=lambda: os.environ.get("STREAMING_DVC_ALLOW_ORIGINS", "*").split(",")
    )


class DvcSettings(BaseModel):
    """Configuration for loading the Streaming DVC model and assets."""

    config_module: str = Field(
        default=os.environ.get(
            "STREAMING_DVC_CONFIG_MODULE",
            "scenic.projects.streaming_dvc.configs.git_anet_streaming_input_output_local",
        )
    )
    checkpoint_path: Optional[Path] = Field(
        default=None,
        description="Path to the pretrained checkpoint directory. Can be omitted for shape checks.",
    )
    stream_root: Path = Field(
        default=Path(os.environ.get("STREAMING_DVC_ROOT", "/content/StreamingDVC"))
    )
    batch_size: int = Field(default=int(os.environ.get("STREAMING_DVC_BATCH_SIZE", 1)))
    frame_batch_size: int = Field(
        default=int(os.environ.get("STREAMING_DVC_FRAME_BATCH_SIZE", 32)),
        description="Number of frames to accumulate before running the feature extractor.",
    )
    vertex_bert_endpoint: Optional[str] = Field(
        default=os.environ.get("VERTEX_BERT_ENDPOINT_ID"),
        description="Vertex AI Endpoint resource name for the external BERT encoder.",
    )
    vertex_project: Optional[str] = Field(
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        description="Project ID used for Vertex AI calls.",
    )
    vertex_location: str = Field(
        default=os.environ.get("VERTEX_LOCATION", "us-central1"),
        description="Region for Vertex AI endpoints.",
    )
    vertex_batch_size: int = Field(
        default=int(os.environ.get("VERTEX_BERT_BATCH_SIZE", 4)),
        description="Batch size for Vertex AI BERT predictions.",
    )
    segment_top_k: int = Field(
        default=int(os.environ.get("STREAMING_DVC_SEGMENT_TOP_K", 4)),
        description="Number of memory tokens to expose to the LLM per batch.",
    )
    memory_summary_dim: int = Field(
        default=int(os.environ.get("STREAMING_DVC_MEMORY_SUMMARY_DIM", 32)),
        description="Dimension of pooled memory embeddings passed to the LLM.",
    )


class LLMSettings(BaseModel):
    """Configuration bindings for external LLM providers."""

    provider: str = Field(default=os.environ.get("LLM_PROVIDER", "gemini"))
    api_key: str = Field(
        default=os.environ.get("LLM_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    )
    model: str = Field(
        default=os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"))
    )
    temperature: float = Field(
        default=float(os.environ.get("LLM_TEMPERATURE", os.environ.get("GEMINI_TEMPERATURE", 0.2)))
    )
    prompt_file: Optional[Path] = Field(
        default=Path(os.environ["LLM_PROMPT_FILE"]) if os.environ.get("LLM_PROMPT_FILE") else None
    )
    vertex_project: Optional[str] = Field(
        default=os.environ.get("VERTEX_LLM_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT"))
    )
    vertex_location: str = Field(
        default=os.environ.get("VERTEX_LLM_LOCATION", os.environ.get("VERTEX_LOCATION", "us-central1"))
    )


class AppSettings(BaseModel):
    """Aggregated application settings."""

    server: ServerSettings = Field(default_factory=ServerSettings)
    dvc: DvcSettings = Field(default_factory=DvcSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


settings = AppSettings()
