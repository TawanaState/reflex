"""
Configuration manager for Project Reflex.
Loads strongly typed settings from environment variables and .env file.
"""

import os
from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Reflex runtime and server settings."""

    # Model and Adapter paths
    DIFFUSION_GEMMA_PATH: str = Field(
        default="google/diffusiongemma-26B-A4B-it",
        description="Path to local model directory or Hugging Face repo ID",
    )
    REFLEX_LORA_PATH: str = Field(
        default="models/reflex_lora_v1",
        description="Path to fine-tuned Reflex LoRA adapter",
    )

    # Runtime mode: native tool calling with optional atomic draft exit, or fixed-step baseline
    REFLEX_MODE: Literal["reflex", "vanilla"] = Field(
        default="reflex",
        description="Runtime mode ('reflex' or 'vanilla')",
    )

    # Native draft early exit. Stable draft count is a heuristic until calibrated
    # on an independent held-out set; confidence is therefore reported as 0.
    REFLEX_ATOMIC_EARLY_EXIT: bool = Field(default=True)
    REFLEX_ATOMIC_STABLE_STEPS: int = Field(default=1, ge=1)
    # Experimental: only fully specified typed-scalar calls are eligible.
    REFLEX_SCALAR_EARLY_EXIT: bool = Field(default=False)
    REFLEX_SCALAR_STABLE_STEPS: int = Field(default=2, ge=1)

    # Server binding
    HOST: str = Field(
        default="0.0.0.0",
        description="Host bind address for FastAPI server",
    )
    PORT: int = Field(
        default=8090,
        description="Port for FastAPI server",
    )

    # Risk gate calibration
    CONFORMAL_EPS: float = Field(
        default=0.02,
        description="Upper bound on selective error rate for fast-path exits (e.g. 0.02 = 2%)",
    )

    # Hardware device
    DEVICE: str = Field(
        default="cuda:0",
        description="Target PyTorch execution device ('cuda:0', 'cuda', or 'cpu')",
    )

    # Generative expansion settings
    MAX_CANVAS_LENGTH: int = Field(
        default=256,
        description="Maximum canvas buffer length for generative expansion",
    )
    GENERATIVE_STEPS: int = Field(
        default=20,
        description="Number of reverse diffusion denoising steps for Phase 2 expansion or vanilla mode",
    )

    # OpenAI-compatible API model naming
    MODEL_ID: str = Field(
        default="reflex-diffusiongemma",
        description="Identifier returned in OpenAI /v1/models endpoint",
    )

    # Optional HF Token
    HF_TOKEN: Optional[str] = Field(
        default=None,
        description="Hugging Face authentication token",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("REFLEX_MODE", mode="before")
    @classmethod
    def normalize_mode(cls, v: str) -> str:
        return v.lower().strip() if isinstance(v, str) else v


@lru_cache()
def get_settings() -> Settings:
    """Returns singleton cached instance of application settings."""
    return Settings()

