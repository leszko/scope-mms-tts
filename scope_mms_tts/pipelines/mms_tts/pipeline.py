"""Text-to-speech pipeline using Hugging Face VitsModel (Meta MMS English by default)."""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer, VitsModel

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import MmsTtsConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

_MAX_CHARS = 500


def _prompts_to_text(prompts: Any) -> str:
    if not prompts:
        return ""
    if isinstance(prompts, list):
        parts: list[str] = []
        for p in prompts:
            if isinstance(p, dict) and p.get("text"):
                parts.append(str(p["text"]).strip())
            elif isinstance(p, str):
                parts.append(p.strip())
        return " ".join(parts).strip()
    if isinstance(prompts, str):
        return prompts.strip()
    return ""


class MmsTtsPipeline(Pipeline):
    @classmethod
    def get_config_class(cls) -> type[BasePipelineConfig]:
        return MmsTtsConfig

    def __init__(
        self,
        model_repo: str = "facebook/mms-tts-eng",
        device: torch.device | None = None,
        **kwargs: Any,
    ):
        _ = kwargs
        self.model_repo = model_repo
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: VitsModel | None = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._last_text: str | None = None

    def prepare(self, **kwargs) -> Requirements | None:
        """Text-only: no video frames required (same as diffusion text mode)."""
        return None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            logger.info(
                "Loading Text-to-Speech model %s on %s", self.model_repo, self.device
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_repo)
            self._model = VitsModel.from_pretrained(self.model_repo)
            self._model.to(self.device)
            self._model.eval()

    def __call__(self, **kwargs) -> dict:
        text = _prompts_to_text(kwargs.get("prompts"))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return {}

        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS]

        if text == self._last_text:
            return {}

        self._ensure_loaded()

        with self._infer_lock:
            assert self._model is not None and self._tokenizer is not None
            inputs = self._tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self._model(**inputs)
            waveform = out.waveform.float()

        audio = waveform.squeeze(0)
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        sr = int(self._model.config.sampling_rate)
        self._last_text = text

        return {
            "audio": audio.unsqueeze(0),
            "audio_sample_rate": sr,
        }
