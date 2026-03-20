"""Lightweight tests (no model download)."""

import torch

from scope_mms_tts.pipelines.mms_tts.pipeline import (
    MmsTtsPipeline,
    _prompts_to_text,
)


def test_prompts_to_text():
    assert _prompts_to_text([{"text": "  hello ", "weight": 1}]) == "hello"
    assert (
        _prompts_to_text([{"text": "a", "weight": 1}, {"text": "b", "weight": 1}])
        == "a b"
    )
    assert _prompts_to_text(None) == ""


def test_pipeline_prepare_returns_none():
    p = MmsTtsPipeline(device=torch.device("cpu"))
    assert p.prepare() is None
