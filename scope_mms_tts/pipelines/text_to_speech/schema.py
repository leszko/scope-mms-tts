from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class TextToSpeechConfig(BasePipelineConfig):
    """Meta MMS English VITS — runs on GPU via transformers (no extra TTS packages).

    Default checkpoint ``facebook/mms-tts-eng``; set ``model_repo`` to another Hugging Face
    ``VitsModel`` repo if desired.
    """

    pipeline_id = "text_to_speech"
    pipeline_name = "Text-to-Speech (MMS VITS)"
    pipeline_description = (
        "Converts your prompt text into speech audio streamed to the browser. "
        "Uses Meta MMS English VITS (transformers). Speak by typing in the prompt box."
    )
    docs_url = "https://huggingface.co/facebook/mms-tts-eng"

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="facebook/mms-tts-eng",
            files=[
                "config.json",
                "model.safetensors",
                "tokenizer_config.json",
                "vocab.json",
                "special_tokens_map.json",
            ],
        ),
    ]

    requires_models = True
    estimated_vram_gb = None

    produces_video: ClassVar[bool] = False
    produces_audio: ClassVar[bool] = True

    modes = {"text": ModeDefaults(default=True)}

    model_repo: str = Field(
        default="facebook/mms-tts-eng",
        description="Hugging Face repo id for a VitsModel-compatible TTS checkpoint.",
        json_schema_extra=ui_field_config(
            order=10,
            label="Model repo",
            category="configuration",
            is_load_param=True,
        ),
    )
