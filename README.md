# scope-mms-tts

A [Daydream Scope](https://github.com/daydreamlive/scope) plugin that streams **text-to-speech** from your prompts using Hugging Face **VitsModel** (default: [Meta MMS English](https://huggingface.co/facebook/mms-tts-eng)).

## Features

- **Prompt → speech** — type in the prompt box; audio plays over WebRTC.
- **GPU-friendly** — runs on CUDA when available; CPU fallback supported.
- **Configurable model** — optional `model_repo` for other `VitsModel` checkpoints.

## Install

**Development (local tree):**

```bash
cd /path/to/scope-mms-tts
uv pip install -e .
```

Add the project path to Scope’s plugin list so entry points load (default file: `~/.daydream-scope/plugins/plugins.txt`):

```
/path/to/scope-mms-tts
```

From the Scope app you can also use **Settings → Plugins** and point at a **git URL** once this repo is published, same pattern as [scope-audio-beep](https://github.com/leszko/scope-audio-beep).

## Pipeline

- **ID:** `text_to_speech`
- **Name:** Text-to-Speech (MMS VITS)

## License

MIT
