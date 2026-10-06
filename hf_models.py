"""Resolve public model names to Hugging Face checkpoint ids."""

from __future__ import annotations

DEEPSEEK_R1 = "Deepseek-R1"

# Weights published under this repo id; use DEEPSEEK_R1 in scripts and docs.
_DEEPSEEK_R1_CHECKPOINT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

_ALIASES = {DEEPSEEK_R1: _DEEPSEEK_R1_CHECKPOINT}


def resolve_hf_model(name: str) -> str:
    return _ALIASES.get(name, name)
