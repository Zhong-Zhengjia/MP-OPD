"""Prompt-template loading for source adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

INPUT_PLACEHOLDER = "<在此处填入输入 JSON>"
EVIDENCE_PLACEHOLDER = "{{expert_evidence}}"


def load_clean_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if INPUT_PLACEHOLDER not in text:
        raise ValueError(f"clean template must contain {INPUT_PLACEHOLDER!r}: {path}")
    return text


def load_expert_templates(directory: Path, expert_names: tuple[str, ...]) -> dict[str, str]:
    templates = {}
    for name in expert_names:
        path = directory / f"mpopd_{name}_prompt.txt"
        text = path.read_text(encoding="utf-8")
        if EVIDENCE_PLACEHOLDER not in text:
            raise ValueError(f"expert template must contain {EVIDENCE_PLACEHOLDER!r}: {path}")
        templates[name] = text
    return templates


def render_clean_prompt(template: str, profile: dict[str, Any]) -> str:
    return template.replace(
        INPUT_PLACEHOLDER,
        __import__("json").dumps(
            {"user_profile_and_history": profile},
            ensure_ascii=False,
            indent=2,
        ),
        1,
    )


def extract_instruction(template: str) -> str:
    instruction = template.split(EVIDENCE_PLACEHOLDER, 1)[0].strip()
    if not instruction:
        raise ValueError("expert template has an empty instruction")
    return instruction
