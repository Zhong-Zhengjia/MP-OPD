"""The versioned, dataset-independent MP-OPD row contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXPERT_NAMES = ("chasing", "long_term", "repurchase", "generalized")
SCHEMA_VERSION = "mpopd.v1"


class SchemaValidationError(ValueError):
    """Raised when a normalized MP-OPD row violates the input contract."""


@dataclass(frozen=True)
class ExpertContext:
    enabled: bool
    evidence_available: bool
    instruction: str | None = None
    instruction_override: str | None = None
    evidence: Any = None


@dataclass(frozen=True)
class MpopdRow:
    data_source: str
    prompt: list[dict[str, str]]
    ability: str
    reward_model: dict[str, Any]
    expert_contexts: dict[str, ExpertContext]
    extra_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        contexts: dict[str, dict[str, Any]] = {}
        for name, context in self.expert_contexts.items():
            # Per §11, direction instructions live in the config by default; a row only
            # carries an instruction when it snapshots one, and ``instruction_override``
            # only when a sample overrides the config default. Omit both when absent so
            # the runtime falls back to the configured expert instructions.
            entry: dict[str, Any] = {
                "enabled": context.enabled,
                "evidence_available": context.evidence_available,
                "evidence": context.evidence,
            }
            if context.instruction is not None:
                entry["instruction"] = context.instruction
            if context.instruction_override is not None:
                entry["instruction_override"] = context.instruction_override
            contexts[name] = entry
        extra_info = dict(self.extra_info)
        extra_info["schema_version"] = SCHEMA_VERSION
        extra_info["expert_contexts"] = contexts
        return {
            "data_source": self.data_source,
            "prompt": self.prompt,
            "ability": self.ability,
            "reward_model": self.reward_model,
            "extra_info": extra_info,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaValidationError(message)


def validate_mpopd_row(row: dict[str, Any]) -> MpopdRow:
    """Validate and normalize one already-adapted row."""
    _require(isinstance(row, dict), "row must be an object")
    _require(isinstance(row.get("data_source"), str) and row["data_source"].strip(), "data_source is required")
    _require(isinstance(row.get("ability"), str) and row["ability"].strip(), "ability is required")

    prompt = row.get("prompt")
    _require(isinstance(prompt, list) and prompt, "prompt must be a non-empty message list")
    for index, message in enumerate(prompt):
        _require(isinstance(message, dict), f"prompt[{index}] must be an object")
        _require(isinstance(message.get("role"), str) and message["role"].strip(), f"prompt[{index}].role is required")
        _require(isinstance(message.get("content"), str), f"prompt[{index}].content must be a string")

    reward_model = row.get("reward_model")
    _require(isinstance(reward_model, dict), "reward_model must be an object")
    extra_info = row.get("extra_info")
    _require(isinstance(extra_info, dict), "extra_info must be an object")
    raw_contexts = extra_info.get("expert_contexts")
    _require(isinstance(raw_contexts, dict), "extra_info.expert_contexts must be an object")
    _require(list(raw_contexts) == list(EXPERT_NAMES), "expert_contexts must use the configured expert order")

    contexts: dict[str, ExpertContext] = {}
    for name in EXPERT_NAMES:
        raw = raw_contexts[name]
        _require(isinstance(raw, dict), f"expert_contexts.{name} must be an object")
        # ``instruction`` is optional (defaults to the config-level direction instruction);
        # ``instruction_override`` optionally replaces that default for one sample. When
        # present, each must be a non-empty string so an empty instruction never silently
        # reaches the runtime.
        instruction = raw.get("instruction")
        if instruction is not None:
            _require(
                isinstance(instruction, str) and instruction.strip(),
                f"expert_contexts.{name}.instruction must be a non-empty string when provided",
            )
        instruction_override = raw.get("instruction_override")
        if instruction_override is not None:
            _require(
                isinstance(instruction_override, str) and instruction_override.strip(),
                f"expert_contexts.{name}.instruction_override must be a non-empty string when provided",
            )
        enabled = raw.get("enabled")
        evidence_available = raw.get("evidence_available")
        _require(isinstance(enabled, bool), f"expert_contexts.{name}.enabled must be boolean")
        _require(
            isinstance(evidence_available, bool),
            f"expert_contexts.{name}.evidence_available must be boolean",
        )
        _require(
            evidence_available is False or "evidence" in raw,
            f"expert_contexts.{name}.evidence is required when evidence_available is true",
        )
        contexts[name] = ExpertContext(
            enabled=enabled,
            evidence_available=evidence_available,
            instruction=instruction.strip() if isinstance(instruction, str) else None,
            instruction_override=(
                instruction_override.strip() if isinstance(instruction_override, str) else None
            ),
            evidence=raw.get("evidence"),
        )

    return MpopdRow(
        data_source=row["data_source"],
        prompt=prompt,
        ability=row["ability"],
        reward_model=reward_model,
        expert_contexts=contexts,
        extra_info={key: value for key, value in extra_info.items() if key != "expert_contexts"},
    )
