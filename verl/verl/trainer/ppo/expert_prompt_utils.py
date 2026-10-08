"""Runtime construction and packing of MP-OPD expert prompts.

This module consumes only the normalized ``extra_info.expert_contexts`` contract.
Source-dataset fields must be converted before data reaches verl.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.config.algorithm import MPOPDConfig
from verl.utils.model import compute_position_id_with_mask


@dataclass(frozen=True)
class ExpertContext:
    enabled: bool
    instruction: str
    evidence_available: bool
    evidence: object | None = None


@dataclass
class ExpertPromptInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    responses: torch.Tensor
    response_mask: torch.Tensor
    expert_mask: torch.Tensor
    flat_batch_indices: torch.Tensor
    flat_expert_indices: torch.Tensor


def normalize_expert_contexts(
    raw: dict[str, object] | None,
    expert_names: list[str],
    default_instructions: dict[str, str],
) -> tuple[list[ExpertContext], torch.BoolTensor]:
    """Resolve one sample's contexts in the configured expert order."""
    contexts: list[ExpertContext] = []
    active: list[bool] = []
    for name in expert_names:
        value = raw.get(name, {}) if isinstance(raw, dict) else {}
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"expert_contexts.{name} must be an object")
        override = value.get("instruction_override", value.get("instruction"))
        instruction = override.strip() if isinstance(override, str) and override.strip() else default_instructions[name]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"expert instruction for {name} must be non-empty")
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"expert_contexts.{name}.enabled must be boolean")
        evidence_available = value.get("evidence_available", False)
        if not isinstance(evidence_available, bool):
            raise ValueError(f"expert_contexts.{name}.evidence_available must be boolean")
        evidence = value.get("evidence")
        contexts.append(ExpertContext(enabled, instruction.strip(), evidence_available, evidence))
        active.append(enabled)
    return contexts, torch.tensor(active, dtype=torch.bool)


def build_expert_messages(
    raw_prompt: list[dict[str, str]],
    expert_name: str,
    context: ExpertContext,
) -> list[dict[str, str]]:
    """Copy a clean chat and append the direction-specific expert context."""
    if not context.enabled:
        raise ValueError(f"cannot build a disabled expert prompt: {expert_name}")
    messages = copy.deepcopy(raw_prompt)
    evidence = ""
    if context.evidence_available:
        evidence = "\n训练期方向证据：\n" + json.dumps(context.evidence, ensure_ascii=False)
    addition = (
        f"\n\n【MP-OPD {expert_name} 专家视角】\n"
        f"{context.instruction}{evidence}"
    )
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": addition.lstrip()})
    else:
        messages[-1]["content"] = messages[-1].get("content", "") + addition
    return messages


def _raw_prompt_for(batch: DataProto, index: int) -> list[dict[str, str]]:
    raw = batch.non_tensor_batch.get("raw_prompt")
    if raw is None:
        raise ValueError(
            "MP-OPD expert prompt packing requires data.return_raw_chat=true "
            "so the clean chat is available in non_tensor_batch['raw_prompt']"
        )
    value = raw[index]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"raw_prompt[{index}] must be a message list")
    return value


def _tokenize_prompt(tokenizer, messages: list[dict[str, str]]) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids.ndim == 2:
        input_ids = input_ids[0]
    return input_ids.to(dtype=torch.long)


def _left_pad(rows: list[torch.Tensor], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((row.numel() for row in rows), default=0)
    input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for index, row in enumerate(rows):
        input_ids[index, -row.numel() :] = row
        attention_mask[index, -row.numel() :] = 1
    return input_ids, attention_mask


def prepare_expert_prompt_inputs(
    batch: DataProto,
    tokenizer,
    config: MPOPDConfig,
) -> ExpertPromptInputs:
    """Build flattened, left-padded expert prompt+response inputs."""
    responses = batch.batch["responses"]
    response_mask = batch.batch.get(
        "response_mask",
        torch.ones_like(responses, dtype=torch.long),
    )
    batch_size = responses.shape[0]
    expert_mask = torch.zeros((batch_size, len(config.expert_names)), dtype=torch.bool)
    prompt_response_rows: list[torch.Tensor] = []
    flat_batch: list[int] = []
    flat_expert: list[int] = []
    for batch_index in range(batch_size):
        raw_contexts = batch.non_tensor_batch.get("expert_contexts", [None] * batch_size)[batch_index]
        contexts, configured_mask = normalize_expert_contexts(
            raw_contexts,
            config.expert_names,
            config.expert_instructions,
        )
        raw_prompt = _raw_prompt_for(batch, batch_index)
        response = responses[batch_index]
        valid_response = response[response_mask[batch_index].bool()]
        for expert_index, (name, context) in enumerate(zip(config.expert_names, contexts, strict=True)):
            if not configured_mask[expert_index]:
                continue
            prompt_ids = _tokenize_prompt(tokenizer, build_expert_messages(raw_prompt, name, context))
            if prompt_ids.numel() > config.max_expert_prompt_length:
                continue
            expert_mask[batch_index, expert_index] = True
            prompt_response_rows.append(torch.cat((prompt_ids, valid_response.to(dtype=torch.long))))
            flat_batch.append(batch_index)
            flat_expert.append(expert_index)

    input_ids, attention_mask = _left_pad(prompt_response_rows, tokenizer.pad_token_id)
    position_ids = compute_position_id_with_mask(attention_mask)
    return ExpertPromptInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        responses=responses[torch.tensor(flat_batch, dtype=torch.long)] if flat_batch else responses[:0],
        response_mask=response_mask[torch.tensor(flat_batch, dtype=torch.long)] if flat_batch else response_mask[:0],
        expert_mask=expert_mask,
        flat_batch_indices=torch.tensor(flat_batch, dtype=torch.long),
        flat_expert_indices=torch.tensor(flat_expert, dtype=torch.long),
    )


def attach_expert_prompt_tensors(batch: DataProto, packed: ExpertPromptInputs) -> DataProto:
    """Attach only tokenized expert inputs; raw contexts stay outside tensor RPC payloads."""
    batch.batch["expert_input_ids"] = packed.input_ids
    batch.batch["expert_attention_mask"] = packed.attention_mask
    batch.batch["expert_position_ids"] = packed.position_ids
    batch.batch["expert_responses"] = packed.responses
    batch.batch["expert_response_mask"] = packed.response_mask
    batch.batch["expert_mask"] = packed.expert_mask
    batch.batch["flat_expert_batch_indices"] = packed.flat_batch_indices
    batch.batch["flat_expert_indices"] = packed.flat_expert_indices
    return batch
