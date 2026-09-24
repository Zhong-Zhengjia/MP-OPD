# Copyright 2026 MP-OPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build isolated prompt-conditioned expert inputs for MP-OPD."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from verl import DataProto
from verl.trainer.config.algorithm import MPOPDConfig
from verl.utils.model import compute_position_id_with_mask

FOCUS_MARKER = "<|mpopd_focus|>"
EVIDENCE_MARKER = "<|mpopd_evidence|>"

_ATTACHED_TENSOR_KEYS = (
    "expert_input_ids",
    "expert_attention_mask",
    "expert_position_ids",
    "expert_responses",
    "expert_response_mask",
)


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
    """Resolve one sample's expert settings in configuration order."""
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError(f"expert_contexts must be a mapping or None, got {type(raw).__name__}")

    contexts = []
    active = []
    raw = raw or {}
    for expert_name in expert_names:
        entry = raw.get(expert_name, {})
        if entry is None:
            entry = {}
        if not isinstance(entry, Mapping):
            raise TypeError(f"expert context for {expert_name!r} must be a mapping, got {type(entry).__name__}")

        default_instruction = default_instructions.get(expert_name)
        override = entry.get("instruction_override")
        instruction = override.strip() if isinstance(override, str) and override.strip() else default_instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"expert {expert_name!r} requires a non-empty instruction")

        enabled = bool(entry.get("enabled", True))
        context = ExpertContext(
            enabled=enabled,
            instruction=instruction.strip(),
            evidence_available=bool(entry.get("evidence_available", False)),
            evidence=entry.get("evidence"),
        )
        contexts.append(context)
        active.append(enabled)

    return contexts, torch.tensor(active, dtype=torch.bool)


def _format_evidence(evidence: object) -> str:
    if isinstance(evidence, str):
        return evidence
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)


def build_expert_messages(
    raw_prompt: list[dict[str, str]],
    expert_name: str,
    context: ExpertContext,
) -> list[dict[str, str]]:
    """Return a copied chat with one expert-only instruction message appended."""
    messages = copy.deepcopy(list(raw_prompt))
    expert_content = f"{FOCUS_MARKER} {expert_name}\n{context.instruction}"
    if context.evidence_available:
        expert_content += f"\n{EVIDENCE_MARKER}\n{_format_evidence(context.evidence)}"
    messages.append({"role": "user", "content": expert_content})
    return messages


def _tokenize_expert_prompt(messages: list[dict[str, str]], tokenizer) -> torch.Tensor:
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokenized = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError(f"tokenizer returned {input_ids.shape[0]} rows for one expert prompt")
        input_ids = input_ids[0]
    if input_ids.ndim != 1:
        raise ValueError(f"expert prompt input_ids must be one-dimensional, got shape {tuple(input_ids.shape)}")
    return input_ids


def _get_non_tensor_rows(batch: DataProto, key: str, default: object) -> list[object]:
    if key not in batch.non_tensor_batch:
        return [default for _ in range(len(batch))]
    values = batch.non_tensor_batch[key]
    if len(values) != len(batch):
        raise ValueError(f"{key} length {len(values)} does not match batch size {len(batch)}")
    return list(values)


def prepare_expert_prompt_inputs(
    batch: DataProto,
    tokenizer,
    config: MPOPDConfig,
) -> ExpertPromptInputs:
    """Build and flatten valid specialized expert prompts without mutating ``batch``."""
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError("raw_prompt is required; set data.return_raw_chat=True for MP-OPD")
    if "responses" not in batch.batch:
        raise ValueError("responses are required to build teacher-forced expert inputs")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id is required for expert prompt left padding")

    raw_prompts = _get_non_tensor_rows(batch, "raw_prompt", None)
    raw_contexts = _get_non_tensor_rows(batch, "expert_contexts", None)
    responses = batch.batch["responses"]
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        response_mask = torch.ones_like(responses, dtype=torch.long)

    batch_size = len(batch)
    num_experts = len(config.expert_names)
    expert_mask = torch.zeros((batch_size, num_experts), dtype=torch.bool, device=responses.device)
    prompt_ids_rows: list[torch.Tensor] = []
    flat_batch_indices: list[int] = []
    flat_expert_indices: list[int] = []

    for batch_idx, (raw_prompt, raw_context) in enumerate(zip(raw_prompts, raw_contexts, strict=True)):
        if not isinstance(raw_prompt, (list, np.ndarray)):
            raise TypeError(f"raw_prompt must be a list or numpy array, got {type(raw_prompt).__name__}")
        contexts, configured_mask = normalize_expert_contexts(
            raw_context,
            config.expert_names,
            config.expert_instructions,
        )
        for expert_idx, (expert_name, context) in enumerate(zip(config.expert_names, contexts, strict=True)):
            if not bool(configured_mask[expert_idx]):
                continue
            messages = build_expert_messages(list(raw_prompt), expert_name, context)
            prompt_ids = _tokenize_expert_prompt(messages, tokenizer)
            if prompt_ids.numel() > config.max_expert_prompt_length:
                continue
            expert_mask[batch_idx, expert_idx] = True
            prompt_ids_rows.append(prompt_ids.to(device=responses.device, dtype=responses.dtype))
            flat_batch_indices.append(batch_idx)
            flat_expert_indices.append(expert_idx)

    flat_batch = torch.tensor(flat_batch_indices, dtype=torch.long, device=responses.device)
    flat_expert = torch.tensor(flat_expert_indices, dtype=torch.long, device=responses.device)
    selected_responses = responses.index_select(0, flat_batch)
    selected_response_mask = response_mask.index_select(0, flat_batch).to(dtype=torch.long)

    if not prompt_ids_rows:
        response_length = responses.shape[1]
        empty_inputs = responses.new_empty((0, response_length))
        empty_mask = response_mask.new_empty((0, response_length), dtype=torch.long)
        return ExpertPromptInputs(
            input_ids=empty_inputs,
            attention_mask=empty_mask,
            position_ids=empty_mask.clone(),
            responses=selected_responses,
            response_mask=selected_response_mask,
            expert_mask=expert_mask,
            flat_batch_indices=flat_batch,
            flat_expert_indices=flat_expert,
        )

    max_prompt_length = max(row.numel() for row in prompt_ids_rows)
    padded_prompts = []
    prompt_masks = []
    for prompt_ids in prompt_ids_rows:
        pad_length = max_prompt_length - prompt_ids.numel()
        padding = prompt_ids.new_full((pad_length,), tokenizer.pad_token_id)
        padded_prompts.append(torch.cat((padding, prompt_ids)))
        prompt_masks.append(
            torch.cat(
                (
                    torch.zeros(pad_length, dtype=torch.long, device=responses.device),
                    torch.ones(prompt_ids.numel(), dtype=torch.long, device=responses.device),
                )
            )
        )

    prompt_tensor = torch.stack(padded_prompts)
    prompt_mask_tensor = torch.stack(prompt_masks)
    input_ids = torch.cat((prompt_tensor, selected_responses), dim=-1)
    attention_mask = torch.cat((prompt_mask_tensor, selected_response_mask), dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)

    return ExpertPromptInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        responses=selected_responses,
        response_mask=selected_response_mask,
        expert_mask=expert_mask,
        flat_batch_indices=flat_batch,
        flat_expert_indices=flat_expert,
    )


def attach_expert_prompt_tensors(batch: DataProto, packed: ExpertPromptInputs) -> DataProto:
    """Return a clean copy of ``batch`` with dense expert transport tensors attached."""
    batch_size, num_experts = packed.expert_mask.shape
    if batch_size != len(batch):
        raise ValueError(f"expert mask batch size {batch_size} does not match source batch size {len(batch)}")

    flat_count = packed.input_ids.shape[0]
    if flat_count != packed.flat_batch_indices.numel() or flat_count != packed.flat_expert_indices.numel():
        raise ValueError("flat expert tensors and reversible indices must have equal lengths")

    sequence_length = packed.input_ids.shape[1]
    response_length = packed.responses.shape[1]
    dense = {
        "expert_input_ids": packed.input_ids.new_zeros((batch_size, num_experts, sequence_length)),
        "expert_attention_mask": packed.attention_mask.new_zeros((batch_size, num_experts, sequence_length)),
        "expert_position_ids": packed.position_ids.new_zeros((batch_size, num_experts, sequence_length)),
        "expert_responses": packed.responses.new_zeros((batch_size, num_experts, response_length)),
        "expert_response_mask": packed.response_mask.new_zeros((batch_size, num_experts, response_length)),
    }
    if flat_count:
        index = (packed.flat_batch_indices, packed.flat_expert_indices)
        packed_values = (
            packed.input_ids,
            packed.attention_mask,
            packed.position_ids,
            packed.responses,
            packed.response_mask,
        )
        for key, value in zip(_ATTACHED_TENSOR_KEYS, packed_values, strict=True):
            dense[key][index] = value

    tensor_batch = batch.batch.clone()
    for key, value in dense.items():
        tensor_batch[key] = value
    tensor_batch["expert_mask"] = packed.expert_mask

    non_tensor_batch = {
        key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in batch.non_tensor_batch.items()
        if key != "expert_contexts"
    }
    return type(batch)(
        batch=tensor_batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=copy.deepcopy(batch.meta_info),
    )


__all__ = [
    "EVIDENCE_MARKER",
    "FOCUS_MARKER",
    "ExpertContext",
    "ExpertPromptInputs",
    "attach_expert_prompt_tensors",
    "build_expert_messages",
    "normalize_expert_contexts",
    "prepare_expert_prompt_inputs",
]
