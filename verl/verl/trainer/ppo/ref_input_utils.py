# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Utilities for preparing reference / proxy model inputs when the ref model uses a
different tokenizer, chat template, or both.

Phase 0/1 cross-tokenizer PUST:
  - Text-bridge responses: primary decode -> ref encode
  - Common-token mask on aligned primary response positions
  - Teacher log-prob alignment back to primary response length
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def tokenizers_need_cross_token_bridge(primary_tokenizer, ref_tokenizer) -> bool:
    """Return True when primary and ref tokenizers are not byte-identical vocab maps."""
    if ref_tokenizer is None:
        return False
    if primary_tokenizer is ref_tokenizer:
        return False
    return primary_tokenizer.get_vocab() != ref_tokenizer.get_vocab()


def build_shared_token_ids(primary_tokenizer, ref_tokenizer) -> set[int]:
    """Token ids present in both vocabs with identical string->id mapping."""
    primary_vocab = primary_tokenizer.get_vocab()
    ref_vocab = ref_tokenizer.get_vocab()
    shared_tokens = set(primary_vocab.keys()) & set(ref_vocab.keys())
    return {primary_vocab[token] for token in shared_tokens if primary_vocab[token] == ref_vocab[token]}


def _get_primary_response_mask(batch: DataProto, batch_idx: int, response_length: int) -> torch.Tensor:
    if "response_mask" in batch.batch:
        return batch.batch["response_mask"][batch_idx].to(dtype=torch.long)
    return torch.ones(response_length, dtype=torch.long)


def _decode_primary_response_text(primary_tokenizer, responses_row: torch.Tensor, response_mask_row: torch.Tensor) -> str:
    valid_len = int(response_mask_row.sum().item())
    if valid_len <= 0:
        return ""
    valid_ids = responses_row[:valid_len]
    return primary_tokenizer.decode(valid_ids, skip_special_tokens=False)


def _build_common_token_mask(
    primary_resp_valid: torch.Tensor,
    ref_resp_ids: torch.Tensor,
    primary_response_length: int,
    response_mask_row: torch.Tensor,
    shared_token_ids: set[int],
) -> torch.Tensor:
    """Mark primary response positions where primary/ref token ids match in shared vocab."""
    common_mask = torch.zeros(primary_response_length, dtype=torch.long)
    min_len = min(int(response_mask_row.sum().item()), len(ref_resp_ids))
    for pos in range(min_len):
        primary_id = int(primary_resp_valid[pos].item())
        ref_id = int(ref_resp_ids[pos].item())
        if primary_id == ref_id and primary_id in shared_token_ids:
            common_mask[pos] = 1
    return common_mask * response_mask_row.to(dtype=torch.long)


def _right_pad_1d(tokens: torch.Tensor, target_len: int, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    cur_len = len(tokens)
    if cur_len >= target_len:
        trimmed = tokens[:target_len]
        mask = torch.ones(target_len, dtype=torch.long)
        return trimmed, mask
    pad_len = target_len - cur_len
    padding = torch.full((pad_len,), pad_token_id, dtype=tokens.dtype)
    padded = torch.cat([tokens, padding], dim=0)
    mask = torch.cat([torch.ones(cur_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)], dim=0)
    return padded, mask


def prepare_ref_model_inputs(
    batch: DataProto,
    ref_tokenizer,
    primary_tokenizer=None,
    cross_token_bridge: bool = False,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> tuple[DataProto, dict]:
    """Prepare ref-side tensors for proxy / teacher log-prob computation.

    When ``cross_token_bridge`` is False (same vocab, possibly different chat template),
    responses are reused from the primary rollout as before.

    When ``cross_token_bridge`` is True (Phase 0/1), responses are bridged through text:
      primary decode -> ref encode, and ``cross_token_opd_mask`` is added for PUST.
    """
    if apply_chat_template_kwargs is None:
        apply_chat_template_kwargs = {}
    
    # Check if raw_prompt is available
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError(
            "raw_prompt not found in batch.non_tensor_batch. "
            "Please set data.return_raw_chat=True in config to enable re-tokenization for ref model."
        )
    if cross_token_bridge and primary_tokenizer is None:
        raise ValueError("primary_tokenizer is required when cross_token_bridge=True.")

    batch_size = len(batch)
    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    responses = batch.batch["responses"]
    primary_response_length = responses.shape[1]

    if "response_mask" in batch.batch:
        primary_response_attention_mask = batch.batch["response_mask"].to(dtype=torch.long)
    else:
        primary_response_attention_mask = torch.ones_like(responses, dtype=torch.long)

    shared_token_ids = (
        build_shared_token_ids(primary_tokenizer, ref_tokenizer) if cross_token_bridge else set()
    )

    ref_prompt_ids_list = []
    ref_response_ids_list = []
    cross_token_masks = []

    for i in range(batch_size):
        # Get the raw messages for this sample
        messages = raw_prompts[i]
        if not isinstance(messages, (list, np.ndarray)):
            raise TypeError(f"raw_prompt must be a list or numpy array, got {type(messages)}")
        messages = list(messages)
        
        # Apply chat template to get the prompt string using ref tokenizer
        ref_prompt_str = ref_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Tokenize prompt
        ref_prompt_output = ref_tokenizer(
            ref_prompt_str, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        ref_prompt_ids = ref_prompt_output["input_ids"][0]
        ref_prompt_ids_list.append(ref_prompt_ids)

        if cross_token_bridge:
            response_mask_row = _get_primary_response_mask(batch, i, primary_response_length)
            response_text = _decode_primary_response_text(
                primary_tokenizer, responses[i], response_mask_row
            )
            ref_resp_ids = torch.tensor(
                ref_tokenizer.encode(response_text, add_special_tokens=False),
                dtype=responses.dtype,
            )
            valid_len = int(response_mask_row.sum().item())
            primary_resp_valid = responses[i, :valid_len]
            cross_token_masks.append(
                _build_common_token_mask(
                    primary_resp_valid=primary_resp_valid,
                    ref_resp_ids=ref_resp_ids,
                    primary_response_length=primary_response_length,
                    response_mask_row=response_mask_row,
                    shared_token_ids=shared_token_ids,
                )
            )
            ref_response_ids_list.append(ref_resp_ids)
        else:
            ref_response_ids_list.append(responses[i])

    max_prompt_len = max(len(ids) for ids in ref_prompt_ids_list)
    ref_prompt_ids_padded = []
    ref_prompt_attention_mask_padded = []

    for ref_prompt_ids in ref_prompt_ids_list:
        prompt_len = len(ref_prompt_ids)
        pad_len = max_prompt_len - prompt_len
        if pad_len > 0:
            # Left pad prompt ids
            padding = torch.full((pad_len,), ref_tokenizer.pad_token_id, dtype=ref_prompt_ids.dtype)
            ref_prompt_ids_padded_item = torch.cat([padding, ref_prompt_ids], dim=0)
            # Create attention mask: 0 for padding, 1 for real tokens
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ], dim=0)
        else:
            ref_prompt_ids_padded_item = ref_prompt_ids
            attention_mask = torch.ones(prompt_len, dtype=torch.long)
        ref_prompt_ids_padded.append(ref_prompt_ids_padded_item)
        ref_prompt_attention_mask_padded.append(attention_mask)

    ref_prompt_ids_tensor = torch.stack(ref_prompt_ids_padded, dim=0)
    ref_prompt_attention_mask_tensor = torch.stack(ref_prompt_attention_mask_padded, dim=0)

    if cross_token_bridge:
        max_ref_response_len = max(len(ids) for ids in ref_response_ids_list)
        ref_responses_padded = []
        ref_response_mask_padded = []
        primary_ref_len_diffs = []

        for i, ref_resp_ids in enumerate(ref_response_ids_list):
            padded_resp, ref_resp_mask = _right_pad_1d(
                ref_resp_ids,
                max_ref_response_len,
                ref_tokenizer.pad_token_id,
            )
            ref_responses_padded.append(padded_resp)
            ref_response_mask_padded.append(ref_resp_mask)
            primary_valid_len = int(primary_response_attention_mask[i].sum().item())
            primary_ref_len_diffs.append(abs(primary_valid_len - len(ref_resp_ids)))

        ref_responses_tensor = torch.stack(ref_responses_padded, dim=0)
        ref_response_mask_tensor = torch.stack(ref_response_mask_padded, dim=0)
        cross_token_opd_mask = torch.stack(cross_token_masks, dim=0)

        ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, ref_responses_tensor], dim=1)
        ref_attention_mask_tensor = torch.cat(
            [ref_prompt_attention_mask_tensor, ref_response_mask_tensor], dim=1
        )

        batch.batch["ref_responses"] = ref_responses_tensor
        batch.batch["ref_response_mask"] = ref_response_mask_tensor
        batch.batch["cross_token_opd_mask"] = cross_token_opd_mask

        valid_primary_tokens = primary_response_attention_mask.sum().clamp(min=1).float()
        common_tokens = cross_token_opd_mask.sum().float()
        stats = {
            "hetero/cross_token_common_ratio": (common_tokens / valid_primary_tokens).item(),
            "hetero/cross_token_primary_ref_len_diff_mean": float(np.mean(primary_ref_len_diffs)),
            "hetero/cross_token_ref_response_len_mean": float(ref_response_mask_tensor.sum(dim=-1).float().mean().item()),
        }
    else:
        ref_input_ids_tensor = torch.cat([ref_prompt_ids_tensor, responses], dim=1)
        ref_attention_mask_tensor = torch.cat(
            [ref_prompt_attention_mask_tensor, primary_response_attention_mask], dim=1
        )
        stats = {}

    ref_position_ids_tensor = compute_position_id_with_mask(ref_attention_mask_tensor)
    
    # Add to batch
    batch.batch["ref_input_ids"] = ref_input_ids_tensor
    batch.batch["ref_attention_mask"] = ref_attention_mask_tensor
    batch.batch["ref_position_ids"] = ref_position_ids_tensor

    logger.info(
        "Prepared ref model inputs: primary_responses=%s ref_input_ids=%s cross_token_bridge=%s",
        tuple(responses.shape),
        tuple(ref_input_ids_tensor.shape),
        cross_token_bridge,
    )

    return batch, stats


_REF_SIDE_LOG_PROB_KEYS = (
    "teacher_log_probs",
    "teacher_base_log_probs",
    "ref_log_prob",
    "base_ref_log_prob",
)


def _align_log_prob_tensor_to_primary(teacher_lp: torch.Tensor, primary_len: int) -> torch.Tensor:
    """Align the log probability tensor to the primary response length."""
    ref_len = teacher_lp.shape[1]
    if ref_len == primary_len:
        return teacher_lp
    if ref_len > primary_len:
        return teacher_lp[:, :primary_len] if teacher_lp.dim() == 2 else teacher_lp[:, :primary_len, :]
    pad_shape = (teacher_lp.shape[0], primary_len - ref_len)
    if teacher_lp.dim() == 3:
        pad_shape = (teacher_lp.shape[0], primary_len - ref_len, teacher_lp.shape[2])
    padding = torch.zeros(pad_shape, dtype=teacher_lp.dtype, device=teacher_lp.device)
    return torch.cat([teacher_lp, padding], dim=1)


def align_ref_side_log_probs_to_primary(
    batch: DataProto,
    keys: tuple[str, ...] | None = None,
) -> tuple[DataProto, dict]:
    """Align ref/proxy-side log probs onto the primary response length axis."""
    if "cross_token_opd_mask" not in batch.batch:
        return batch, {}

    primary_len = batch.batch["responses"].shape[1]
    if keys is None:
        keys = _REF_SIDE_LOG_PROB_KEYS

    for key in keys:
        if key not in batch.batch:
            continue
        batch.batch[key] = _align_log_prob_tensor_to_primary(batch.batch[key], primary_len)

    return batch, {}


def align_teacher_log_probs_to_primary(batch: DataProto) -> tuple[DataProto, dict]:
    """Backward-compatible wrapper used by PUST / hetero OPD."""
    return align_ref_side_log_probs_to_primary(
        batch,
        keys=("teacher_log_probs", "teacher_base_log_probs"),
    )


def maybe_apply_sequence_level_opd_fallback(
    batch: DataProto,
    common_ratio_threshold: float = 0.8,
) -> tuple[DataProto, dict]:
    """Phase 1 extension hook for sparse common-token regions.

    TODO(Phase 1+): When ``hetero/cross_token_common_ratio`` falls below
    ``common_ratio_threshold``, replace token-level advantages with a sequence-level
    PUST signal broadcast across ``response_mask`` (for larger tokenizer gaps).
    """
    stats: dict[str, float] = {}
    if "cross_token_opd_mask" not in batch.batch:
        return batch, stats

    response_mask = batch.batch["response_mask"].float()
    common_mask = batch.batch["cross_token_opd_mask"].float()
    common_ratio = (common_mask.sum() / response_mask.sum().clamp(min=1.0)).item()
    stats["hetero/cross_token_common_ratio_post_align"] = common_ratio

    if common_ratio < common_ratio_threshold:
        # TODO(Phase 1+): implement sequence-level fallback here.
        logger.warning(
            "cross_token_common_ratio=%.4f < threshold=%.2f; "
            "sequence-level PUST fallback not implemented yet.",
            common_ratio,
            common_ratio_threshold,
        )
        stats["hetero/cross_token_sequence_fallback_would_trigger"] = 1.0
    else:
        stats["hetero/cross_token_sequence_fallback_would_trigger"] = 0.0

    return batch, stats
