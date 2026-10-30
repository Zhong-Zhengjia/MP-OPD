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

"""Field-span aggregation for MP-OPD (method_new §6).

method_new §6 replaces the plain token-mean reverse KL with a field-balanced
aggregation over category (``兴趣``) / rationale (``理由``) / structural spans of
the student's JSON response::

    Lc[u,i] = mean(loss_token over category span of entry i)
    Lr[u,i] = mean(loss_token over rationale span of entry i)
    Lcontent = mean_u(mean_i(eta * Lc[u,i] + (1-eta) * Lr[u,i]))
    Lformat = mean_u(mean(loss_token over structural positions))
    L = Lcontent + gamma * Lformat

The hierarchy -- field-within-entry, entry-within-user, user-within-batch -- is
mandatory: pooling every content token across the batch would let long responses
dominate and is explicitly forbidden by method_new. ``gamma`` defaults to 0.0 so
``Lformat`` is off by default and only ``Lcontent`` is computed.

Span recovery reuses the probe labeler logic (``scripts/expert_variance_probe.py``
``label_response``): decode each response token id individually and accumulate
character offsets, so spans stay aligned to the actual token ids rather than a
re-tokenization of the decoded text (which can drift by a boundary token). This
module re-implements that logic so the training loop need not depend on the
probe script.

Exception policy (method_new §6 leaves these open; we pick a default):
  * malformed JSON / no entry (no 兴趣 and no 理由 values found):
    ``parse_ok=False``. The sample's ``Lcontent`` is skipped, but every response
    token is tagged structural so ``Lformat`` still covers it (only matters when
    ``gamma>0``).
  * empty field (an entry with a 兴趣 but no 理只, or a value that decodes to no
    tokens): that field does not participate in its entry's mean; the entry loss
    collapses to the available field (weights are not renormalized across the
    missing term -- ``eta`` still scales the category side).
  * misaligned entries (``#兴趣 != #理由``): per-entry pairing is unreliable, so
    the sample degrades to per-field pooling (all category tokens averaged, all
    rationale tokens averaged) combined with ``eta``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

__all__ = [
    "FieldMasks",
    "compute_field_masks",
    "compute_mpopd_field_loss",
]


# Match the JSON values for the Chinese keys used by the recommendation schema
# (see baseline/student_prompt.txt). The capture group is the unescaped value.
_INTEREST_RE = re.compile(r'"兴趣"\s*:\s*"((?:[^"\\]|\\.)*)"')
_REASON_RE = re.compile(r'"理由"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _span_within(token_span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    """True if ``token_span`` lies inside one of ``spans`` (half-open char ranges)."""
    ts, te = token_span
    for s, e in spans:
        if ts >= s and te <= e:
            return True
    return False


def _token_char_spans(tokenizer, r_ids: list[int], r_text: str) -> list[tuple[int, int]]:
    """Map each response token id to its character span [start, end) in ``r_text``.

    Decoding each id individually and accumulating lengths aligns spans to the
    ACTUAL ``r_ids`` rather than a re-tokenization of ``r_text`` (those can differ
    by a BOS/merge token and silently shift labels). For byte-level BPE (Qwen)
    the individual decodes concatenate to ``r_text``; we verify and fall back to
    re-tokenization offsets if they diverge.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for tid in r_ids:
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    if pos == len(r_text):
        return spans
    # Individual-decode concatenation diverged from r_text (rare). Fall back to
    # re-tokenizing r_text for offset mapping; labels are best-effort in this case.
    try:
        enc = tokenizer(r_text, return_offsets_mapping=True, add_special_tokens=False)
        retok_offsets = enc["offset_mapping"]
    except Exception:
        # Tokenizer does not support return_offsets_mapping; pad with (-1,-1).
        retok_offsets = []
    if len(retok_offsets) == len(r_ids):
        return list(retok_offsets)
    out = list(retok_offsets[: len(r_ids)])
    while len(out) < len(r_ids):
        out.append((-1, -1))
    return out


@dataclass
class FieldMasks:
    """Per-position field labels plus per-entry token indices for content loss.

    ``category_mask`` / ``rationale_mask`` / ``structure_mask`` are bool [B,T]
    aligned to the full input sequence (same T as ``token_kl``); only response
    positions (where ``response_mask=1``) can be set. ``parse_ok`` is bool [B]:
    True iff the response yielded at least one 兴趣 or 理由 value.

    ``entry_category_idx`` / ``entry_rationale_idx`` are ragged: for each sample,
    a list (one per entry) of 1-D LongTensors holding the full-sequence positions
    of that entry's category / rationale tokens. They drive the per-entry means
    in ``compute_mpopd_field_loss``. When entries are misaligned (unequal counts)
    the sample is flagged via ``misaligned`` and the loss falls back to per-field
    pooling, so the per-entry lists are not used for that sample.
    """

    category_mask: torch.Tensor
    rationale_mask: torch.Tensor
    structure_mask: torch.Tensor
    parse_ok: torch.Tensor
    entry_category_idx: list[list[torch.Tensor]] = field(default_factory=list)
    entry_rationale_idx: list[list[torch.Tensor]] = field(default_factory=list)
    misaligned: torch.Tensor | None = None


def compute_field_masks(
    tokenizer,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> FieldMasks:
    """Label each response position with its field (category / rationale / structure).

    Args:
        tokenizer: a HF tokenizer (must support ``decode``; ``return_offsets_mapping``
            is used only as a fallback).
        response_ids: long [B, T_response] -- the generated response token ids (the
            ``responses`` column, NOT the full ``input_ids``). ``T_response`` is the
            response length and matches ``response_mask`` and the actor's per-position
            ``token_kl``, so the returned masks align with ``token_kl`` directly.
        response_mask: bool/float [B, T_response] -- 1 on valid (non-pad) response
            tokens, 0 on padding within the response.

    Returns:
        FieldMasks with [B, T_response] bool masks aligned to ``response_ids`` plus
        per-entry token-index lists. Tensors live on ``response_ids.device``.
    """
    device = response_ids.device
    B, T = response_ids.shape
    rm = response_mask.bool()
    category_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    rationale_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    structure_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    parse_ok = torch.zeros(B, dtype=torch.bool, device=device)
    misaligned = torch.zeros(B, dtype=torch.bool, device=device)
    entry_category_idx: list[list[torch.Tensor]] = [[] for _ in range(B)]
    entry_rationale_idx: list[list[torch.Tensor]] = [[] for _ in range(B)]

    for b in range(B):
        resp_pos = rm[b].nonzero(as_tuple=False).squeeze(-1)
        n_resp = int(resp_pos.numel())
        if n_resp == 0:
            continue
        r_ids = response_ids[b, resp_pos].tolist()
        r_text = tokenizer.decode(r_ids, skip_special_tokens=False)
        offsets = _token_char_spans(tokenizer, r_ids, r_text)
        interest_spans = [m.span(1) for m in _INTEREST_RE.finditer(r_text)]
        reason_spans = [m.span(1) for m in _REASON_RE.finditer(r_text)]
        if not interest_spans and not reason_spans:
            # parse_fail: every response token is structural; no content loss.
            structure_mask[b, resp_pos] = True
            continue
        parse_ok[b] = True
        if len(interest_spans) != len(reason_spans):
            misaligned[b] = True

        # Flat per-position labels (used for Lformat and for the per-field
        # fallback when entries are misaligned).
        for i in range(min(n_resp, len(offsets))):
            ts, te = offsets[i]
            fp = int(resp_pos[i].item())
            if ts < 0 or te <= ts:
                structure_mask[b, fp] = True
                continue
            if _span_within((ts, te), interest_spans):
                category_mask[b, fp] = True
            elif _span_within((ts, te), reason_spans):
                rationale_mask[b, fp] = True
            else:
                structure_mask[b, fp] = True

        # Per-entry token-index lists (only used when entries are aligned).
        if not misaligned[b]:
            for ei, sp in enumerate(interest_spans):
                idxs = [
                    int(resp_pos[i].item())
                    for i in range(min(n_resp, len(offsets)))
                    if offsets[i][0] >= 0 and offsets[i][1] > offsets[i][0]
                    and offsets[i][0] >= sp[0] and offsets[i][1] <= sp[1]
                ]
                entry_category_idx[b].append(torch.tensor(idxs, dtype=torch.long, device=device))
            for ei, sp in enumerate(reason_spans):
                idxs = [
                    int(resp_pos[i].item())
                    for i in range(min(n_resp, len(offsets)))
                    if offsets[i][0] >= 0 and offsets[i][1] > offsets[i][0]
                    and offsets[i][0] >= sp[0] and offsets[i][1] <= sp[1]
                ]
                entry_rationale_idx[b].append(torch.tensor(idxs, dtype=torch.long, device=device))

    return FieldMasks(
        category_mask=category_mask,
        rationale_mask=rationale_mask,
        structure_mask=structure_mask,
        parse_ok=parse_ok,
        entry_category_idx=entry_category_idx,
        entry_rationale_idx=entry_rationale_idx,
        misaligned=misaligned,
    )


def _mean_over_positions(token_kl_row: torch.Tensor, positions: torch.Tensor) -> torch.Tensor | None:
    """Mean of ``token_kl_row`` over ``positions``; None if positions is empty."""
    if positions.numel() == 0:
        return None
    return token_kl_row[positions].mean()


def compute_mpopd_field_loss(
    token_kl: torch.Tensor,
    field_masks: FieldMasks,
    eta: float,
    gamma: float,
    response_mask: torch.Tensor,
    valid_samples: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Aggregate per-position reverse KL into the field-balanced loss (method_new §6).

    Args:
        token_kl: [B,T] per-position reverse KL (carries gradient from the student).
        field_masks: from :func:`compute_field_masks`.
        eta: category/rationale weight, 0<eta<1.
        gamma: structural-loss coefficient, >=0 (0.0 => Lformat omitted).
        response_mask: [B,T] -- 1 on response positions.
        valid_samples: [B] bool -- samples with at least one active expert. Content
            and structure losses are both restricted to these samples.

    Returns:
        ``(loss, diagnostics)`` where loss is a scalar (gradient-bearing) and
        diagnostics has ``content_samples``, ``format_samples``, ``parse_ok_rate``.
    """
    device = token_kl.device
    B = token_kl.shape[0]
    rm = response_mask.bool()
    valid = valid_samples.bool() if valid_samples is not None else torch.ones(B, dtype=torch.bool, device=device)

    content_terms: list[torch.Tensor] = []
    format_terms: list[torch.Tensor] = []
    parse_ok_count = 0
    valid_count = 0

    for b in range(B):
        if not valid[b]:
            continue
        valid_count += 1
        if bool(field_masks.parse_ok[b].item()):
            parse_ok_count += 1

        # Lformat: structural positions of this sample (response only). Computed
        # for every valid sample regardless of parse_ok -- malformed responses
        # have every response token tagged structural.
        if gamma > 0.0:
            struct_pos = field_masks.structure_mask[b] & rm[b]
            if struct_pos.any():
                format_terms.append(token_kl[b][struct_pos].mean())

        # Lcontent: only for valid + parse_ok samples.
        if not valid[b] or not bool(field_masks.parse_ok[b].item()):
            continue

        if not bool(field_masks.misaligned[b].item()):
            # Per-entry: mean within each entry's category/rationale span, combine,
            # then mean over entries.
            cat_entries = field_masks.entry_category_idx[b]
            rat_entries = field_masks.entry_rationale_idx[b]
            n_entries = max(len(cat_entries), len(rat_entries))
            entry_losses: list[torch.Tensor] = []
            for ei in range(n_entries):
                lc = _mean_over_positions(token_kl[b], cat_entries[ei]) if ei < len(cat_entries) else None
                lr = _mean_over_positions(token_kl[b], rat_entries[ei]) if ei < len(rat_entries) else None
                if lc is not None and lr is not None:
                    entry_losses.append(eta * lc + (1.0 - eta) * lr)
                elif lc is not None:
                    entry_losses.append(lc)
                elif lr is not None:
                    entry_losses.append(lr)
            if entry_losses:
                content_terms.append(torch.stack(entry_losses).mean())
        else:
            # Misaligned entries: fall back to per-field pooling within the sample.
            cat_pos = field_masks.category_mask[b] & rm[b]
            rat_pos = field_masks.rationale_mask[b] & rm[b]
            lc = _mean_over_positions(token_kl[b], cat_pos)
            lr = _mean_over_positions(token_kl[b], rat_pos)
            if lc is not None and lr is not None:
                content_terms.append(eta * lc + (1.0 - eta) * lr)
            elif lc is not None:
                content_terms.append(lc)
            elif lr is not None:
                content_terms.append(lr)

    Lcontent = torch.stack(content_terms).mean() if content_terms else torch.zeros((), device=device)
    Lformat = torch.stack(format_terms).mean() if format_terms else torch.zeros((), device=device)
    loss = Lcontent + gamma * Lformat
    diagnostics = {
        "content_samples": float(len(content_terms)),
        "format_samples": float(len(format_terms)),
        "parse_ok_rate": float(parse_ok_count) / float(valid_count) if valid_count else 0.0,
    }
    return loss, diagnostics
