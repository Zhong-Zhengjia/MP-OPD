"""Unit tests for MP-OPD field-span aggregation (method_new §6).

These run on CPU without a real HF model. A char-level mock tokenizer maps
``id == ord(char)`` so each response character is one token, which lets us
construct JSON responses with known 兴趣/理由 spans and assert the masks and
the field-balanced loss exactly.
"""

from __future__ import annotations

import torch

from verl.trainer.ppo.field_loss import (
    FieldMasks,
    compute_field_masks,
    compute_mpopd_field_loss,
)


class _CharTokenizer:
    """Char-level tokenizer: ``decode([tid]) == chr(tid)``.

    Each character is exactly one token, so per-id decodes concatenate to the
    full text and ``_token_char_spans`` returns ``[(0,1),(1,2),...]``. This is
    enough to exercise the span→token mapping without a real model.
    """

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(t)) for t in ids)

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        result = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


TOK = _CharTokenizer()
PROMPT_LEN = 4
PROMPT_IDS = [1, 2, 3, 4]  # arbitrary; never decoded (response_mask excludes them)


def _make_inputs(*response_texts: str):
    """Build (input_ids [B,T], response_mask [B,T]) with the response at the end."""
    rows, masks = [], []
    for rtext in response_texts:
        r_ids = [ord(c) for c in rtext]
        rows.append(PROMPT_IDS + r_ids)
        masks.append([0] * PROMPT_LEN + [1] * len(r_ids))
    maxlen = max(len(r) for r in rows)
    # left-pad the prompt side so the response stays right-aligned and the
    # response tokens always start at the same column per sample is NOT required;
    # we just pad to the max length with zeros on the right and extend masks.
    input_ids = torch.zeros(len(rows), maxlen, dtype=torch.long)
    response_mask = torch.zeros(len(rows), maxlen, dtype=torch.long)
    for b, (r, m) in enumerate(zip(rows, masks)):
        input_ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
        response_mask[b, : len(m)] = torch.tensor(m, dtype=torch.long)
    return input_ids, response_mask


def _resp_positions(response_mask, b=0):
    return response_mask[b].nonzero(as_tuple=False).squeeze(-1)


def _kl_for_response(response_text, token_values, default=0.0, prompt_len=PROMPT_LEN):
    """Return a [1, T] token_kl row. ``token_values`` maps a substring of
    ``response_text`` to a scalar; tokens at those char positions get the value,
    other response tokens get ``default``. Prompt positions are 0.
    """
    r_ids = [ord(c) for c in response_text]
    T = prompt_len + len(r_ids)
    kl = torch.zeros(1, T)
    for i in range(len(r_ids)):
        kl[0, prompt_len + i] = default
    for substr, val in token_values.items():
        start = response_text.find(substr)
        assert start >= 0, f"substring {substr!r} not in response"
        for j in range(len(substr)):
            kl[0, prompt_len + start + j] = val
    return kl


def _batch_kl(rtexts, per_sample_values, defaults, total_len, prompt_len=PROMPT_LEN):
    """Build [len(rtexts), total_len] token_kl. Each response's KL is placed at
    ``[b, prompt_len : prompt_len + len(rtext)]`` and the rest is 0, so samples of
    different lengths can coexist in one batch.
    """
    kl = torch.zeros(len(rtexts), total_len)
    for b, (rtext, vals) in enumerate(zip(rtexts, per_sample_values)):
        dflt = defaults[b] if isinstance(defaults, (list, tuple)) else defaults
        row = _kl_for_response(rtext, vals, default=dflt, prompt_len=prompt_len).squeeze(0)
        kl[b, prompt_len : prompt_len + len(rtext)] = row[prompt_len:]
    return kl


# --------------------------------------------------------------------------
# compute_field_masks
# --------------------------------------------------------------------------


def test_single_entry_masks():
    rtext = '{"predictions":[{"兴趣":"手机","理由":"常用"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    resp_pos = _resp_positions(response_mask).tolist()
    # map char index -> token position
    cat_idx = resp_pos[rtext.find("手机")]
    cat_idx2 = resp_pos[rtext.find("手机") + 1]
    rat_idx = resp_pos[rtext.find("常用")]
    rat_idx2 = resp_pos[rtext.find("常用") + 1]
    assert masks.parse_ok[0].item() is True or bool(masks.parse_ok[0])
    assert masks.category_mask[0, cat_idx] and masks.category_mask[0, cat_idx2]
    assert masks.rationale_mask[0, rat_idx] and masks.rationale_mask[0, rat_idx2]
    assert not masks.category_mask[0, rat_idx]
    assert not masks.rationale_mask[0, cat_idx]
    # one entry, aligned
    assert len(masks.entry_category_idx[0]) == 1
    assert len(masks.entry_rationale_idx[0]) == 1
    assert masks.entry_category_idx[0][0].numel() == 2
    assert masks.entry_rationale_idx[0][0].numel() == 2
    assert not bool(masks.misaligned[0])


def test_malformed_json_parse_fail_and_structure_all_response():
    rtext = "this is not json at all"
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    assert not bool(masks.parse_ok[0])
    assert not masks.category_mask.any()
    assert not masks.rationale_mask.any()
    # every response token is structural
    assert masks.structure_mask[0][_resp_positions(response_mask)].all()
    assert len(masks.entry_category_idx[0]) == 0


def test_empty_category_field_entry_uses_only_rationale():
    rtext = '{"predictions":[{"兴趣":"","理由":"常用"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    assert bool(masks.parse_ok[0])
    # empty 兴趣 value -> no category tokens, 1 entry, 2 rationale tokens
    assert len(masks.entry_category_idx[0]) == 1
    assert masks.entry_category_idx[0][0].numel() == 0
    assert masks.entry_rationale_idx[0][0].numel() == 2
    assert not bool(masks.misaligned[0])  # 1 兴趣 == 1 理由


def test_misaligned_entries_flagged():
    # 2 兴趣 values, 1 理由 value -> misaligned
    rtext = '{"predictions":[{"兴趣":"手机","理由":"常用"},{"兴趣":"书"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    assert bool(masks.misaligned[0])


# --------------------------------------------------------------------------
# compute_mpopd_field_loss
# --------------------------------------------------------------------------


def _single_entry_kl(rtext, cat_val, rat_val, struct_val=0.0):
    return _kl_for_response(rtext, {"手机": cat_val, "常用": rat_val}, default=struct_val)


def test_lcontent_single_entry_matches_eta_combination():
    rtext = '{"predictions":[{"兴趣":"手机","理由":"常用"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    token_kl = _single_entry_kl(rtext, cat_val=2.0, rat_val=4.0)
    # category tokens = 2.0 -> Lc=2; rationale = 4.0 -> Lr=4; eta=0.5 -> 3.0
    eta = 0.5
    loss, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=eta, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    assert torch.allclose(loss, torch.tensor(eta * 2.0 + (1 - eta) * 4.0), atol=1e-6)
    assert diag["content_samples"] == 1.0


def test_per_entry_then_mean_not_pooled():
    # Two entries with different category KL; per-entry mean differs from pooled.
    rtext = '{"predictions":[{"兴趣":"手机","理由":"常用"},{"兴趣":"书","理由":"学习"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    # entry1 category "手机"=2.0, entry2 category "书"=10.0; all rationale=4.0
    token_kl = _kl_for_response(rtext, {"手机": 2.0, "书": 10.0, "常用": 4.0, "学习": 4.0})
    eta = 0.5
    loss, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=eta, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    # per-entry: entry1 = 0.5*2+0.5*4 = 3; entry2 = 0.5*10+0.5*4 = 7; mean = 5.0
    assert torch.allclose(loss, torch.tensor(5.0), atol=1e-6)
    # pooled-field would be 0.5*((2+2+10)/3)+0.5*((4+4)/2) = 0.5*4.667+0.5*4 = 4.333 != 5.0


def test_empty_category_field_loss_uses_only_rationale():
    rtext = '{"predictions":[{"兴趣":"","理由":"常用"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    token_kl = _kl_for_response(rtext, {"常用": 4.0}, default=0.0)
    loss, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    # only rationale contributes -> Lr = 4.0
    assert torch.allclose(loss, torch.tensor(4.0), atol=1e-6)


def test_malformed_json_skips_content_format_covers_all_when_gamma():
    rtext = "not json"
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    token_kl = _kl_for_response(rtext, {}, default=1.0)
    # gamma=0 -> Lcontent skipped (parse_fail), Lformat omitted -> 0
    loss0, _ = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    assert torch.allclose(loss0, torch.tensor(0.0), atol=1e-6)
    # gamma=1 -> Lformat = mean over all-structural response = 1.0
    loss1, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=1.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    assert torch.allclose(loss1, torch.tensor(1.0), atol=1e-6)
    assert diag["content_samples"] == 0.0
    assert diag["format_samples"] == 1.0


def test_misaligned_falls_back_to_per_field_pooling():
    rtext = '{"predictions":[{"兴趣":"手机","理由":"常用"},{"兴趣":"书"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    assert bool(masks.misaligned[0])
    # category tokens: 手机(2) + 书(1) = 3 tokens all = 2.0; rationale: 常用(2) = 4.0
    token_kl = _kl_for_response(rtext, {"手机": 2.0, "书": 2.0, "常用": 4.0}, default=0.0)
    eta = 0.5
    loss, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=eta, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    # pooled: 0.5*((2+2+2)/3) + 0.5*((4+4)/2) = 0.5*2 + 0.5*4 = 3.0
    assert torch.allclose(loss, torch.tensor(3.0), atol=1e-6)


def test_gamma_adds_structural_term():
    rtext = '{"predictions":[{"兴趣":"手机","理由":"常用"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    token_kl = _single_entry_kl(rtext, cat_val=2.0, rat_val=4.0, struct_val=1.0)
    # Lcontent = 3.0, Lformat = mean over structure = 1.0; gamma=0.5 -> 3.0 + 0.5*1.0
    loss, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=0.5,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    assert torch.allclose(loss, torch.tensor(3.5), atol=1e-6)
    assert diag["format_samples"] == 1.0


def test_invalid_samples_excluded_from_both_content_and_format():
    rtext0 = '{"predictions":[{"兴趣":"手机","理由":"常用"}]}'
    rtext1 = '{"predictions":[{"兴趣":"书","理由":"学"}]}'
    input_ids, response_mask = _make_inputs(rtext0, rtext1)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    token_kl = _batch_kl(
        [rtext0, rtext1],
        [{"手机": 2.0, "常用": 4.0}, {"书": 2.0, "学": 4.0}],
        defaults=[1.0, 1.0],
        total_len=input_ids.shape[1],
    )
    # sample 1 invalid
    valid = torch.tensor([True, False])
    loss, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=0.5,
        response_mask=response_mask, valid_samples=valid,
    )
    # only sample 0 contributes: Lcontent=3.0, Lformat=1.0 -> 3.5
    assert torch.allclose(loss, torch.tensor(3.5), atol=1e-6)
    assert diag["content_samples"] == 1.0
    assert diag["format_samples"] == 1.0


def test_parse_ok_rate_diagnostic():
    rtext0 = '{"predictions":[{"兴趣":"手机","理由":"常用"}]}'  # ok
    rtext1 = "garbage"  # fail
    input_ids, response_mask = _make_inputs(rtext0, rtext1)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    token_kl = _batch_kl(
        [rtext0, rtext1],
        [{"手机": 2.0, "常用": 4.0}, {}],
        defaults=[0.0, 1.0],
        total_len=input_ids.shape[1],
    )
    _, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True, True]),
    )
    assert abs(diag["parse_ok_rate"] - 0.5) < 1e-6


def test_field_loss_preserves_gradient():
    rtext = '{"predictions":[{"兴趣":"手机","理由":"常用"}]}'
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    token_kl = _single_entry_kl(rtext, cat_val=2.0, rat_val=4.0)
    token_kl.requires_grad_(True)
    loss, _ = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    loss.backward()
    assert token_kl.grad is not None
    assert token_kl.grad.abs().sum() > 0


def test_all_malformed_preserves_gradient_for_backward():
    # Regression: when every sample in the micro-batch has parse_ok=False
    # (malformed JSON, no 兴趣/理由 found), content_terms is empty. The
    # fallback must stay grad-bearing or the downstream (loss*scale).backward()
    # raises "element 0 of tensors does not require grad" -- which is exactly
    # what happened in the first MP-OPD smoke run (0.8B student at step 0
    # rarely emits valid JSON, and ppo_micro_batch_size_per_gpu=1 so each
    # micro-batch is one sample). The grad-bearing zero (token_kl.sum()*0.0)
    # keeps the graph so backward succeeds with a zero gradient.
    rtext = "this is not json at all"
    input_ids, response_mask = _make_inputs(rtext)
    masks = compute_field_masks(TOK, input_ids, response_mask)
    assert not bool(masks.parse_ok[0])  # precondition: empty content_terms
    token_kl = _kl_for_response(rtext, {}, default=1.0)
    token_kl.requires_grad_(True)
    loss, diag = compute_mpopd_field_loss(
        token_kl, masks, eta=0.5, gamma=0.0,
        response_mask=response_mask, valid_samples=torch.tensor([True]),
    )
    # loss is zero (no content), but must carry a grad_fn so backward works.
    assert loss.requires_grad
    assert loss.grad_fn is not None
    loss.backward()  # must not raise
    assert token_kl.grad is not None
    # zero loss -> zero gradient (the sample correctly contributes nothing).
    assert torch.equal(token_kl.grad, torch.zeros_like(token_kl.grad))
    assert diag["content_samples"] == 0.0
