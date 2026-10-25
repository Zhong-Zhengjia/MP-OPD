#!/usr/bin/env python3
"""Pure-stdlib tests for expert_variance_probe labeler and prompt builder.

No torch / vLLM / transformers needed — safe to run on the local Mac:
  python scripts/test_expert_variance_probe.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_variance_probe import (  # noqa: E402
    build_expert_messages,
    label_response,
    load_probe_prompts,
)


class _FakeTok:
    """Char-level tokenizer stub for label_response testing.

    Each char -> its ord; decode inverts that so _token_char_spans (which
    decodes each id individually and accumulates lengths) recovers the exact
    per-char spans, matching the real byte-level-BPE property the probe
    relies on.
    """

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids = [ord(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


def test_build_expert_messages_appends_instruction():
    raw = [{"role": "user", "content": "BASE"}]
    msgs = build_expert_messages(raw, "【专家】", False, None)
    assert msgs[-1]["content"] == "BASE\n\n【专家】"
    assert raw[0]["content"] == "BASE"  # not mutated


def test_build_expert_messages_with_evidence():
    raw = [{"role": "user", "content": "BASE"}]
    msgs = build_expert_messages(raw, "【专家】", True, {"袜子": 1})
    assert "你的领域证据" in msgs[-1]["content"]
    assert "袜子" in msgs[-1]["content"]
    assert msgs[-1]["content"].startswith("BASE\n\n【专家】")


def test_label_response_category_vs_reason():
    tok = _FakeTok()
    r_text = '{"predictions":[{"兴趣":"袜子","理由":"近期点击"}]}'
    r_ids = [ord(c) for c in r_text]
    labels = label_response(tok, r_text, r_ids)
    # 兴趣 value "袜子" -> label 1
    for ch in "袜子":
        assert labels[r_text.index(ch)] == (1, "兴趣"), f"char {ch!r}: {labels[r_text.index(ch)]}"
    # 理由 value "近期点击" -> label 0, field 理由
    for ch in "近期点击":
        assert labels[r_text.index(ch)] == (0, "理由"), f"char {ch!r}: {labels[r_text.index(ch)]}"
    # structural tokens
    assert labels[r_text.index("{")] == (0, "structure")
    assert labels[r_text.index("[")] == (0, "structure")
    assert labels[r_text.index(",")] == (0, "structure")


def test_label_response_multiple_predictions():
    tok = _FakeTok()
    r_text = '{"predictions":[{"兴趣":"A","理由":"X"},{"兴趣":"B","理由":"Y"}]}'
    r_ids = [ord(c) for c in r_text]
    labels = label_response(tok, r_text, r_ids)
    for cat in ("A", "B"):
        assert labels[r_text.index(cat)] == (1, "兴趣")
    for reason in ("X", "Y"):
        assert labels[r_text.index(reason)] == (0, "理由")


def test_label_response_parse_fail():
    tok = _FakeTok()
    r_text = "这不是合法JSON"
    r_ids = [ord(c) for c in r_text]
    labels = label_response(tok, r_text, r_ids)
    assert all(l == 0 and f == "parse_fail" for l, f in labels)


def test_load_probe_prompts_filters_full_evidence():
    data = []
    for s in ["高价值消费者", "潜在用户"]:
        for k in range(3):
            ec = {
                name: {
                    "evidence_available": (k == 0),
                    "instruction": f"【{name}】",
                }
                for name in ["chasing", "long_term", "repurchase", "generalized"]
            }
            data.append({
                "prompt": [{"role": "user", "content": "X"}],
                "extra_info": {
                    "user_id": f"u{k}_{s}",
                    "buyer_type_cn": s,
                    "expert_contexts": ec,
                },
            })
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        path = f.name
    # Each stratum has exactly 1 full-evidence sample (k=0).
    prompts = load_probe_prompts(Path(path), ["高价值消费者", "潜在用户"], 1)
    assert len(prompts) == 2
    assert all(p["sample_idx"] == 0 for p in prompts)
    # Asking for 2 per stratum should raise (only 1 available).
    raised = False
    try:
        load_probe_prompts(Path(path), ["高价值消费者"], 2)
    except ValueError as e:
        raised = True
        assert "insufficient full-evidence" in str(e)
    assert raised, "expected ValueError for insufficient samples"
    # Evidence filtering is correct: selected prompts have all 4 evidence.
    for p in prompts:
        for name in ["chasing", "long_term", "repurchase", "generalized"]:
            assert p["expert_contexts"][name]["evidence_available"] is True


def test_load_probe_prompts_order_preserved():
    # First n full-evidence samples per stratum are selected in file order.
    data = []
    for s in ["高价值消费者"]:
        for k in range(5):
            # k=2 has no evidence; the rest do. Expect u0, u1, u3 selected (3).
            ec = {
                name: {
                    "evidence_available": (k != 2),
                    "instruction": f"【{name}】",
                }
                for name in ["chasing", "long_term", "repurchase", "generalized"]
            }
            data.append({
                "prompt": [{"role": "user", "content": "X"}],
                "extra_info": {
                    "user_id": f"u{k}",
                    "buyer_type_cn": s,
                    "expert_contexts": ec,
                },
            })
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        path = f.name
    prompts = load_probe_prompts(Path(path), ["高价值消费者"], 3)
    assert [p["user_id"] for p in prompts] == ["u0", "u1", "u3"], (
        f"order wrong: {[p['user_id'] for p in prompts]}"
    )


if __name__ == "__main__":
    test_build_expert_messages_appends_instruction()
    test_build_expert_messages_with_evidence()
    test_label_response_category_vs_reason()
    test_label_response_multiple_predictions()
    test_label_response_parse_fail()
    test_load_probe_prompts_filters_full_evidence()
    test_load_probe_prompts_order_preserved()
    print("OK: all tests passed")
