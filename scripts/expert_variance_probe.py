#!/usr/bin/env python3
"""Expert variance probe: record per-token top-20 raw probabilities across
6 conditions for a handful of test prompts, for offline between/within-expert
variance analysis. NOT a training tool; lives on the analysis branch.

For each selected prompt:
  1. Sample one response R from the student (vLLM) with logprobs=20. The
     student's top-20 IDs per position and their full-vocab softmax probs are
     the ``student_base`` distribution (free, from the sampling logprobs), and
     the 20 IDs are the shared candidate set for all 6 conditions.
  2. Re-score those same 20 IDs under 5 teacher conditions (HF forward, exact
     full-vocab softmax at those IDs, one sequence at a time for stability):
       expert_base       teacher + base prompt
       expert_chasing    teacher + chasing expert prompt
       expert_long_term  teacher + long_term expert prompt
       expert_repurchase teacher + repurchase expert prompt
       expert_generalized teacher + generalized expert prompt
  3. Label each response token: 1 = category-name token (inside a 兴趣 value),
     0 = reason/structure token.
  4. Emit one JSONL line per (prompt, response token position).

Selection: from data/train_1000.jsonl, pick N prompts per stratum
(buyer_type_cn), keeping only samples where ALL 4 experts have
evidence_available=true (so expert domain differences are visible). Defaults:
高价值消费者 x3 + 潜在用户 x3 = 6 prompts.

Usage (server, conda env verl):
  python scripts/expert_variance_probe.py \\
    --student-path /home/hechongchong/.jupyter/zhongzhengjia/www/basemodel/student \\
    --teacher-path /home/hechongchong/.jupyter/zhongzhengjia/www/basemodel/teacher \\
    --train-jsonl data/train_1000.jsonl \\
    --output scripts/expert_variance_probe.jsonl

GPU layout: student vLLM runs on GPU 0 (tensor_parallel_size=1); teacher HF
runs on GPU 1 (--teacher-gpu). Ensure both are free. Lower
--student-gpu-memory-utilization if GPU 0 is partially occupied.

Output JSONL schema (one line per token position):
  user_id, stratum, sample_idx, token_pos, token_id, token_text,
  label (1=category, 0=reason/structure), field (兴趣/理由/structure/parse_fail),
  top20_ids (shared 20 IDs), student_base, expert_base, expert_chasing,
  expert_long_term, expert_repurchase, expert_generalized
  *_probs are RAW full-vocab softmax probs at the 20 IDs (not top-20
  renormalized); downstream code does the top-k renorm + ANOVA.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

EXPERT_NAMES = ["chasing", "long_term", "repurchase", "generalized"]


# ---------------------------------------------------------------------------
# Prompt construction (mirrors verl/verl/trainer/ppo/expert_prompt_utils.py
# :build_expert_messages, inlined here so the probe is standalone and does
# not pull the training stack / torch at import time).
# ---------------------------------------------------------------------------
def build_expert_messages(
    raw_prompt: list[dict[str, str]],
    instruction: str,
    evidence_available: bool,
    evidence: object | None,
) -> list[dict[str, str]]:
    """Append the expert instruction (and hard evidence boundary if any) to a
    copy of the clean chat. Matches the training-time expert prompt exactly."""
    messages = copy.deepcopy(raw_prompt)
    addition = f"\n\n{instruction}"
    if evidence_available:
        evidence_text = json.dumps(evidence, ensure_ascii=False)
        addition += f"\n\n你的领域证据（候选类目）如下，预测只能从这些类目中选择：\n{evidence_text}"
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": addition.lstrip()})
    else:
        messages[-1]["content"] = messages[-1].get("content", "") + addition
    return messages


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render messages to a string with the assistant generation prompt.

    enable_thinking=False matches baseline/run_student_infer.py (Qwen3
    non-thinking mode). Falls back if the tokenizer does not support it.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------
def load_probe_prompts(
    train_jsonl: Path,
    strata: list[str],
    n_per_stratum: int,
) -> list[dict[str, Any]]:
    """Pick n_per_stratum prompts per stratum, each with all 4 experts' evidence."""
    by_stratum: dict[str, list[dict[str, Any]]] = {s: [] for s in strata}
    needed = {s: n_per_stratum for s in strata}
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            if all(len(by_stratum[s]) >= needed[s] for s in strata):
                break
            obj = json.loads(line)
            ei = obj.get("extra_info", {})
            stratum = ei.get("buyer_type_cn")
            if stratum not in by_stratum:
                continue
            if len(by_stratum[stratum]) >= n_per_stratum:
                continue
            ec = ei.get("expert_contexts", {})
            if not all(ec.get(k, {}).get("evidence_available", False) for k in EXPERT_NAMES):
                continue
            by_stratum[stratum].append({
                "user_id": ei.get("user_id", ""),
                "stratum": stratum,
                "raw_prompt": obj["prompt"],  # list[dict[str,str]]
                "expert_contexts": ec,
            })
    selected: list[dict[str, Any]] = []
    short: list[str] = []
    for s in strata:
        if len(by_stratum[s]) < n_per_stratum:
            short.append(f"{s}={len(by_stratum[s])}/{n_per_stratum}")
            continue
        for i, p in enumerate(by_stratum[s][:n_per_stratum]):
            p["sample_idx"] = i
            selected.append(p)
    if short:
        # Build a "what's available" hint so the user can pick strata that
        # actually have enough full-evidence samples without re-running a
        # separate diagnostic. Single full pass; cheap.
        avail: dict[str, int] = {}
        with open(train_jsonl, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                bt = obj.get("extra_info", {}).get("buyer_type_cn")
                if bt is None:
                    continue
                ec = obj.get("extra_info", {}).get("expert_contexts", {})
                if all(
                    ec.get(k, {}).get("evidence_available", False)
                    for k in EXPERT_NAMES
                ):
                    avail[bt] = avail.get(bt, 0) + 1
        hint = ", ".join(f"{k}={v}" for k, v in sorted(avail.items(), key=lambda kv: -kv[1]))
        raise ValueError(
            f"insufficient full-evidence samples for: {', '.join(short)}. "
            f"Available full-evidence counts per stratum: {hint}"
        )
    return selected


# ---------------------------------------------------------------------------
# Token labeling: 1 = category-name token, 0 = reason/structure token
# ---------------------------------------------------------------------------
_INTEREST_RE = re.compile(r'"兴趣"\s*:\s*"((?:[^"\\]|\\.)*)"')
_REASON_RE = re.compile(r'"理由"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _span_within(token_span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    ts, te = token_span
    for s, e in spans:
        if ts >= s and te <= e:
            return True
    return False


def label_response(tokenizer, r_text: str, r_ids: list[int]) -> list[tuple[int, str]]:
    """Return per-token (label, field) aligned to r_ids.

    label=1 for tokens whose span falls inside a 兴趣 value; 0 otherwise.
    field ∈ {兴趣, 理由, structure, parse_fail}. If the response is not valid
    JSON (no 兴趣/理由 values found), every token is tagged parse_fail.
    """
    enc = tokenizer(r_text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if ids != r_ids:
        print(
            f"[WARN] tokenizer re-tokenization differs from vLLM r_ids "
            f"(len {len(ids)} vs {len(r_ids)}); labels are best-effort",
            file=sys.stderr,
        )
    interest_spans = [m.span(1) for m in _INTEREST_RE.finditer(r_text)]
    reason_spans = [m.span(1) for m in _REASON_RE.finditer(r_text)]
    if not interest_spans and not reason_spans:
        return [(0, "parse_fail") for _ in r_ids]

    labels: list[tuple[int, str]] = []
    for i in range(len(r_ids)):
        if i < len(offsets):
            ts, te = offsets[i]
        else:
            ts, te = -1, -1
        if ts < 0 or te <= ts:
            labels.append((0, "structure"))
            continue
        if _span_within((ts, te), interest_spans):
            labels.append((1, "兴趣"))
        elif _span_within((ts, te), reason_spans):
            labels.append((0, "理由"))
        else:
            labels.append((0, "structure"))
    return labels


# ---------------------------------------------------------------------------
# Student sampling (vLLM)
# ---------------------------------------------------------------------------
def _logprob_value(lp: Any) -> float:
    """Extract the logprob float from a vLLM Logprob object or dict."""
    if hasattr(lp, "logprob"):
        return float(lp.logprob)
    if isinstance(lp, dict):
        return float(lp.get("logprob", lp.get("logprob")))
    return float(lp)


def sample_student(llm, tokenizer, prompts_text: list[str], max_tokens: int, seed: int):
    """Return list of (r_text, r_ids, top20_ids, top20_probs) per prompt.

    top20_ids[t] = student's top-20 token IDs at position t (shared candidate
    set). top20_probs[t] = full-vocab softmax prob at those IDs (raw, from
    vLLM logprobs which are full-vocab log-softmax, not top-k renormalized).
    """
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=0.1,
        top_p=0.90,
        top_k=20,
        max_tokens=max_tokens,
        logprobs=20,
        seed=seed,
    )
    outputs = llm.generate(prompts_text, sp)
    results = []
    for out in outputs:
        o = out.outputs[0]
        r_text = o.text
        r_ids = list(o.token_ids)
        top20_ids: list[list[int]] = []
        top20_probs: list[list[float]] = []
        for lp_dict in o.logprobs:
            items = [(int(tid), _logprob_value(lp)) for tid, lp in lp_dict.items()]
            items.sort(key=lambda kv: -kv[1])
            top20 = items[:20]
            top20_ids.append([tid for tid, _ in top20])
            top20_probs.append([math.exp(lp) for _, lp in top20])
        results.append((r_text, r_ids, top20_ids, top20_probs))
    return results


# ---------------------------------------------------------------------------
# Teacher scoring (HF forward, exact full-vocab softmax at the 20 IDs)
# ---------------------------------------------------------------------------
def teacher_forward_probs(
    model, tokenizer, prompt_text: str, r_ids: list[int],
    top20_ids_list: list[list[int]], device: str, max_model_len: int,
) -> list[list[float]]:
    """Forward prompt+R through the teacher; return per-position top-20
    full-vocab softmax probs (aligned to r_ids).

    logits[i] predicts token i+1, so r_ids[t] is predicted at position
    prompt_len + t - 1. We slice logits to those positions, run log_softmax
    over the full vocab (numerically stable), and index the 20 IDs. The full
    logits tensor is freed right after the slice; with sdpa attention +
    use_cache=False the transient memory is the model weights + a ~11GB
    logits buffer (bf16, 36k x 152k) which fits a 140GB GPU.
    """
    import torch

    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0].tolist()
    full_ids_list = prompt_ids + r_ids
    # Left-truncate the prompt if over budget; never drop response tokens.
    if len(full_ids_list) > max_model_len:
        overflow = len(full_ids_list) - max_model_len
        if overflow >= len(prompt_ids):
            raise ValueError(
                f"response alone ({len(r_ids)}) exceeds max_model_len ({max_model_len})"
            )
        prompt_ids = prompt_ids[overflow:]
        full_ids_list = prompt_ids + r_ids
    full_ids = torch.tensor([full_ids_list], dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    r_len = len(r_ids)
    with torch.no_grad():
        logits = model(full_ids, use_cache=False).logits[0]  # [seq, vocab]
    start = prompt_len - 1
    end = start + r_len
    pos_logits = logits[start:end].to(torch.float32)  # [r_len, vocab]
    del logits
    log_probs = torch.log_softmax(pos_logits, dim=-1)  # [r_len, vocab]
    out_probs: list[list[float]] = []
    for t in range(r_len):
        ids_t = torch.tensor(top20_ids_list[t], device=log_probs.device, dtype=torch.long)
        out_probs.append(log_probs[t, ids_t].exp().cpu().tolist())
    del pos_logits, log_probs
    return out_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--student-path", required=True, help="vLLM student model dir")
    p.add_argument("--teacher-path", required=True, help="HF teacher model dir")
    p.add_argument("--train-jsonl", default="data/train_1000.jsonl")
    p.add_argument("--output", default="scripts/expert_variance_probe.jsonl")
    p.add_argument(
        "--strata", default="核心高价值消费者,活跃消费者",
        help="comma-separated buyer_type_cn strata to sample from",
    )
    p.add_argument("--n-per-stratum", type=int, default=3)
    p.add_argument("--max-model-len", type=int, default=36864)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--student-gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--teacher-gpu", type=int, default=1)
    args = p.parse_args(argv)

    strata = [s.strip() for s in args.strata.split(",") if s.strip()]
    prompts = load_probe_prompts(Path(args.train_jsonl), strata, args.n_per_stratum)
    counts = ", ".join(
        f"{s}×{sum(1 for p in prompts if p['stratum'] == s)}" for s in strata
    )
    print(f"[probe] selected {len(prompts)} prompts: {counts}")

    # Student uses GPU 0 by default (vLLM TP=1 picks the first visible GPU).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student_path)
    base_prompt_texts = [render_chat(tokenizer, p["raw_prompt"]) for p in prompts]

    from vllm import LLM

    llm = LLM(
        model=args.student_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.student_gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
    )
    sampled = sample_student(
        llm, tokenizer, base_prompt_texts, args.max_tokens, args.seed
    )
    # Release vLLM; GPU 0 may stay occupied by worker procs until exit, so the
    # teacher goes on a different GPU (--teacher-gpu).
    del llm
    gc.collect()

    import torch
    from transformers import AutoModelForCausalLM

    torch.cuda.empty_cache()
    device = f"cuda:{args.teacher_gpu}"
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_path)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device).eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, p in enumerate(prompts):
            r_text, r_ids, top20_ids, student_base = sampled[i]
            print(
                f"[probe] prompt {i + 1}/{len(prompts)} user={p['user_id']} "
                f"stratum={p['stratum']} response_tokens={len(r_ids)}"
            )
            if not r_ids:
                print("[probe]   empty response, skipping")
                continue
            labels = label_response(tokenizer, r_text, r_ids)

            # Build the 5 teacher conditions: base prompt + 4 expert prompts.
            ec = p["expert_contexts"]
            conditions: list[tuple[str, str]] = [("expert_base", base_prompt_texts[i])]
            for name in EXPERT_NAMES:
                ctx = ec.get(name, {})
                instr = ctx.get("instruction", "")
                instr = instr.strip() if isinstance(instr, str) else ""
                if not instr:
                    raise ValueError(
                        f"user {p['user_id']} expert {name} missing instruction"
                    )
                msgs = build_expert_messages(
                    p["raw_prompt"], instr,
                    bool(ctx.get("evidence_available", False)),
                    ctx.get("evidence"),
                )
                conditions.append((f"expert_{name}", render_chat(teacher_tok, msgs)))

            # Forward each condition one at a time (stability over speed).
            cond_probs: dict[str, list[list[float]]] = {"student_base": student_base}
            for cond_name, cond_text in conditions:
                probs = teacher_forward_probs(
                    teacher, teacher_tok, cond_text, r_ids, top20_ids,
                    device, args.max_model_len,
                )
                cond_probs[cond_name] = probs
                print(f"[probe]   {cond_name}: scored {len(probs)} positions")

            for t in range(len(r_ids)):
                label, field = labels[t]
                row = {
                    "user_id": p["user_id"],
                    "stratum": p["stratum"],
                    "sample_idx": p["sample_idx"],
                    "token_pos": t,
                    "token_id": r_ids[t],
                    "token_text": tokenizer.decode([r_ids[t]]),
                    "label": label,
                    "field": field,
                    "top20_ids": top20_ids[t],
                    "student_base": student_base[t],
                    "expert_base": cond_probs["expert_base"][t],
                    "expert_chasing": cond_probs["expert_chasing"][t],
                    "expert_long_term": cond_probs["expert_long_term"][t],
                    "expert_repurchase": cond_probs["expert_repurchase"][t],
                    "expert_generalized": cond_probs["expert_generalized"][t],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_lines += 1
            fout.flush()
    print(f"[probe] wrote {n_lines} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
