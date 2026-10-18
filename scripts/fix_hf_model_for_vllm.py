#!/usr/bin/env python3
"""One-shot fixer: make an exported MP-OPD hf_model checkpoint loadable by vLLM.

The base student is a Qwen3.5-Embedding model, so the actor state_dict was
saved with ``language_model.`` prefixed keys (the embedding wrapper holds the
causal-LM backbone under a ``language_model`` submodule), and config.json's
architectures is ``Qwen3_5ForEmbedding``. vLLM's Qwen3.5 causal-LM model
expects bare ``model.*`` + ``lm_head.*`` keys, so loading the raw export fails
with "no module or parameter named 'language_model'".

This script rewrites an already-exported hf_model in place:
  1. Strip the ``language_model.`` prefix from every safetensors weight key.
  2. Rewrite config.json architectures to the ``*ForCausalLM`` sibling.
  3. If no ``lm_head`` key is present, force tie_word_embeddings=true so the
     tied head is built from embed_tokens at load time.

Usage:
  python scripts/fix_hf_model_for_vllm.py <path-to-hf_model-dir>
  python scripts/fix_hf_model_for_vllm.py models/mpopd_*/global_step_99/hf_model

Idempotent: safe to re-run (no-op once keys are already stripped). Makes a
backup of config.json on the first run only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def fix_config(hf_dir: Path) -> bool:
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        print(f"[ERR] no config.json in {hf_dir}")
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    changed = False
    if arch.endswith("ForEmbedding"):
        new_arch = arch[: -len("ForEmbedding")] + "ForCausalLM"
        backup = cfg_path.with_suffix(".json.bak")
        if not backup.exists():
            shutil.copy2(cfg_path, backup)
        cfg["architectures"] = [new_arch]
        changed = True
        print(f"[config] architectures: {arch} -> {new_arch}")
    # tie_word_embeddings: only force-true if there is no lm_head in the weights
    return changed, cfg, arch


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    hf_dir = Path(argv[1]).expanduser().resolve()
    if not hf_dir.is_dir():
        print(f"[ERR] not a directory: {hf_dir}")
        return 1

    st_files = sorted(hf_dir.glob("model*.safetensors"))
    if not st_files:
        print(f"[ERR] no model*.safetensors in {hf_dir}")
        return 1

    # Check whether any key still carries the language_model. prefix.
    needs_strip = False
    sample_keys: list[str] = []
    for st in st_files:
        with safe_open(st, framework="pt") as f:
            keys = f.keys()
            sample_keys = list(keys)[:5]
            if any(k.startswith("language_model.") for k in keys):
                needs_strip = True
                break
    print(f"[scan] sample keys: {sample_keys}")
    print(f"[scan] needs language_model. strip: {needs_strip}")

    has_lm_head = False
    for st in st_files:
        with safe_open(st, framework="pt") as f:
            if any(k == "lm_head.weight" or k.startswith("lm_head.") for k in f.keys()):
                has_lm_head = True
                break
    print(f"[scan] has lm_head: {has_lm_head}")

    # Rewrite config.json
    cfg_path = hf_dir / "config.json"
    cfg_changed = False
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        archs = cfg.get("architectures") or []
        arch = archs[0] if archs else ""
        if arch.endswith("ForEmbedding"):
            new_arch = arch[: -len("ForEmbedding")] + "ForCausalLM"
            backup = cfg_path.with_suffix(".json.bak")
            if not backup.exists():
                shutil.copy2(cfg_path, backup)
            cfg["architectures"] = [new_arch]
            cfg_changed = True
            print(f"[config] architectures: {arch} -> {new_arch}")
        if not has_lm_head and cfg.get("tie_word_embeddings") is None:
            cfg["tie_word_embeddings"] = True
            cfg_changed = True
            print("[config] tie_word_embeddings: (unset) -> true (no lm_head in weights)")
        if cfg_changed:
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Rewrite safetensors keys: strip language_model. prefix
    if needs_strip:
        for st in st_files:
            new_tensors: dict[str, object] = {}
            with safe_open(st, framework="pt") as f:
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    nk = k[len("language_model."):] if k.startswith("language_model.") else k
                    new_tensors[nk] = tensor
            # save to a temp then replace (safetensors can't read+write same file)
            tmp = st.with_suffix(".safetensors.tmp")
            save_file(new_tensors, str(tmp), metadata={"format": "pt"})
            tmp.replace(st)
            print(f"[weights] stripped language_model. prefix in {st.name} ({len(new_tensors)} tensors)")
    else:
        print("[weights] no language_model. prefix found, nothing to strip")

    print("[done] hf_model is now loadable by vLLM as a causal-LM generator")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
