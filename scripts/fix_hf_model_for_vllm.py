#!/usr/bin/env python3
"""One-shot fixer: make an exported MP-OPD hf_model checkpoint loadable by vLLM.

The base student is a Qwen3.5-Embedding model, so the actor state_dict was
saved with ``language_model.`` prefixed keys: the embedding wrapper holds the
causal-LM backbone under a ``language_model`` submodule, and config.json's
architectures is ``Qwen3_5ForEmbedding``. vLLM's ``Qwen3_5ForCausalLM`` expects
the backbone under ``model.*`` plus a top-level ``lm_head.*``. The wrapper's
``language_model`` IS the ForCausalLM ``model`` submodule, so the prefix must
be *replaced* with ``model.``, not stripped — stripping leaves bare keys
(``embed_tokens.weight``) and vLLM then errors with
"no module or parameter named 'embed_tokens'".

This script rewrites an already-exported hf_model in place:
  1. Remap every weight key to the form vLLM expects:
       language_model.<backbone>.*  ->  model.<backbone>.*
       language_model.lm_head.*      ->  lm_head.*
       (bare ``<backbone>.*`` from a prior bad strip -> ``model.<backbone>.*``)
  2. Rewrite config.json architectures to the ``*ForCausalLM`` sibling.
  3. If no ``lm_head`` key is present, force tie_word_embeddings=true so the
     tied head is built from embed_tokens at load time.

Usage:
  python scripts/fix_hf_model_for_vllm.py <path-to-hf_model-dir>
  python scripts/fix_hf_model_for_vllm.py models/mpopd_*/global_step_99/hf_model

Idempotent: safe to re-run (no-op once keys are already in vLLM form). Makes a
backup of config.json on the first config change only.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def remap_vllm_key(k: str) -> str:
    """Map a checkpoint weight key to the key vLLM's Qwen3_5ForCausalLM expects.

    The embedding wrapper's ``language_model`` is the ForCausalLM ``model``
    submodule: the backbone (embed_tokens/layers/norm) sits directly under it,
    so ``language_model.`` becomes ``model.``. A wrapper that already nests
    ``model.`` or carries a real ``lm_head`` is handled by peeling only the
    wrapper prefix. A bare key produced by a prior bad strip (e.g.
    ``embed_tokens.weight``) is also prefixed with ``model.``.
    """
    if k.startswith("language_model."):
        inner = k[len("language_model."):]
        if inner.startswith("model.") or inner.startswith("lm_head"):
            return inner
        return "model." + inner
    # Bare backbone key from a prior strip: add the model. submodule prefix.
    # Only touch the known top-level backbone names so we never re-prefix an
    # already-correct model.* / lm_head.* key.
    if k.startswith(("embed_tokens.", "layers.", "norm.")) or k in ("embed_tokens.weight", "norm.weight"):
        return "model." + k
    return k


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

    # Detect whether any key still needs remapping, and whether lm_head exists.
    needs_rewrite = False
    has_lm_head = False
    sample_keys: list[str] = []
    for st in st_files:
        with safe_open(st, framework="pt") as f:
            keys = list(f.keys())
            if not sample_keys:
                sample_keys = keys[:5]
            for k in keys:
                if remap_vllm_key(k) != k:
                    needs_rewrite = True
                nk = remap_vllm_key(k)
                if nk == "lm_head.weight" or nk.startswith("lm_head."):
                    has_lm_head = True
    print(f"[scan] sample keys: {sample_keys}")
    print(f"[scan] remapped sample: {[remap_vllm_key(k) for k in sample_keys]}")
    print(f"[scan] needs key remap: {needs_rewrite}")
    print(f"[scan] has lm_head: {has_lm_head}")

    # Rewrite config.json: architectures -> *ForCausalLM, tie if no lm_head.
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
        else:
            print(f"[config] architectures already: {archs}")
        if not has_lm_head:
            if cfg.get("tie_word_embeddings") is None:
                cfg["tie_word_embeddings"] = True
                cfg_changed = True
                print("[config] tie_word_embeddings: (unset) -> true (no lm_head in weights)")
            else:
                print(f"[config] tie_word_embeddings already: {cfg.get('tie_word_embeddings')}")
        if cfg_changed:
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(f"[ERR] no config.json in {hf_dir}")
        return 1

    # Rewrite safetensors keys via the remap.
    if needs_rewrite:
        for st in st_files:
            new_tensors: dict[str, object] = {}
            with safe_open(st, framework="pt") as f:
                for k in f.keys():
                    new_tensors[remap_vllm_key(k)] = f.get_tensor(k)
            # save to a temp then replace (safetensors can't read+write same file)
            tmp = st.with_suffix(".safetensors.tmp")
            save_file(new_tensors, str(tmp), metadata={"format": "pt"})
            tmp.replace(st)
            print(f"[weights] remapped keys in {st.name} ({len(new_tensors)} tensors)")
    else:
        print("[weights] all keys already in vLLM form, nothing to remap")

    print("[done] hf_model is now loadable by vLLM as a causal-LM generator")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
