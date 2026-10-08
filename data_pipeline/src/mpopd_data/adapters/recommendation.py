"""Adapter for the current recommendation split JSONL format."""

from __future__ import annotations

from typing import Any

from mpopd_data.prompts import extract_instruction, render_clean_prompt
from mpopd_data.schema import EXPERT_NAMES, validate_mpopd_row

SOURCE_EVIDENCE_FIELDS = {
    "chasing": "ground_truth_chasing",
    "long_term": "ground_truth_long_term",
    "repurchase": "ground_truth_repurchase",
    "generalized": "ground_truth_generalized",
}


def adapt_recommendation_row(
    source: dict[str, Any],
    *,
    clean_template: str,
    expert_templates: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(source.get("profile_llm"), dict):
        raise ValueError("profile_llm must be an object")
    if not isinstance(source.get("ground_truth"), dict):
        raise ValueError("ground_truth must be an object")

    contexts = {}
    for name in EXPERT_NAMES:
        evidence = source.get(SOURCE_EVIDENCE_FIELDS[name], {})
        if not isinstance(evidence, dict):
            raise ValueError(f"{SOURCE_EVIDENCE_FIELDS[name]} must be an object")
        contexts[name] = {
            "enabled": True,
            "instruction": extract_instruction(expert_templates[name]),
            "evidence_available": bool(evidence),
            "evidence": evidence,
        }

    row = {
        "data_source": "recommendation",
        "prompt": [{"role": "user", "content": render_clean_prompt(clean_template, source["profile_llm"])}],
        "ability": "Recommendation",
        "reward_model": {"ground_truth": source["ground_truth"]},
        "extra_info": {
            "schema_version": "mpopd.v1",
            "user_id": source.get("user_id"),
            "buyer_type_cn": source.get("ubuyereffectivetype"),
            "expert_contexts": contexts,
        },
    }
    return validate_mpopd_row(row).to_dict()
