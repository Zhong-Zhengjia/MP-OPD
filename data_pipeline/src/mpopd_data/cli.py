"""Command line entry points for the MP-OPD data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpopd_data.adapters.recommendation import adapt_recommendation_row
from mpopd_data.converters.jsonl_to_jsonl import convert_jsonl
from mpopd_data.prompts import load_clean_template, load_expert_templates
from mpopd_data.schema import EXPERT_NAMES


def recommendation_adapter(args):
    clean_template = load_clean_template(args.clean_template)
    expert_templates = load_expert_templates(args.expert_prompt_dir, EXPERT_NAMES)
    return lambda row: adapt_recommendation_row(
        row,
        clean_template=clean_template,
        expert_templates=expert_templates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="mpopd-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    recommendation = subparsers.add_parser("recommendation")
    recommendation.add_argument("--input", type=Path, required=True)
    recommendation.add_argument("--output", type=Path, required=True)
    recommendation.add_argument("--clean-template", type=Path, required=True)
    recommendation.add_argument("--expert-prompt-dir", type=Path, required=True)
    recommendation.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.command == "recommendation":
        adapter = recommendation_adapter(args)
        if args.validate_only:
            count = 0
            with args.input.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        adapter(json.loads(line))
                        count += 1
            print(f"validated {count} records")
        else:
            count = convert_jsonl(args.input, args.output, adapter)
            print(f"wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
