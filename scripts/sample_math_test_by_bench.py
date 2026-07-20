#!/usr/bin/env python3
"""Randomly sample rows from each benchmark in MathTestTotal/test.parquet."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = (
    "/mnt/phwfile/datafrontier/fudaocheng/datasets/G-OPD-Training-Data/MathTestTotal/test.parquet"
)
BENCH_COL = "data_source"


def compute_sample_size(total: int, ratio: float) -> int:
    if ratio <= 0.0 or total <= 0:
        return 0
    return min(total, int(math.ceil(ratio * total)))


def sample_by_bench(
    df: pd.DataFrame,
    default_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if BENCH_COL not in df.columns:
        raise ValueError(f"missing required column '{BENCH_COL}'")

    sampled_parts: list[pd.DataFrame] = []
    for _, group in df.groupby(BENCH_COL, sort=True):
        sample_size = compute_sample_size(len(group), default_ratio)
        if sample_size > 0:
            sampled_parts.append(group.sample(n=sample_size, random_state=seed, replace=False))

    if not sampled_parts:
        return df.iloc[0:0].copy()
    return pd.concat(sampled_parts, ignore_index=True)


def print_sampled_stats(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    default_ratio: float,
) -> None:
    original_counts = original_df[BENCH_COL].value_counts().sort_index()
    sampled_counts = sampled_df[BENCH_COL].value_counts().sort_index()

    print("\n=== sampled dataset stats ===")
    print(f"default_ratio: {default_ratio:.4f}")
    print(f"{'bench':<14} {'original':>10} {'sampled':>10} {'ratio':>10}")
    print("-" * 48)

    original_total = 0
    sampled_total = 0
    for bench in original_counts.index:
        orig_n = int(original_counts[bench])
        samp_n = int(sampled_counts.get(bench, 0))
        bench_ratio = samp_n / orig_n if orig_n > 0 else 0.0
        print(f"{bench:<14} {orig_n:>10d} {samp_n:>10d} {bench_ratio:>10.4f}")
        original_total += orig_n
        sampled_total += samp_n

    overall_ratio = sampled_total / original_total if original_total > 0 else 0.0
    print("-" * 48)
    print(f"{'ALL':<14} {original_total:>10d} {sampled_total:>10d} {overall_ratio:>10.4f}")
    print(f"columns: {list(sampled_df.columns)}")
    if "ability" in sampled_df.columns:
        print("ability counts:")
        print(sampled_df["ability"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample a fixed ratio from each benchmark in test.parquet."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="input parquet path")
    parser.add_argument("--output", default="data/opd/training/g_opd/MathTestTotal/test-sampled_10pct.parquet", help="output parquet path")
    parser.add_argument("--default-ratio", type=float, default=0.1, help="sampling ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    if not 0.0 <= args.default_ratio <= 1.0:
        parser.error("--default-ratio must be in [0, 1]")

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        ratio_pct = int(round(args.default_ratio * 100))
        output_path = input_path.with_name(
            f"{input_path.stem}-sampled_{ratio_pct}pct{input_path.suffix}"
        )
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    df = pd.read_parquet(input_path)
    sampled_df = sample_by_bench(df, default_ratio=args.default_ratio, seed=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_parquet(output_path, index=False)

    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"seed:   {args.seed}")
    print_sampled_stats(df, sampled_df, args.default_ratio)


if __name__ == "__main__":
    main()
