import os
import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)

def wrap_unprocessed_dataset(updf: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = updf.copy()

    required_cols = {"problem_idx", "problem", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name}: missing columns {missing}")

    df["problem_idx"] = df["problem_idx"].astype(int)

    has_problem_type = "problem_type" in df.columns

    def build_prompt(problem: str):
        suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        return [{"role": "user", "content": f"{problem}{suffix}"}]

    def build_reward_model(ans):
        return {"ground_truth": str(ans), "style": "rule"}

    def build_extra_info(row):
        info = {"index": int(row["problem_idx"]), "split": "test"}
        if has_problem_type:
            info["problem_type"] = row["problem_type"]
        return info

    out = pd.DataFrame({
        "id": df["problem_idx"].apply(lambda i: f"{dataset_name}_{int(i)}"),
        "data_source": dataset_name,
        "prompt": df["problem"].apply(build_prompt),
        "ability": "math",
        "reward_model": df["answer"].apply(build_reward_model),
        "extra_info": df.apply(build_extra_info, axis=1),
    })
    return out


process_datasets_info = [
    {
        "name": "AIME2024",
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/AIME2024/test.parquet",
    },
    {
        "name": "AIME2025",
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/AIME2025/test.parquet",
    },
]

unprocessed_dataset_info = [
    {
        "name": 'AIME2026',
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/MathArena/aime_2026/data/train-00000-of-00001.parquet"
    },
    {
        "name": "CMIMC2025",
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/MathArena/cmimc_2025/data/train-00000-of-00001.parquet"
    },
    {
        "name": "HMMT2025FEB",
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/MathArena/hmmt_feb_2025/data/train-00000-of-00001.parquet"
    },
    {
        "name": "HMMT2026FEB",
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/MathArena/hmmt_feb_2026/data/train-00000-of-00001.parquet"
    },
    {
        "name": "HMMT2025NOV",
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/MathArena/hmmt_nov_2025/data/train-00000-of-00001.parquet"
    },
    {
        "name": "SMT2025",
        "file_path": "/mnt/petrelfs/fudaocheng/datasets/MathArena/smt_2025/data/train-00000-of-00001.parquet"
    },
]