import numpy as np
import pandas as pd
import sqlite3
from rich import print

from typing import Dict, List, Set

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)


original_data = '/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet'
model_1_sqlite = 'eval_outputs/Qwen3-1.7B_DeepMath-103K_pass@8.sqlite'
model_2_sqlite = 'eval_outputs/Qwen3-4B_DeepMath-103K_pass@8.sqlite'

output_train_path = '/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/train_union_passed_70.parquet'
output_val_path = '/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/val_union_passed_30.parquet'
output_val_mini_path = '/mnt/petrelfs/fudaocheng/datasets/G-OPD-Training-Data/DeepMath-103K/val_union_mini_100.parquet'


def analysis_eval_sqlite(db_path: str) -> Dict[str, List[str]]:
    final_res = {
        "passed_ids": [],
        "not_passed_ids": []
    }

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                question_id,
                MAX(COALESCE(acc, 0)) AS max_acc
            FROM generations
            GROUP BY question_id
        """)

        rows = cursor.fetchall()

        for question_id, max_acc in rows:
            if max_acc == 1:
                final_res["passed_ids"].append(question_id)
            else:
                final_res["not_passed_ids"].append(question_id)

    finally:
        conn.close()

    return final_res


def extract_index_from_question_id(question_id: str) -> int:
    try:
        return int(question_id.rsplit("_", 1)[-1])
    except Exception as e:
        raise ValueError(f"can not parse index from question_id: {question_id}") from e


def main():
    model_1_res = analysis_eval_sqlite(model_1_sqlite)
    model_2_res = analysis_eval_sqlite(model_2_sqlite)

    model_1_passed_ids = set(model_1_res["passed_ids"])
    model_2_passed_ids = set(model_2_res["passed_ids"])


    union_passed_ids = model_1_passed_ids | model_2_passed_ids
    print(f"[green]model_1 passed:[/green] {len(model_1_passed_ids)}")
    print(f"[green]model_2 passed:[/green] {len(model_2_passed_ids)}")
    print(f"[green]union passed:[/green] {len(union_passed_ids)}")


    selected_indices: Set[int] = {extract_index_from_question_id(qid) for qid in union_passed_ids}
    print(f"[cyan]selected indices:[/cyan] {len(selected_indices)}")


    df = pd.read_parquet(original_data)
    print(f"[yellow]original data size:[/yellow] {len(df)}")


    df["index"] = df["extra_info"].apply(lambda x: x["index"] if isinstance(x, dict) and "index" in x else None)


    selected_df = df[df["index"].isin(selected_indices)].copy()
    print(f"[magenta]selected data size:[/magenta] {len(selected_df)}")

    matched_indices = set(selected_df["index"].tolist())
    missing_indices = selected_indices - matched_indices
    print(f"[red]missing indices:[/red] {len(missing_indices)}")
    if len(missing_indices) > 0:
        print(f"[red]example missing indices:[/red] {list(sorted(missing_indices))[:20]}")

    selected_df = selected_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    split_idx = int(len(selected_df) * 0.7)
    train_df = selected_df.iloc[:split_idx].copy()
    val_df = selected_df.iloc[split_idx:].copy()

    print(f"[blue]train size:[/blue] {len(train_df)}")
    print(f"[blue]val size:[/blue] {len(val_df)}")

    train_df = train_df.drop(columns=["index"])
    val_df = val_df.drop(columns=["index"])

    train_df.to_parquet(output_train_path, index=False)
    val_df.to_parquet(output_val_path, index=False)

    print(f"[bold green]train parquet saved to:[/bold green] {output_train_path}")
    print(f"[bold green]val parquet saved to:[/bold green] {output_val_path}")


def build_mini_validation_set(validation_set_path: str, mini_validation_set_path: str, sample_size: int = 100, random_state: int = 42):
    val_df = pd.read_parquet(validation_set_path)
    print(f"[yellow]original validation size:[/yellow] {len(val_df)}")

    if len(val_df) < sample_size:
        raise ValueError(f"validation set size {len(val_df)} is smaller than sample_size {sample_size}")

    mini_val_df = val_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    print(f"[blue]mini validation size:[/blue] {len(mini_val_df)}")

    mini_val_df.to_parquet(mini_validation_set_path, index=False)
    print(f"[bold green]mini validation parquet saved to:[/bold green] {mini_validation_set_path}")


if __name__ == "__main__":
    # main()
    build_mini_validation_set(output_val_path, output_val_mini_path)