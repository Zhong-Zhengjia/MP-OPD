import json
from rich import print
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sqlite3
from typing import Dict, List


def analysis_eval(file_path: str):
    final_res = {
        "passed_ids": [],
        "not_passed_ids": []
    }
    with open(file_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problem_id = obj["id"]
            acc_list = obj["acc_list"]
            if any(acc_list):
                final_res["passed_ids"].append(problem_id)
            else:
                final_res["not_passed_ids"].append(problem_id)

    return final_res



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


def get_acc(final_res):
    total = len(final_res["passed_ids"]) + len(final_res["not_passed_ids"])
    return len(final_res["passed_ids"]) / total if total > 0 else 0.0


def analysis_passed_ids(passed_ids_1, passed_ids_2):
    passed_ids_1_set = set(passed_ids_1)
    passed_ids_2_set = set(passed_ids_2)

    intersection_set = passed_ids_1_set & passed_ids_2_set
    only_1 = passed_ids_1_set - passed_ids_2_set
    only_2 = passed_ids_2_set - passed_ids_1_set
    union_set = passed_ids_1_set | passed_ids_2_set

    return intersection_set, only_1, only_2, union_set


def _annotate_segment(ax, x, bottom, height, text, color="black", fontsize=11, min_height=4):
    if height <= 0:
        return
    if height < min_height:
        ax.text(
            x, bottom + height + 1.2, text,
            ha="center", va="bottom", fontsize=fontsize, color=color
        )
    else:
        ax.text(
            x, bottom + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=color
        )


def plot_passed_comparison(
    passed_ids_1,
    passed_ids_2,
    label1="Model 1",
    label2="Model 2",
    save_path=None
):
    intersection_set, only_1, only_2, union_set = analysis_passed_ids(passed_ids_1, passed_ids_2)

    inter_cnt = len(intersection_set)
    only1_cnt = len(only_1)
    only2_cnt = len(only_2)
    union_cnt = len(union_set)

    COLOR_ONLY1 = "#A6BDD7"   # muted blue-gray
    COLOR_INTER = "#2B6C8A"   # deep teal-blue
    COLOR_ONLY2 = "#E7B3A7"   # muted rose
    COLOR_UNION = "#D9D9D9"   # light neutral gray
    EDGE = "#3A3A3A"
    GRID = "#D9D9D9"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = [0.00, 0.70, 1.40] 
    labels = [label1, label2, "Union"]
    width = 0.58

    # 第1列: only_1 + inter
    ax.bar(
        x[0], only1_cnt, width=width,
        color=COLOR_ONLY1, edgecolor=EDGE, linewidth=1.0, zorder=3
    )
    ax.bar(
        x[0], inter_cnt, width=width, bottom=only1_cnt,
        color=COLOR_INTER, edgecolor=EDGE, linewidth=1.0, zorder=3
    )

    ax.bar(
        x[1], only1_cnt, width=width,
        color="white", edgecolor="none", alpha=0.0, zorder=1
    )
    ax.bar(
        x[1], inter_cnt, width=width, bottom=only1_cnt,
        color=COLOR_INTER, edgecolor=EDGE, linewidth=1.0, zorder=3
    )
    ax.bar(
        x[1], only2_cnt, width=width, bottom=only1_cnt + inter_cnt,
        color=COLOR_ONLY2, edgecolor=EDGE, linewidth=1.0, zorder=3
    )

    # 第3列: union
    ax.bar(
        x[2], union_cnt, width=width,
        color=COLOR_UNION, edgecolor=EDGE, linewidth=1.0, zorder=3
    )

    # 分段标注
    _annotate_segment(ax, x[0], 0, only1_cnt, f"{only1_cnt}")
    _annotate_segment(ax, x[0], only1_cnt, inter_cnt, f"{inter_cnt}", color="white")

    _annotate_segment(ax, x[1], only1_cnt, inter_cnt, f"{inter_cnt}", color="white")
    _annotate_segment(ax, x[1], only1_cnt + inter_cnt, only2_cnt, f"{only2_cnt}")

    _annotate_segment(ax, x[2], 0, union_cnt, f"{union_cnt}")


    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Passed Problems")

    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=GRID, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_ylim(0, union_cnt * 1.12)

    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    legend_handles = [
        Patch(facecolor=COLOR_ONLY1, edgecolor=EDGE, label=f"Only in {label1}"),
        Patch(facecolor=COLOR_INTER, edgecolor=EDGE, label="Intersection"),
        Patch(facecolor=COLOR_ONLY2, edgecolor=EDGE, label=f"Only in {label2}"),
        Patch(facecolor=COLOR_UNION, edgecolor=EDGE, label="Union"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[green]Figure saved to:[/green] {save_path}")


if __name__ == "__main__":
    model_name_1 = "Qwen3-1.7B"
    model_name_2 = "Qwen3-4B"
    num_samples = 8 

    # fp_1 = f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name_1}_MathTestTotal_pass@{num_samples}.jsonl"
    # fp_2 = f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name_2}_MathTestTotal_pass@{num_samples}.jsonl"

    db_1 = f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name_1}_DeepMath-103K_pass@{num_samples}.sqlite"
    db_2 = f"/mnt/petrelfs/fudaocheng/codes/G-OPD/eval_outputs/{model_name_2}_DeepMath-103K_pass@{num_samples}.sqlite"

    # final_res_1 = analysis_eval(fp_1)
    # final_res_2 = analysis_eval(fp_2)

    final_res_1 = analysis_eval_sqlite(db_1)
    final_res_2 = analysis_eval_sqlite(db_2)

    plot_passed_comparison(
        final_res_1["passed_ids"],
        final_res_2["passed_ids"],
        label1=model_name_1,
        label2=model_name_2,
        save_path=f"/mnt/petrelfs/fudaocheng/codes/G-OPD/plots/{model_name_1}_{model_name_2}_DeepMath-103K_pass@{num_samples}.png"
    )