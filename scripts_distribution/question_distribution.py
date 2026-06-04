import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple, Sequence, Optional, Union
import numpy as np
import matplotlib.pyplot as plt


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


def analysis_passed_ids(passed_ids_1, passed_ids_2):
    passed_ids_1_set = set(passed_ids_1)
    passed_ids_2_set = set(passed_ids_2)

    intersection_set = passed_ids_1_set & passed_ids_2_set
    only_1 = passed_ids_1_set - passed_ids_2_set
    only_2 = passed_ids_2_set - passed_ids_1_set
    union_set = passed_ids_1_set | passed_ids_2_set

    return intersection_set, only_1, only_2, union_set


def analysis_train_eval_sqlite_by_step(
    db_path: Union[str, Path]
) -> Tuple[List[int], Dict[int, Set[str]]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT global_step
            FROM validation_results
            WHERE global_step IS NOT NULL
            ORDER BY global_step
            """
        )
        steps = [int(row[0]) for row in cursor.fetchall()]
        passed_ids_by_step = {step: set() for step in steps}
        cursor.execute(
            """
            SELECT
                global_step,
                question_id,
                MAX(COALESCE(reward, 0)) AS max_reward
            FROM validation_results
            WHERE global_step IS NOT NULL
            GROUP BY global_step, question_id
            HAVING max_reward = 1
            ORDER BY global_step
            """
        )
        for global_step, question_id, _ in cursor.fetchall():
            passed_ids_by_step[int(global_step)].add(question_id)
    finally:
        conn.close()
    return steps, passed_ids_by_step


def analysis_train_distribution_by_step(
    steps: Sequence[int],
    train_passed_ids_by_step: Dict[int, Set[str]],
    inter_set: Set[str],
    Qwen3_4B_only_set: Set[str],
    Qwen3_1_7B_only_set: Set[str],
) -> np.ndarray:
    inter_set = set(inter_set)
    Qwen3_4B_only_set = set(Qwen3_4B_only_set)
    Qwen3_1_7B_only_set = set(Qwen3_1_7B_only_set)
    dist = np.zeros((len(steps), 3), dtype=float)
    for i, step in enumerate(steps):
        passed_ids = set(train_passed_ids_by_step.get(step, set()))
        denom = len(passed_ids)
        if denom == 0:
            continue
        dist[i, 0] = len(passed_ids & Qwen3_1_7B_only_set) / len(Qwen3_1_7B_only_set) * 100
        dist[i, 1] = len(passed_ids & inter_set) / len(inter_set) * 100
        dist[i, 2] = len(passed_ids & Qwen3_4B_only_set) / len(Qwen3_4B_only_set) * 100
    return dist


def plot_train_distribution_by_step(
    steps: Sequence[int],
    dist: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
):
    steps = list(steps)
    dist = np.asarray(dist)

    if dist.shape[1] != 3:
        raise ValueError(f"dist should have shape [num_steps, 3], but got {dist.shape}")

    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 26,
        "axes.titlesize": 28,
        "xtick.labelsize": 17,
        "ytick.labelsize": 22,
        "legend.fontsize": 19,
        "axes.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # 放大图片宽度，steps 越多图越宽
    fig_width = max(18, len(steps) * 0.9)
    fig_height = 8.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    x = np.arange(len(steps))
    width = 0.26

    colors = [
        "#A9CFE7",
        "#E8A6A1",
        "#F3C58A",
    ]

    labels = [
        "Qwen3-1.7B only",
        "Intersection",
        "Qwen3-4B only",
    ]

    bar_containers = []

    for i in range(3):
        bars = ax.bar(
            x + (i - 1) * width,
            dist[:, i],
            width=width,
            color=colors[i],
            edgecolor="white",
            linewidth=1.2,
            label=labels[i],
        )
        bar_containers.append(bars)

    # 在每个柱子上标注百分比
    for bars in bar_containers:
        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.2,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                rotation=90,
                color="black",
                clip_on=False,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

    ax.set_xlabel("Global step", labelpad=12)
    ax.set_ylabel("Percentage (%)", labelpad=12)

    if title is not None:
        ax.set_title(title, pad=58)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right")

    # 给柱顶文字留空间
    ax.set_ylim(0, 110)

    ax.grid(axis="y", color="#D9D9D9", linewidth=1.1, alpha=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        columnspacing=1.8,
        handlelength=1.8,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight")

    plt.close(fig)

if __name__ == '__main__':
    from pathlib import Path

    Qwen3_4B_val_mini_db = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/datasets/"
        "G-OPD-Training-Data/DeepMath-103K/"
        "val_union_mini_1000_Qwen3-4B_pass@8.sqlite"
    )

    Qwen3_1_7B_val_mini_db = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/datasets/"
        "G-OPD-Training-Data/DeepMath-103K/"
        "val_union_mini_1000_Qwen3-1.7B_pass@8.sqlite"
    )

    Qwen3_4B_res = analysis_eval_sqlite(Qwen3_4B_val_mini_db)
    Qwen3_1_7B_res = analysis_eval_sqlite(Qwen3_1_7B_val_mini_db)

    inter_set, Qwen3_4B_only_set, Qwen3_1_7B_only_set, union_set = analysis_passed_ids(
        Qwen3_4B_res['passed_ids'],
        Qwen3_1_7B_res['passed_ids']
    )

    print("Qwen3-4B passed question ids:", len(Qwen3_4B_res['passed_ids']))
    print("Qwen3-1.7B passed question ids:", len(Qwen3_1_7B_res['passed_ids']))
    print("Intersection:", len(inter_set))
    print("Qwen3-4B only:", len(Qwen3_4B_only_set))
    print("Qwen3-1.7B only:", len(Qwen3_1_7B_only_set))
    print("Union:", len(union_set))

    train_model_name = "Qwen3-1.7B-T4B-Math_GB1024_OB0"
    date_time="0529_23"

    train_model_db = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/LP_offline_distribution/"
        f"LP_{train_model_name}_{date_time}/"
        "validation_results.db"
    )

    steps, train_passed_ids_by_step = analysis_train_eval_sqlite_by_step(train_model_db)

    print(steps)
    dist = analysis_train_distribution_by_step(
        steps=steps,
        train_passed_ids_by_step=train_passed_ids_by_step,
        inter_set=inter_set,
        Qwen3_4B_only_set=Qwen3_4B_only_set,
        Qwen3_1_7B_only_set=Qwen3_1_7B_only_set,
    )

    plot_train_distribution_by_step(
        steps,
        dist,
        save_path=f"plots_distribution/{train_model_name}_question_distribution.png",
        title=f"{train_model_name} Question Distribution",
    )