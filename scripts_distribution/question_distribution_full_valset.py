import json
import math
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


def get_accuracy_dict_steps_train_model_sqlite(train_model_db_path):
    acc_by_step = {}

    conn = sqlite3.connect(train_model_db_path)
    try:
        cursor = conn.execute(
            """
            SELECT
                global_step,
                question_id,
                AVG(CAST(reward AS REAL)) AS accuracy
            FROM validation_results
            GROUP BY global_step, question_id
            ORDER BY global_step, question_id
            """
        )

        for global_step, question_id, accuracy in cursor:
            acc_by_step.setdefault(global_step, {})[question_id] = accuracy

    finally:
        conn.close()

    return acc_by_step


def build_ordered_bins(
    acc_dict_1,
    acc_dict_2,
    shared_threshold=0.75,
    shared_ratio=0.25,
    left_side="model2",
    bin_size=5,
    sort_mode="minmax",
):
    left_ids, center_ids, right_ids = split_problem_ids(
        acc_dict_1,
        acc_dict_2,
        shared_threshold=shared_threshold,
        shared_ratio=shared_ratio,
        left_side=left_side,
        sort_mode=sort_mode,
    )

    n_left, n_center, n_right = auto_region_bins(
        left_ids,
        center_ids,
        right_ids,
        bin_size=bin_size,
    )

    def make_bins(ids, n_bins):
        if len(ids) == 0 or n_bins == 0:
            return []
        chunks = np.array_split(np.array(ids, dtype=object), n_bins)
        return [list(chunk) for chunk in chunks]

    left_bins = make_bins(left_ids, n_left)
    center_bins = make_bins(center_ids, n_center)
    right_bins = make_bins(right_ids, n_right)

    ordered_bins = left_bins + center_bins + right_bins

    left_end = len(left_bins)
    center_end = len(left_bins) + len(center_bins)

    return ordered_bins, left_end, center_end


def accuracy_curve_from_ordered_bins(acc_dict, ordered_bins):
    y = []

    for ids in ordered_bins:
        values = [acc_dict[x] for x in ids if x in acc_dict]
        if len(values) == 0:
            y.append(np.nan)
        else:
            y.append(float(np.mean(values)))

    return np.array(y)


def plot_train_model_step_distribution(
    step_acc_dict,
    ordered_bins,
    left_end,
    center_end,
    output_folder,
    train_label="Train model",
    base_acc_dict_1=None,
    base_acc_dict_2=None,
    base_label_1="Qwen3-4B",
    base_label_2="Qwen3-1.7B",
    save_prefix="train_model",
):
    from pathlib import Path

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(ordered_bins))
    if len(x) == 0:
        raise ValueError("ordered_bins is empty")

    red = "#E99A9A"
    blue = "#89BBD5"
    green = "#5DAE8B"

    base_y1 = None
    base_y2 = None

    if base_acc_dict_1 is not None:
        base_y1 = accuracy_curve_from_ordered_bins(base_acc_dict_1, ordered_bins)

    if base_acc_dict_2 is not None:
        base_y2 = accuracy_curve_from_ordered_bins(base_acc_dict_2, ordered_bins)

    save_paths = []

    for global_step in sorted(step_acc_dict):
        y = accuracy_curve_from_ordered_bins(step_acc_dict[global_step], ordered_bins)

        fig, ax = plt.subplots(figsize=(17, 9.2))

        base_handles = []
        base_labels = []

        if base_y1 is not None:
            h1, = plt.plot(
                x,
                base_y1,
                color=red,
                linewidth=3,
                alpha=0.8,
                label=base_label_1,
            )
            base_handles.append(h1)
            base_labels.append(base_label_1)

        if base_y2 is not None:
            h2, = plt.plot(
                x,
                base_y2,
                color=blue,
                linewidth=3,
                alpha=0.8,
                label=base_label_2,
            )
            base_handles.append(h2)
            base_labels.append(base_label_2)

        train_legend_label = f"{train_label} step {global_step}"

        h_train, = plt.plot(
            x,
            y,
            color=green,
            linewidth=5,
            label=train_legend_label,
        )

        plt.fill_between(x, 0, y, color=green, alpha=0.22)

        if 0 < left_end < len(x):
            plt.axvline(left_end - 0.5, color="gray", linewidth=1.5, alpha=0.25)

        if 0 < center_end < len(x):
            plt.axvline(center_end - 0.5, color="gray", linewidth=1.5, alpha=0.25)

        plt.ylabel("Binned accuracy", fontsize=32, fontweight="bold")
        plt.ylim(-0.03, 1.10)

        if len(x) == 1:
            plt.xlim(-0.5, 0.5)
        else:
            plt.xlim(0, len(x) - 1)

        tick_positions = [
            0,
            max(0, left_end - 1),
            max(0, center_end - 1),
            len(x) - 1,
        ]
        tick_labels = ["low", "high", "high", "low"]

        plt.xticks(
            tick_positions,
            tick_labels,
            fontsize=22,
            fontweight="bold",
        )
        plt.yticks(fontsize=22, fontweight="bold")

        plt.grid(axis="y", alpha=0.28, linewidth=1.2)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

        legend_artists = []

        if base_handles:
            leg1 = fig.legend(
                base_handles,
                base_labels,
                fontsize=20,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=len(base_handles),
                handlelength=3.0,
                columnspacing=2.0,
                borderaxespad=0.0,
            )
            legend_artists.append(leg1)

        leg2 = fig.legend(
            [h_train],
            [train_legend_label],
            fontsize=20,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=1,
            handlelength=3.0,
            borderaxespad=0.0,
        )
        legend_artists.append(leg2)

        fig.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.09,
            right=0.98,
        )

        save_path = output_folder / f"{save_prefix}_global_step_{global_step:03d}.png"

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.15,
            bbox_extra_artists=legend_artists,
        )

        plt.close(fig)

        save_paths.append(save_path)

    return save_paths


def get_accuracy_dict_sqlite(db_path):
    acc_dict = {}

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT
                question_id,
                AVG(CAST(acc AS REAL)) AS accuracy
            FROM generations
            WHERE acc IS NOT NULL
            GROUP BY question_id
            """
        )

        for question_id, accuracy in cursor:
            acc_dict[question_id] = accuracy

    finally:
        conn.close()

    return acc_dict


def get_accuracy_dict(jsonl_path):
    acc_dict = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            acc_list = item["acc_list"]
            acc_dict[item["id"]] = sum(acc_list) / len(acc_list)

    return acc_dict


def split_problem_ids(
    acc_dict_1,
    acc_dict_2,
    shared_threshold=0.75,
    shared_ratio=0.25,
    left_side="model2",
    sort_mode="minmax",
):
    ids = list(set(acc_dict_1) & set(acc_dict_2))

    if shared_threshold is None:
        ids_by_shared = sorted(
            ids,
            key=lambda x: (
                min(acc_dict_1[x], acc_dict_2[x]),
                acc_dict_1[x] + acc_dict_2[x],
            ),
            reverse=True,
        )
        k = max(1, int(len(ids_by_shared) * shared_ratio))
        center_ids = set(ids_by_shared[:k])
    else:
        center_ids = {
            x for x in ids
            if min(acc_dict_1[x], acc_dict_2[x]) >= shared_threshold
        }

    rest_ids = [x for x in ids if x not in center_ids]

    if left_side == "model1":
        left_ids = [x for x in rest_ids if acc_dict_1[x] >= acc_dict_2[x]]
        right_ids = [x for x in rest_ids if acc_dict_2[x] > acc_dict_1[x]]
    else:
        left_ids = [x for x in rest_ids if acc_dict_2[x] >= acc_dict_1[x]]
        right_ids = [x for x in rest_ids if acc_dict_1[x] > acc_dict_2[x]]

    def key_minmax(x):
        a1 = acc_dict_1[x]
        a2 = acc_dict_2[x]
        return (
            min(a1, a2),
            max(a1, a2),
        )

    def key_mean(x):
        a1 = acc_dict_1[x]
        a2 = acc_dict_2[x]
        return (
            (a1 + a2) / 2,
            min(a1, a2),
            max(a1, a2),
        )

    def key_dominant_left(x):
        a1 = acc_dict_1[x]
        a2 = acc_dict_2[x]

        if left_side == "model1":
            return (
                a1,
                a2,
                a1 - a2,
            )
        else:
            return (
                a2,
                a1,
                a2 - a1,
            )

    def key_dominant_right(x):
        a1 = acc_dict_1[x]
        a2 = acc_dict_2[x]

        if left_side == "model1":
            return (
                a2,
                a1,
                a2 - a1,
            )
        else:
            return (
                a1,
                a2,
                a1 - a2,
            )

    if sort_mode == "dominant":
        left_ids = sorted(left_ids, key=key_dominant_left)
        right_ids = sorted(right_ids, key=key_dominant_right, reverse=True)

    elif sort_mode == "mean":
        left_ids = sorted(left_ids, key=key_mean)
        right_ids = sorted(right_ids, key=key_mean, reverse=True)

    elif sort_mode == "minmax":
        left_ids = sorted(left_ids, key=key_minmax)
        right_ids = sorted(right_ids, key=key_minmax, reverse=True)

    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode}")

    center_ids = sorted(
        center_ids,
        key=lambda x: acc_dict_1[x] - acc_dict_2[x],
        reverse=(left_side == "model1"),
    )

    return left_ids, center_ids, right_ids


def bin_region(ids, acc_dict_1, acc_dict_2, n_bins):
    if len(ids) == 0:
        return [], []

    n_bins = min(n_bins, len(ids))
    chunks = np.array_split(np.array(ids, dtype=object), n_bins)

    y1 = [np.mean([acc_dict_1[x] for x in chunk]) for chunk in chunks]
    y2 = [np.mean([acc_dict_2[x] for x in chunk]) for chunk in chunks]

    return y1, y2


def auto_region_bins(left_ids, center_ids, right_ids, bin_size=7):
    def n_bins(ids):
        if len(ids) == 0:
            return 0
        return max(1, int(round(len(ids) / bin_size)))

    return (
        n_bins(left_ids),
        n_bins(center_ids),
        n_bins(right_ids),
    )


def build_binned_curves(
    acc_dict_1,
    acc_dict_2,
    shared_threshold=0.75,
    shared_ratio=0.25,
    left_side="model2",
    bin_size=5,
    sort_mode="minmax",
):
    left_ids, center_ids, right_ids = split_problem_ids(
        acc_dict_1,
        acc_dict_2,
        shared_threshold=shared_threshold,
        shared_ratio=shared_ratio,
        left_side=left_side,
        sort_mode=sort_mode,
    )

    region_bins = auto_region_bins(left_ids, center_ids, right_ids, bin_size)

    y1_left, y2_left = bin_region(left_ids, acc_dict_1, acc_dict_2, region_bins[0])
    y1_center, y2_center = bin_region(center_ids, acc_dict_1, acc_dict_2, region_bins[1])
    y1_right, y2_right = bin_region(right_ids, acc_dict_1, acc_dict_2, region_bins[2])

    y1 = np.array(y1_left + y1_center + y1_right)
    y2 = np.array(y2_left + y2_center + y2_right)
    x = np.arange(len(y1))

    left_end = len(y1_left)
    center_end = len(y1_left) + len(y1_center)

    ordered_ids = left_ids + center_ids + right_ids

    return x, y1, y2, ordered_ids, left_end, center_end


def plot_accuracy_distribution(
    acc_dict_1,
    acc_dict_2,
    label_1="Qwen3-1.7B",
    label_2="Qwen3-4B",
    save_path=None,
    shared_threshold=0.75,
    shared_ratio=0.25,
    left_side="model2",
    bin_size=5,
    sort_mode="minmax",
):
    x, y1, y2, ordered_ids, left_end, center_end = build_binned_curves(
        acc_dict_1,
        acc_dict_2,
        shared_threshold=shared_threshold,
        shared_ratio=shared_ratio,
        left_side=left_side,
        bin_size=bin_size,
        sort_mode=sort_mode,
    )

    red = "#E99A9A"
    blue = "#89BBD5"

    plt.figure(figsize=(15, 6))

    plt.plot(x, y1, color=red, linewidth=5, label=label_1)
    plt.plot(x, y2, color=blue, linewidth=5, label=label_2)

    plt.fill_between(x, 0, y1, color=red, alpha=0.18)
    plt.fill_between(x, 0, y2, color=blue, alpha=0.22)

    plt.axvline(left_end - 0.5, color="gray", linewidth=1.5, alpha=0.25)
    plt.axvline(center_end - 0.5, color="gray", linewidth=1.5, alpha=0.25)

    plt.ylabel("Binned accuracy", fontsize=32, fontweight="bold")
    plt.ylim(-0.03, 1.14)
    plt.xlim(0, len(x) - 1)

    plt.xticks(
        [0, max(0, left_end - 1), max(0, center_end - 1), len(x) - 1],
        ["low", "high", "high", "low"],
        fontsize=22,
        fontweight="bold",
    )
    plt.yticks(fontsize=22, fontweight="bold")

    plt.grid(axis="y", alpha=0.28, linewidth=1.2)
    plt.legend(fontsize=22, frameon=False, loc="upper right")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return ordered_ids


def compare_two_jsonl_distributions(
    acc_dict_1,
    acc_dict_2,
    label_1="Qwen3-1.7B",
    label_2="Qwen3-4B",
    save_path="accuracy_distribution.png",
    shared_threshold=0.75,
    shared_ratio=0.25,
    left_side="model2",
    bin_size=5,
    sort_mode='minmax',
):
    ordered_ids = plot_accuracy_distribution(
        acc_dict_1,
        acc_dict_2,
        label_1=label_1,
        label_2=label_2,
        save_path=save_path,
        shared_threshold=shared_threshold,
        shared_ratio=shared_ratio,
        left_side=left_side,
        bin_size=bin_size,
        sort_mode=sort_mode,
    )

    return acc_dict_1, acc_dict_2, ordered_ids

if __name__ == '__main__':
    from pathlib import Path

    num_samples = 16

    model_name_list = ['DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen3-4B', 'Skywork-OR1-Math-7B']
    model_name_1 = 'DeepSeek-R1-Distill-Qwen-1.5B'
    model_name_2 = 'DeepSeek-R1-Distill-Qwen-7B'

    val_1000_db_1 = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/datasets/"
        "G-OPD-Training-Data/DeepMath-103K/"
        f"val_1000_{model_name_1}_pass@{num_samples}.sqlite"
    )

    val_1000_db_2 = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/datasets/"
        "G-OPD-Training-Data/DeepMath-103K/"
        f"val_1000_{model_name_2}_pass@{num_samples}.sqlite"
    )

    acc_dict_1 = get_accuracy_dict_sqlite(val_1000_db_1)
    acc_dict_2 = get_accuracy_dict_sqlite(val_1000_db_2)

    acc_dict_1, acc_dict_2, ordered_ids = compare_two_jsonl_distributions(
        acc_dict_1,
        acc_dict_2,
        label_1=model_name_1,
        label_2=model_name_2,
        save_path=f"plots_distribution/question_distribution/val_1000_{model_name_1}_{model_name_2}_pass@{num_samples}.png",
        bin_size=10,
        sort_mode="mean",   # minmax, mean, dominant
     )

    # train_model_name = "Qwen3-1.7B-T4B-Math_GB1024_OB0"
    # date_time = "0602_10"

    # train_model_db = Path(
    #     "/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/LP_offline_distribution/"
    #     f"LP_{train_model_name}_{date_time}/"
    #     "validation_results.db"
    # )

    # output_folder = f"plots_distribution/question_distribution/{train_model_name}_{date_time}/"

    # ordered_bins, left_end, center_end = build_ordered_bins(
    #     acc_dict_1,
    #     acc_dict_2,
    #     shared_threshold=0.75,
    #     shared_ratio=0.25,
    #     left_side="model2",
    #     bin_size=10,
    #     sort_mode="mean",
    # )

    # train_step_acc_dict = get_accuracy_dict_steps_train_model_sqlite(train_model_db)

    # plot_train_model_step_distribution(
    #     train_step_acc_dict,
    #     ordered_bins,
    #     left_end,
    #     center_end,
    #     output_folder=output_folder,
    #     train_label=train_model_name,
    #     base_acc_dict_1=acc_dict_1,
    #     base_acc_dict_2=acc_dict_2,
    #     base_label_1="Qwen3-4B",
    #     base_label_2="Qwen3-1.7B",
    #     save_prefix=train_model_name,
    # )