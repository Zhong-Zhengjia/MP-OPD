import json
import numpy as np
import matplotlib.pyplot as plt


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

        left_ids = sorted(
            left_ids,
            key=lambda x: (
                acc_dict_1[x],
                acc_dict_2[x],
                acc_dict_1[x] - acc_dict_2[x],
            ),
        )

        center_ids = sorted(
            center_ids,
            key=lambda x: acc_dict_1[x] - acc_dict_2[x],
            reverse=True,
        )

        right_ids = sorted(
            right_ids,
            key=lambda x: (
                acc_dict_2[x],
                acc_dict_1[x],
                acc_dict_2[x] - acc_dict_1[x],
            ),
            reverse=True,
        )

    else:
        left_ids = [x for x in rest_ids if acc_dict_2[x] >= acc_dict_1[x]]
        right_ids = [x for x in rest_ids if acc_dict_1[x] > acc_dict_2[x]]

        left_ids = sorted(
            left_ids,
            key=lambda x: (
                acc_dict_2[x],
                acc_dict_1[x],
                acc_dict_2[x] - acc_dict_1[x],
            ),
        )

        center_ids = sorted(
            center_ids,
            key=lambda x: acc_dict_1[x] - acc_dict_2[x],
        )

        right_ids = sorted(
            right_ids,
            key=lambda x: (
                acc_dict_1[x],
                acc_dict_2[x],
                acc_dict_1[x] - acc_dict_2[x],
            ),
            reverse=True,
        )

    return left_ids, list(center_ids), right_ids


def bin_region(ids, acc_dict_1, acc_dict_2, n_bins):
    if len(ids) == 0:
        return [], []

    n_bins = min(n_bins, len(ids))
    chunks = np.array_split(np.array(ids, dtype=object), n_bins)

    y1 = [np.mean([acc_dict_1[x] for x in chunk]) for chunk in chunks]
    y2 = [np.mean([acc_dict_2[x] for x in chunk]) for chunk in chunks]

    return y1, y2


def build_binned_curves(
    acc_dict_1,
    acc_dict_2,
    shared_threshold=0.75,
    shared_ratio=0.25,
    left_side="model2",
    region_bins=(16, 10, 16),
):
    left_ids, center_ids, right_ids = split_problem_ids(
        acc_dict_1,
        acc_dict_2,
        shared_threshold=shared_threshold,
        shared_ratio=shared_ratio,
        left_side=left_side,
    )

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
    region_bins=(14, 10, 14),
):
    x, y1, y2, ordered_ids, left_end, center_end = build_binned_curves(
        acc_dict_1,
        acc_dict_2,
        shared_threshold=shared_threshold,
        shared_ratio=shared_ratio,
        left_side=left_side,
        region_bins=region_bins,
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

    top_y = 1.08

    # if left_side == "model2":
    #     left_title = f"{label_2} side"
    #     right_title = f"{label_1} side"
    #     left_color = blue
    #     right_color = red
    # else:
    #     left_title = f"{label_1} side"
    #     right_title = f"{label_2} side"
    #     left_color = red
    #     right_color = blue

    # plt.text(
    #     max(0, left_end / 2 - 0.5),
    #     top_y,
    #     left_title,
    #     ha="center",
    #     va="bottom",
    #     fontsize=26,
    #     fontweight="bold",
    #     color=left_color,
    # )

    # plt.text(
    #     (left_end + center_end) / 2 - 0.5,
    #     top_y,
    #     "shared high-accuracy region",
    #     ha="center",
    #     va="bottom",
    #     fontsize=26,
    #     fontweight="bold",
    #     color="#444444",
    # )

    # plt.text(
    #     (center_end + len(x)) / 2 - 0.5,
    #     top_y,
    #     right_title,
    #     ha="center",
    #     va="bottom",
    #     fontsize=26,
    #     fontweight="bold",
    #     color=right_color,
    # )

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
    file_path_1,
    file_path_2,
    label_1="Qwen3-1.7B",
    label_2="Qwen3-4B",
    save_path="accuracy_distribution.png",
    shared_threshold=0.75,
    shared_ratio=0.25,
    left_side="model2",
    region_bins=(14, 10, 14),
):
    acc_dict_1 = get_accuracy_dict(file_path_1)
    acc_dict_2 = get_accuracy_dict(file_path_2)

    ordered_ids = plot_accuracy_distribution(
        acc_dict_1,
        acc_dict_2,
        label_1=label_1,
        label_2=label_2,
        save_path=save_path,
        shared_threshold=shared_threshold,
        shared_ratio=shared_ratio,
        left_side=left_side,
        region_bins=region_bins,
    )

    return acc_dict_1, acc_dict_2, ordered_ids

if __name__ == '__main__':
    file_path_1 = "../G-OPD/eval_outputs/Qwen3-1.7B_MathTestTotal_pass@16.jsonl"
    file_path_2 = "../G-OPD/eval_outputs/Qwen3-4B_MathTestTotal_pass@16.jsonl"

    acc_dict_1, acc_dict_2, ordered_ids = compare_two_jsonl_distributions(
        file_path_1,
        file_path_2,
        label_1="Qwen3-1.7B",
        label_2="Qwen3-4B",
        save_path="plots_distribution/accuracy_distribution_testset.png",
    )