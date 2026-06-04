import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def count_token_ids_in_texts(tokenizer, texts, vocab_size, batch_size=512):
    counts = np.zeros(vocab_size, dtype=np.int64)
    total = 0

    for i in range(0, len(texts), batch_size):
        batch = [x if x is not None else "" for x in texts[i:i + batch_size]]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        ids = [t for seq in encoded for t in seq]
        if ids:
            arr = np.asarray(ids, dtype=np.int64)
            counts += np.bincount(arr, minlength=vocab_size)[:vocab_size]
            total += int(arr.size)

    return counts, total


def get_base_token_freq_sqlite(db_path, tokenizer, vocab_size, batch_size=512, sqlite_batch_size=4096):
    counts = np.zeros(vocab_size, dtype=np.int64)
    total = 0

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT response
            FROM generations
            WHERE response IS NOT NULL
            """
        )

        while True:
            rows = cursor.fetchmany(sqlite_batch_size)
            if not rows:
                break

            texts = [r[0] for r in rows]
            c, n = count_token_ids_in_texts(
                tokenizer,
                texts,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )
            counts += c
            total += n

    finally:
        conn.close()

    if total == 0:
        return np.zeros(vocab_size, dtype=np.float64), counts, total

    return counts.astype(np.float64) / float(total), counts, total


def get_train_token_freq_steps_sqlite(db_path, tokenizer, vocab_size, batch_size=512, sqlite_batch_size=4096):
    counts_by_step = {}
    total_by_step = {}

    def flush(step, texts):
        if step is None or not texts:
            return
        c, n = count_token_ids_in_texts(
            tokenizer,
            texts,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )
        counts_by_step.setdefault(step, np.zeros(vocab_size, dtype=np.int64))
        total_by_step.setdefault(step, 0)
        counts_by_step[step] += c
        total_by_step[step] += n

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT global_step, response
            FROM validation_results
            WHERE response IS NOT NULL
            ORDER BY global_step
            """
        )

        current_step = None
        buffer = []

        while True:
            rows = cursor.fetchmany(sqlite_batch_size)
            if not rows:
                break

            for global_step, response in rows:
                if current_step is None:
                    current_step = global_step

                if global_step != current_step:
                    flush(current_step, buffer)
                    current_step = global_step
                    buffer = []

                buffer.append(response)

        flush(current_step, buffer)

    finally:
        conn.close()

    freq_by_step = {}
    for step in counts_by_step:
        total = total_by_step[step]
        if total == 0:
            freq_by_step[step] = np.zeros(vocab_size, dtype=np.float64)
        else:
            freq_by_step[step] = counts_by_step[step].astype(np.float64) / float(total)

    return freq_by_step, counts_by_step, total_by_step


def smoothed_freq_from_counts(counts, total, smoothing=0.5):
    vocab_size = len(counts)
    return (counts.astype(np.float64) + float(smoothing)) / (float(total) + float(smoothing) * float(vocab_size))


def select_top_log_ratio_tokens(
    counts_4b,
    total_4b,
    counts_1_7b,
    total_1_7b,
    top_k=3000,
    min_base_count=20,
    smoothing=0.5,
    excluded_token_ids=None,
):
    p4 = smoothed_freq_from_counts(counts_4b, total_4b, smoothing=smoothing)
    p17 = smoothed_freq_from_counts(counts_1_7b, total_1_7b, smoothing=smoothing)

    log_gap_4b_minus_1_7b = np.log10(p4) - np.log10(p17)
    abs_gap = np.abs(log_gap_4b_minus_1_7b)

    valid = (counts_4b + counts_1_7b) >= int(min_base_count)
    valid &= np.isfinite(abs_gap)

    if excluded_token_ids is not None:
        for token_id in excluded_token_ids:
            if 0 <= int(token_id) < len(valid):
                valid[int(token_id)] = False

    candidate_ids = np.where(valid)[0]
    if len(candidate_ids) == 0:
        raise ValueError("No valid tokens after filtering")

    top_k = min(int(top_k), len(candidate_ids))
    candidate_scores = abs_gap[candidate_ids]
    selected_local = np.argpartition(-candidate_scores, top_k - 1)[:top_k]
    top_ids = candidate_ids[selected_local]

    left_ids = [int(x) for x in top_ids if log_gap_4b_minus_1_7b[x] < 0]
    right_ids = [int(x) for x in top_ids if log_gap_4b_minus_1_7b[x] >= 0]

    left_ids = sorted(left_ids, key=lambda x: abs_gap[x])
    right_ids = sorted(right_ids, key=lambda x: abs_gap[x], reverse=True)

    return left_ids, right_ids, left_ids + right_ids, log_gap_4b_minus_1_7b


def make_bins(ids, bin_size=30):
    if len(ids) == 0:
        return []
    n_bins = max(1, int(round(len(ids) / bin_size)))
    n_bins = min(n_bins, len(ids))
    chunks = np.array_split(np.asarray(ids, dtype=np.int64), n_bins)
    return [chunk.astype(np.int64) for chunk in chunks]


def build_ordered_token_bins_from_counts(
    counts_4b,
    total_4b,
    counts_1_7b,
    total_1_7b,
    top_k=3000,
    bin_size=30,
    min_base_count=20,
    smoothing=0.5,
    excluded_token_ids=None,
):
    left_ids, right_ids, ordered_ids, log_gap = select_top_log_ratio_tokens(
        counts_4b=counts_4b,
        total_4b=total_4b,
        counts_1_7b=counts_1_7b,
        total_1_7b=total_1_7b,
        top_k=top_k,
        min_base_count=min_base_count,
        smoothing=smoothing,
        excluded_token_ids=excluded_token_ids,
    )

    left_bins = make_bins(left_ids, bin_size=bin_size)
    right_bins = make_bins(right_ids, bin_size=bin_size)

    ordered_bins = left_bins + right_bins
    left_end = len(left_bins)

    return ordered_bins, ordered_ids, left_ids, right_ids, left_end, log_gap


def binned_log_freq_from_counts(counts, total, ordered_bins, smoothing=0.5):
    p = smoothed_freq_from_counts(counts, total, smoothing=smoothing)
    logp = np.log10(p)
    y = []
    for ids in ordered_bins:
        if len(ids) == 0:
            y.append(np.nan)
        else:
            y.append(float(np.nanmean(logp[ids])))
    return np.asarray(y, dtype=np.float64)


def binned_freq_from_counts(counts, total, ordered_bins, smoothing=0.5):
    p = smoothed_freq_from_counts(counts, total, smoothing=smoothing)
    y = []
    for ids in ordered_bins:
        if len(ids) == 0:
            y.append(np.nan)
        else:
            y.append(float(np.nanmean(p[ids])))
    return np.asarray(y, dtype=np.float64)


def contrast_from_log_curves(y, y4, y17, gap_floor_dex=0.03, clip=3.0):
    mid = 0.5 * (y4 + y17)
    gap = y4 - y17
    direction = np.sign(gap)
    direction[direction == 0] = 1.0
    denom = np.maximum(np.abs(gap) * 0.5, float(gap_floor_dex) * 0.5)
    c = ((y - mid) * direction) / denom
    return np.clip(c, -float(clip), float(clip))


def apply_bin_axis_ticks(ax, n_bins, left_end, fontsize=20):
    tick_positions = []
    tick_labels = []

    if left_end > 0:
        tick_positions.extend([0, left_end - 1])
        tick_labels.extend(["1.7B high", "1.7B very high"])

    if left_end < n_bins:
        tick_positions.extend([left_end, n_bins - 1])
        tick_labels.extend(["4B very high", "4B high"])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=fontsize, fontweight="bold")


def save_selected_tokens(
    tokenizer,
    save_path,
    ordered_ids,
    counts_4b,
    total_4b,
    counts_1_7b,
    total_1_7b,
    log_gap_4b_minus_1_7b,
    smoothing=0.5,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    p4 = smoothed_freq_from_counts(counts_4b, total_4b, smoothing=smoothing)
    p17 = smoothed_freq_from_counts(counts_1_7b, total_1_7b, smoothing=smoothing)

    with open(save_path, "w", encoding="utf-8") as f:
        for rank, token_id in enumerate(ordered_ids):
            token_id = int(token_id)
            item = {
                "rank": int(rank),
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "count_qwen3_4b": int(counts_4b[token_id]),
                "count_qwen3_1_7b": int(counts_1_7b[token_id]),
                "freq_qwen3_4b_smoothed": float(p4[token_id]),
                "freq_qwen3_1_7b_smoothed": float(p17[token_id]),
                "log10_freq_qwen3_4b": float(np.log10(p4[token_id])),
                "log10_freq_qwen3_1_7b": float(np.log10(p17[token_id])),
                "log10_gap_4b_minus_1_7b": float(log_gap_4b_minus_1_7b[token_id]),
                "abs_log10_gap": float(abs(log_gap_4b_minus_1_7b[token_id])),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def plot_base_token_distribution_enhanced(
    counts_4b,
    total_4b,
    counts_1_7b,
    total_1_7b,
    ordered_bins,
    left_end,
    save_path,
    label_4b="Qwen3-4B",
    label_1_7b="Qwen3-1.7B",
    smoothing=0.5,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(ordered_bins))
    y4 = binned_log_freq_from_counts(counts_4b, total_4b, ordered_bins, smoothing=smoothing)
    y17 = binned_log_freq_from_counts(counts_1_7b, total_1_7b, ordered_bins, smoothing=smoothing)
    gap = y4 - y17

    red = "#E99A9A"
    blue = "#89BBD5"
    gray = "#AAB3C2"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        gridspec_kw={"height_ratios": [1.15, 0.85], "hspace": 0.12},
        sharex=True,
    )

    ax = axes[0]
    ax.fill_between(x, y4, y17, color=gray, alpha=0.22)
    ax.plot(x, y4, color=red, linewidth=4.5, label=label_4b)
    ax.plot(x, y17, color=blue, linewidth=4.5, label=label_1_7b)

    if 0 < left_end < len(x):
        ax.axvline(left_end - 0.5, color="gray", linewidth=1.6, alpha=0.35)

    y_min = float(np.nanmin([np.nanmin(y4), np.nanmin(y17)]))
    y_max = float(np.nanmax([np.nanmax(y4), np.nanmax(y17)]))
    margin = (y_max - y_min) * 0.08 if y_max > y_min else 0.1

    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_ylabel("log10 binned\nfrequency", fontsize=28, fontweight="bold")
    ax.tick_params(axis="y", labelsize=21)
    ax.grid(axis="y", alpha=0.25, linewidth=1.1)
    ax.legend(fontsize=22, frameon=False, loc="upper right")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax2 = axes[1]
    ax2.axhline(0.0, color="black", linewidth=1.4, alpha=0.45)
    ax2.fill_between(x, 0.0, gap, where=gap >= 0, color=red, alpha=0.32, interpolate=True)
    ax2.fill_between(x, 0.0, gap, where=gap < 0, color=blue, alpha=0.32, interpolate=True)
    ax2.plot(x, gap, color="black", linewidth=2.2, alpha=0.75)

    if 0 < left_end < len(x):
        ax2.axvline(left_end - 0.5, color="gray", linewidth=1.6, alpha=0.35)

    gap_abs = float(np.nanmax(np.abs(gap)))
    gap_margin = gap_abs * 0.18 if gap_abs > 0 else 0.05
    ax2.set_ylim(-gap_abs - gap_margin, gap_abs + gap_margin)
    ax2.set_ylabel("base log gap\n4B - 1.7B", fontsize=25, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=19)
    ax2.grid(axis="y", alpha=0.25, linewidth=1.1)

    if len(x) == 1:
        ax2.set_xlim(-0.5, 0.5)
    else:
        ax2.set_xlim(0, len(x) - 1)

    apply_bin_axis_ticks(ax2, len(x), left_end, fontsize=18)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_linewidth(2.2)
        a.spines["bottom"].set_linewidth(2.2)
        for tick in a.get_yticklabels():
            tick.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_train_model_step_contrast_distribution(
    train_counts_by_step,
    train_total_by_step,
    ordered_bins,
    left_end,
    output_folder,
    counts_4b,
    total_4b,
    counts_1_7b,
    total_1_7b,
    train_label="Train model",
    label_4b="Qwen3-4B",
    label_1_7b="Qwen3-1.7B",
    save_prefix="train_model",
    smoothing=0.5,
    gap_floor_dex=0.03,
    contrast_clip=3.0,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(ordered_bins))
    if len(x) == 0:
        raise ValueError("ordered_bins is empty")

    red = "#E99A9A"
    blue = "#89BBD5"
    green = "#5DAE8B"
    gray = "#AAB3C2"

    base_y4 = binned_log_freq_from_counts(counts_4b, total_4b, ordered_bins, smoothing=smoothing)
    base_y17 = binned_log_freq_from_counts(counts_1_7b, total_1_7b, ordered_bins, smoothing=smoothing)
    base_c4 = contrast_from_log_curves(base_y4, base_y4, base_y17, gap_floor_dex=gap_floor_dex, clip=contrast_clip)
    base_c17 = contrast_from_log_curves(base_y17, base_y4, base_y17, gap_floor_dex=gap_floor_dex, clip=contrast_clip)

    save_paths = []

    for global_step in sorted(train_counts_by_step):
        train_y = binned_log_freq_from_counts(
            train_counts_by_step[global_step],
            train_total_by_step[global_step],
            ordered_bins,
            smoothing=smoothing,
        )
        train_c = contrast_from_log_curves(
            train_y,
            base_y4,
            base_y17,
            gap_floor_dex=gap_floor_dex,
            clip=contrast_clip,
        )

        clipped_progress = (np.clip(train_c, -1.0, 1.0) + 1.0) * 0.5
        mean_progress = float(np.nanmean(clipped_progress))

        all_y = np.concatenate([base_y4, base_y17, train_y])
        y_min = float(np.nanmin(all_y))
        y_max = float(np.nanmax(all_y))
        margin = (y_max - y_min) * 0.08 if y_max > y_min else 0.1

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(18, 10.5),
            gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.10},
            sharex=True,
        )

        ax = axes[0]
        ax.fill_between(x, base_y4, base_y17, color=gray, alpha=0.20)
        h4, = ax.plot(x, base_y4, color=red, linewidth=3.0, alpha=0.78, label=label_4b)
        h17, = ax.plot(x, base_y17, color=blue, linewidth=3.0, alpha=0.78, label=label_1_7b)
        ht, = ax.plot(
            x,
            train_y,
            color=green,
            linewidth=5.2,
            label=f"{train_label} step {global_step}",
        )

        if 0 < left_end < len(x):
            ax.axvline(left_end - 0.5, color="gray", linewidth=1.5, alpha=0.35)

        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_ylabel("log10 binned\nfrequency", fontsize=27, fontweight="bold")
        ax.tick_params(axis="y", labelsize=20)
        ax.grid(axis="y", alpha=0.25, linewidth=1.1)

        ax.legend(
            handles=[h4, h17, ht],
            fontsize=20,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=3,
            handlelength=3.0,
            columnspacing=1.6,
        )

        ax2 = axes[1]
        ax2.axhspan(0.0, contrast_clip, color=red, alpha=0.055)
        ax2.axhspan(-contrast_clip, 0.0, color=blue, alpha=0.055)
        ax2.axhline(1.0, color=red, linewidth=2.0, alpha=0.45)
        ax2.axhline(-1.0, color=blue, linewidth=2.0, alpha=0.45)
        ax2.axhline(0.0, color="black", linewidth=1.5, alpha=0.45)

        ax2.plot(x, base_c4, color=red, linewidth=2.4, alpha=0.45)
        ax2.plot(x, base_c17, color=blue, linewidth=2.4, alpha=0.45)
        ax2.fill_between(x, 0.0, train_c, color=green, alpha=0.18)
        ax2.plot(x, train_c, color=green, linewidth=5.0)

        if 0 < left_end < len(x):
            ax2.axvline(left_end - 0.5, color="gray", linewidth=1.5, alpha=0.35)

        ax2.text(
            0.012,
            0.92,
            f"mean clipped progress to 4B = {mean_progress:.3f}",
            transform=ax2.transAxes,
            fontsize=19,
            fontweight="bold",
            ha="left",
            va="top",
        )

        ax2.set_ylim(-contrast_clip, contrast_clip)
        ax2.set_ylabel("base-normalized\ncontrast", fontsize=27, fontweight="bold")
        ax2.tick_params(axis="y", labelsize=20)
        ax2.grid(axis="y", alpha=0.25, linewidth=1.1)

        if len(x) == 1:
            ax2.set_xlim(-0.5, 0.5)
        else:
            ax2.set_xlim(0, len(x) - 1)

        apply_bin_axis_ticks(ax2, len(x), left_end, fontsize=18)

        ax2.text(
            1.004,
            0.84,
            "toward\n4B",
            transform=ax2.transAxes,
            color=red,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax2.text(
            1.004,
            0.16,
            "toward\n1.7B",
            transform=ax2.transAxes,
            color=blue,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="center",
        )

        for a in axes:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["left"].set_linewidth(2.3)
            a.spines["bottom"].set_linewidth(2.3)
            for tick in a.get_yticklabels():
                tick.set_fontweight("bold")

        fig.subplots_adjust(top=0.88, bottom=0.11, left=0.08, right=0.95)

        save_path = output_folder / f"{save_prefix}_contrast_global_step_{int(global_step):03d}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
        plt.close(fig)
        save_paths.append(save_path)

    return save_paths


def plot_train_contrast_heatmap(
    train_counts_by_step,
    train_total_by_step,
    ordered_bins,
    left_end,
    output_folder,
    counts_4b,
    total_4b,
    counts_1_7b,
    total_1_7b,
    train_label="Train model",
    label_4b="Qwen3-4B",
    label_1_7b="Qwen3-1.7B",
    save_name="train_contrast_heatmap.png",
    smoothing=0.5,
    gap_floor_dex=0.03,
    contrast_clip=2.0,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    steps = sorted(train_counts_by_step)
    if not steps:
        return None

    x = np.arange(len(ordered_bins))

    red = "#E99A9A"
    blue = "#89BBD5"

    base_y4 = binned_log_freq_from_counts(counts_4b, total_4b, ordered_bins, smoothing=smoothing)
    base_y17 = binned_log_freq_from_counts(counts_1_7b, total_1_7b, ordered_bins, smoothing=smoothing)

    mat = []
    progress = []
    for step in steps:
        train_y = binned_log_freq_from_counts(
            train_counts_by_step[step],
            train_total_by_step[step],
            ordered_bins,
            smoothing=smoothing,
        )
        train_c = contrast_from_log_curves(
            train_y,
            base_y4,
            base_y17,
            gap_floor_dex=gap_floor_dex,
            clip=contrast_clip,
        )
        mat.append(train_c)
        progress.append(float(np.nanmean((np.clip(train_c, -1.0, 1.0) + 1.0) * 0.5)))

    mat = np.asarray(mat, dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        gridspec_kw={"height_ratios": [0.75, 1.25], "hspace": 0.10},
        sharex=True,
    )

    ax = axes[0]
    ax.fill_between(x, base_y4, base_y17, color="#AAB3C2", alpha=0.22)
    ax.plot(x, base_y4, color=red, linewidth=3.2, label=label_4b)
    ax.plot(x, base_y17, color=blue, linewidth=3.2, label=label_1_7b)

    if 0 < left_end < len(x):
        ax.axvline(left_end - 0.5, color="gray", linewidth=1.5, alpha=0.35)

    y_min = float(np.nanmin([np.nanmin(base_y4), np.nanmin(base_y17)]))
    y_max = float(np.nanmax([np.nanmax(base_y4), np.nanmax(base_y17)]))
    margin = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_ylabel("base log10\nfrequency", fontsize=24, fontweight="bold")
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", alpha=0.25, linewidth=1.1)
    ax.legend(fontsize=20, frameon=False, loc="upper right")

    ax2 = axes[1]
    im = ax2.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-contrast_clip,
        vmax=contrast_clip,
        extent=(-0.5, len(x) - 0.5, len(steps) - 0.5, -0.5),
    )

    if 0 < left_end < len(x):
        ax2.axvline(left_end - 0.5, color="black", linewidth=1.2, alpha=0.35)

    if len(steps) <= 20:
        ytick_positions = np.arange(len(steps))
        ytick_labels = [f"{int(s)}" for s in steps]
    else:
        ytick_positions = np.linspace(0, len(steps) - 1, 10).round().astype(int)
        ytick_labels = [f"{int(steps[i])}" for i in ytick_positions]

    ax2.set_yticks(ytick_positions)
    ax2.set_yticklabels(ytick_labels, fontsize=18, fontweight="bold")
    ax2.set_ylabel("global step", fontsize=25, fontweight="bold")
    apply_bin_axis_ticks(ax2, len(x), left_end, fontsize=18)

    if len(x) == 1:
        ax2.set_xlim(-0.5, 0.5)
    else:
        ax2.set_xlim(-0.5, len(x) - 0.5)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.024, pad=0.015)
    cbar.set_label("contrast: 4B positive, 1.7B negative", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=16)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax2.set_title(
        f"{train_label} contrast trajectory, final progress={progress[-1]:.3f}",
        fontsize=22,
        fontweight="bold",
        pad=12,
    )

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_linewidth(2.2)
        a.spines["bottom"].set_linewidth(2.2)
        for tick in a.get_yticklabels():
            tick.set_fontweight("bold")

    save_path = output_folder / save_name
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    model_path = "/mnt/phwfile/datafrontier/fudaocheng/checkpoints/huggingface/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    vocab_size = len(tokenizer)

    num_samples = 64

    qwen3_4b_db = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/datasets/"
        "G-OPD-Training-Data/DeepMath-103K/"
        f"val_union_mini_1000_Qwen3-4B_pass@{num_samples}.sqlite"
    )

    qwen3_1_7b_db = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/datasets/"
        "G-OPD-Training-Data/DeepMath-103K/"
        f"val_union_mini_1000_Qwen3-1.7B_pass@{num_samples}.sqlite"
    )

    train_model_name = "Qwen3-1.7B-T4B-Math_GB1024_OB0"
    date_time = "0602_10"

    train_model_db = Path(
        "/mnt/phwfile/datafrontier/fudaocheng/checkpoints/trained/LP_offline_distribution/"
        f"LP_{train_model_name}_{date_time}/"
        "validation_results.db"
    )

    output_folder = Path(
        f"plots_distribution/token_distribution_contrast/{train_model_name}_{date_time}/"
    )

    batch_size = 512
    sqlite_batch_size = 4096
    top_k = 3000
    bin_size = 30
    min_base_count = 20
    smoothing = 0.5
    gap_floor_dex = 0.03
    contrast_clip = 3.0
    heatmap_contrast_clip = 2.0

    excluded_token_ids = set()
    if tokenizer.all_special_ids is not None:
        excluded_token_ids.update(int(x) for x in tokenizer.all_special_ids if x is not None)

    freq_4b, counts_4b, total_4b = get_base_token_freq_sqlite(
        qwen3_4b_db,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        batch_size=batch_size,
        sqlite_batch_size=sqlite_batch_size,
    )

    freq_1_7b, counts_1_7b, total_1_7b = get_base_token_freq_sqlite(
        qwen3_1_7b_db,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        batch_size=batch_size,
        sqlite_batch_size=sqlite_batch_size,
    )

    ordered_bins, ordered_ids, left_ids, right_ids, left_end, log_gap_4b_minus_1_7b = build_ordered_token_bins_from_counts(
        counts_4b=counts_4b,
        total_4b=total_4b,
        counts_1_7b=counts_1_7b,
        total_1_7b=total_1_7b,
        top_k=top_k,
        bin_size=bin_size,
        min_base_count=min_base_count,
        smoothing=smoothing,
        excluded_token_ids=excluded_token_ids,
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    save_selected_tokens(
        tokenizer=tokenizer,
        save_path=output_folder / "selected_top_log_ratio_tokens.jsonl",
        ordered_ids=ordered_ids,
        counts_4b=counts_4b,
        total_4b=total_4b,
        counts_1_7b=counts_1_7b,
        total_1_7b=total_1_7b,
        log_gap_4b_minus_1_7b=log_gap_4b_minus_1_7b,
        smoothing=smoothing,
    )

    plot_base_token_distribution_enhanced(
        counts_4b=counts_4b,
        total_4b=total_4b,
        counts_1_7b=counts_1_7b,
        total_1_7b=total_1_7b,
        ordered_bins=ordered_bins,
        left_end=left_end,
        save_path=output_folder / f"base_token_distribution_logratio_top{top_k}.png",
        label_4b="Qwen3-4B",
        label_1_7b="Qwen3-1.7B",
        smoothing=smoothing,
    )

    train_freq_by_step, train_counts_by_step, train_total_by_step = get_train_token_freq_steps_sqlite(
        train_model_db,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        batch_size=batch_size,
        sqlite_batch_size=sqlite_batch_size,
    )

    plot_train_model_step_contrast_distribution(
        train_counts_by_step=train_counts_by_step,
        train_total_by_step=train_total_by_step,
        ordered_bins=ordered_bins,
        left_end=left_end,
        output_folder=output_folder,
        counts_4b=counts_4b,
        total_4b=total_4b,
        counts_1_7b=counts_1_7b,
        total_1_7b=total_1_7b,
        train_label=train_model_name,
        label_4b="Qwen3-4B",
        label_1_7b="Qwen3-1.7B",
        save_prefix=train_model_name,
        smoothing=smoothing,
        gap_floor_dex=gap_floor_dex,
        contrast_clip=contrast_clip,
    )

    heatmap_path = plot_train_contrast_heatmap(
        train_counts_by_step=train_counts_by_step,
        train_total_by_step=train_total_by_step,
        ordered_bins=ordered_bins,
        left_end=left_end,
        output_folder=output_folder,
        counts_4b=counts_4b,
        total_4b=total_4b,
        counts_1_7b=counts_1_7b,
        total_1_7b=total_1_7b,
        train_label=train_model_name,
        label_4b="Qwen3-4B",
        label_1_7b="Qwen3-1.7B",
        save_name=f"{train_model_name}_contrast_heatmap.png",
        smoothing=smoothing,
        gap_floor_dex=gap_floor_dex,
        contrast_clip=heatmap_contrast_clip,
    )

    base_y4 = binned_log_freq_from_counts(counts_4b, total_4b, ordered_bins, smoothing=smoothing)
    base_y17 = binned_log_freq_from_counts(counts_1_7b, total_1_7b, ordered_bins, smoothing=smoothing)

    train_progress_by_step = {}
    for step in sorted(train_counts_by_step):
        train_y = binned_log_freq_from_counts(
            train_counts_by_step[step],
            train_total_by_step[step],
            ordered_bins,
            smoothing=smoothing,
        )
        train_c = contrast_from_log_curves(
            train_y,
            base_y4,
            base_y17,
            gap_floor_dex=gap_floor_dex,
            clip=contrast_clip,
        )
        train_progress_by_step[str(step)] = float(np.nanmean((np.clip(train_c, -1.0, 1.0) + 1.0) * 0.5))

    summary = {
        "vocab_size": int(vocab_size),
        "base_4b_total_tokens": int(total_4b),
        "base_1_7b_total_tokens": int(total_1_7b),
        "top_k": int(top_k),
        "bin_size": int(bin_size),
        "min_base_count": int(min_base_count),
        "smoothing": float(smoothing),
        "gap_floor_dex": float(gap_floor_dex),
        "contrast_clip": float(contrast_clip),
        "num_left_tokens_1_7b_higher": int(len(left_ids)),
        "num_right_tokens_4b_higher": int(len(right_ids)),
        "num_bins": int(len(ordered_bins)),
        "left_end": int(left_end),
        "train_steps": [int(x) for x in sorted(train_counts_by_step)],
        "train_total_tokens_by_step": {
            str(k): int(v) for k, v in sorted(train_total_by_step.items())
        },
        "train_progress_to_4b_by_step": train_progress_by_step,
        "heatmap_path": str(heatmap_path) if heatmap_path is not None else None,
    }

    with open(output_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)