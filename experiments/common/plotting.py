from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_predicted_vs_real_plot(path, real, predicted, groups, is_anchor, title):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    real_values = np.asarray(real, dtype=np.float64)
    pred_values = np.asarray(predicted, dtype=np.float64)
    group_values = np.asarray(groups, dtype=object)
    anchor_values = np.asarray(is_anchor, dtype=bool)
    anchor_labels = np.where(anchor_values, "anchor", "non-anchor")
    sns.scatterplot(
        x=real_values,
        y=pred_values,
        hue=group_values,
        style=anchor_labels,
        ax=ax,
        s=55,
        alpha=0.85,
    )

    low = float(min(np.min(real_values), np.min(pred_values)))
    high = float(max(np.max(real_values), np.max(pred_values)))
    ax.plot(
        [low, high], [low, high], linestyle="--", linewidth=2, color="red", label="y=x"
    )
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_n_vs_spearman(path, sweep_by_method, title, linestyle="-"):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))

    methods = sorted(sweep_by_method.keys())
    palette = sns.color_palette("deep", n_colors=max(1, len(methods)))

    if methods:
        step = 0.12
        center = 0.5 * (len(methods) - 1)
    for idx, (color, anchor_method) in enumerate(zip(palette, methods)):
        sweep_values = sweep_by_method[anchor_method]
        n_array = np.asarray(sweep_values["n_values"], dtype=np.float64)
        s_array = np.asarray(sweep_values["scores"], dtype=np.float64)
        c_array = np.asarray(sweep_values["ci95"], dtype=np.float64)
        n_plot = n_array + (idx - center) * step if methods else n_array
        ax.plot(
            n_plot,
            s_array,
            marker="o",
            linewidth=1,
            alpha=0.9,
            linestyle=linestyle,
            color=color,
            label=anchor_method,
        )
        lower = s_array - c_array
        upper = s_array + c_array
        ax.fill_between(n_plot, lower, upper, alpha=0.08, color=color)

    if methods:
        ax.legend(loc="best")

    ax.set_xlabel("n")
    ax.set_ylabel("Spearman")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
