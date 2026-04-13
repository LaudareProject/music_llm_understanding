from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


MAX_GROUPS_IN_LEGEND = 10


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
        style_order=["non-anchor", "anchor"],
        markers={"non-anchor": "o", "anchor": "X"},
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
    if np.unique(group_values).size <= MAX_GROUPS_IN_LEGEND:
        ax.legend(loc="best")
    else:
        handles, labels = ax.get_legend_handles_labels()
        marker_labels = {"anchor", "non-anchor"}
        filtered = [
            (handle, label)
            for handle, label in zip(handles, labels)
            if label in marker_labels
        ]
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if filtered:
            marker_handles, marker_names = zip(*filtered)
            ax.legend(marker_handles, marker_names, loc="best")
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


def save_n_vs_spearman_combined(path, n_sweep, title):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    direction_label = "direction-based"
    regression_label = "regression hyper-plane"
    mode_styles = {
        False: {"linestyle": "-", "marker": "o", "mode_label": direction_label},
        True: {"linestyle": "--", "marker": "s", "mode_label": regression_label},
    }

    all_methods = sorted(
        set(n_sweep.get(False, {}).keys()) | set(n_sweep.get(True, {}).keys())
    )
    palette = sns.color_palette("deep", n_colors=max(1, len(all_methods)))
    color_by_method = {method: color for method, color in zip(all_methods, palette)}

    all_n_values = set()
    for mode in (False, True):
        for anchor_method in sorted(n_sweep.get(mode, {}).keys()):
            sweep_values = n_sweep[mode][anchor_method]
            n_array = np.asarray(sweep_values["n_values"], dtype=np.int64)
            if n_array.size == 0:
                continue
            all_n_values.update(int(n) for n in n_array.tolist())

            s_array = np.asarray(sweep_values["scores"], dtype=np.float64)
            c_array = np.asarray(sweep_values["ci95"], dtype=np.float64)
            style = mode_styles[mode]
            color = color_by_method[anchor_method]
            ax.plot(
                n_array,
                s_array,
                marker=style["marker"],
                markersize=5.5,
                linewidth=1.4,
                alpha=0.95,
                linestyle=style["linestyle"],
                color=color,
                label="_nolegend_",
            )
            lower = s_array - c_array
            upper = s_array + c_array
            ax.fill_between(n_array, lower, upper, alpha=0.08, color=color)

    if all_n_values:
        n_ticks = sorted(all_n_values)
        ax.set_xticks(n_ticks)
        ax.set_xlim(float(min(n_ticks)), float(max(n_ticks)))

    ax.grid(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.6)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=color_by_method[anchor_method],
            linestyle="-",
            marker="o",
            markersize=5.5,
            linewidth=1.6,
            label=anchor_method,
        )
        for anchor_method in all_methods
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=mode_styles[False]["linestyle"],
            marker=mode_styles[False]["marker"],
            markersize=5.5,
            linewidth=1.6,
            label=mode_styles[False]["mode_label"],
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=mode_styles[True]["linestyle"],
            marker=mode_styles[True]["marker"],
            markersize=5.5,
            linewidth=1.6,
            label=mode_styles[True]["mode_label"],
        ),
    ]
    legend_handles = method_handles + mode_handles
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            ncol=3,
            fontsize=9,
            framealpha=0.92,
            columnspacing=1.2,
            handletextpad=0.5,
            borderpad=0.5,
        )

    ax.set_xlabel("n")
    ax.set_ylabel("Spearman")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
