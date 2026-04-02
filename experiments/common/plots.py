import numpy as np


def _palette(labels):
    unique = sorted(set(labels))
    try:
        import seaborn as sns

        colors = sns.color_palette(None, n_colors=len(unique))
    except ModuleNotFoundError:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20", max(len(unique), 1))
        colors = [cmap(i) for i in range(len(unique))]
    return {label: colors[i] for i, label in enumerate(unique)}


def save_scatter(
    path,
    x,
    y,
    groups,
    title,
    xlabel,
    ylabel,
    ref_line=False,
    anchors_a=None,
    anchors_b=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anchors_a = [] if anchors_a is None else anchors_a
    anchors_b = [] if anchors_b is None else anchors_b

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = list(groups)
    color_map = _palette(groups)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for group in sorted(set(groups)):
        idx = [i for i, value in enumerate(groups) if value == group]
        ax.scatter(x[idx], y[idx], s=26, alpha=0.8, color=color_map[group], label=group)
    if anchors_a is not None:
        ax.scatter(
            anchors_a,
            anchors_a,
            alpha=0.7,
            color="black",
            marker="X",
            label="Anchors A",
        )
    if anchors_b is not None:
        ax.scatter(
            anchors_b,
            anchors_b,
            alpha=0.7,
            color="black",
            marker="D",
            label="Anchors B",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    if len(set(groups)) <= 12:
        ax.legend(loc="best", fontsize=7)
    if ref_line:
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="red",
            alpha=0.7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_projection_strip(path, projections, item_labels, groups, title, xlabel):
    """Plot 1-D projection values: x = item (labeled), y = projection scalar.

    Parameters
    ----------
    path : Path
        Output file path.
    projections : array-like, shape (n_items,)
        One projection value per item.
    item_labels : list[str]
        One label per item (used for x-axis ticks).
    groups : list[str]
        One group name per item (used for colour).
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    projections = np.asarray(projections, dtype=np.float64)
    n_items = len(item_labels)
    color_map = _palette(groups)
    xs = np.arange(n_items)

    fig, ax = plt.subplots(figsize=(max(8.0, n_items * 0.35), 4.8))
    colors = [color_map[g] for g in groups]
    ax.scatter(xs, projections, s=26, alpha=0.8, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(item_labels, rotation=90, fontsize=6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Projection value")
    ax.grid(True, linestyle="--", alpha=0.35)
    if len(set(groups)) <= 12:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[lbl],
                markersize=6,
                label=lbl,
            )
            for lbl in sorted(set(groups))
        ]
        ax.legend(handles=handles, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_table_png(path, headers, rows, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(14.0, len(headers) * 1.0), max(4.0, len(rows) * 0.5 + 2.5))
    )
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
