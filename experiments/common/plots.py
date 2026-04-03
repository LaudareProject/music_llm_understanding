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


def save_heatmap(path, row_labels, col_labels, values, title, cmap="viridis"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float64)
    fig, ax = plt.subplots(
        figsize=(max(6.0, len(col_labels) * 1.1), max(3.5, len(row_labels) * 0.7 + 1.5))
    )
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_image_matrix(
    path, image_paths, row_labels, col_labels, title, cell_texts=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(4.5, n_cols * 4.0), max(3.5, n_rows * 3.2)),
        squeeze=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            ax = axes[i][j]
            image_path = image_paths[i][j]
            if image_path is not None and image_path.exists():
                ax.imshow(mpimg.imread(image_path))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
            if cell_texts is not None and cell_texts[i][j]:
                ax.text(
                    0.02,
                    0.02,
                    cell_texts[i][j],
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_label, fontsize=10)
            if j == 0:
                ax.set_ylabel(row_label, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_grouped_bar_chart(path, categories, series, title, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    categories = list(categories)
    xs = np.arange(len(categories), dtype=np.float64)
    width = 0.38
    offsets = np.linspace(-width / 2.0, width / 2.0, num=max(len(series), 1))
    fig, ax = plt.subplots(figsize=(max(8.0, len(categories) * 0.7), 4.8))
    for offset, (label, values, color) in zip(offsets, series):
        ax.bar(xs + offset, values, width=width, label=label, color=color, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_stacked_bar_chart(path, categories, series, title, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    categories = list(categories)
    xs = np.arange(len(categories), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(8.0, len(categories) * 0.7), 4.8))
    bottoms = np.zeros(len(categories), dtype=np.float64)
    for item in series:
        if len(item) == 3:
            label, values, color = item
        else:
            label, values = item
            color = None
        values = np.asarray(values, dtype=np.float64)
        ax.bar(xs, values, bottom=bottoms, label=label, color=color, alpha=0.9)
        bottoms = bottoms + values
    ax.set_xticks(xs)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_multi_line_chart(path, x_values, series, title, xlabel, ylabel):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.asarray(x_values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (label, y_values) in enumerate(series):
        y_values = np.asarray(y_values, dtype=np.float64)
        ax.plot(
            x_values,
            y_values,
            marker=markers[i % len(markers)],
            linewidth=1.8,
            markersize=5,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_multi_line_grid_chart(path, panels, ncols, title, xlabel, ylabel):
    import math
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    panel_count = max(len(panels), 1)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(panel_count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(6.0, ncols * 5.2), max(4.0, nrows * 3.8)),
        squeeze=False,
    )
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= len(panels):
            ax.axis("off")
            continue
        panel_title, x_values, series = panels[idx]
        x_values = np.asarray(x_values, dtype=np.float64)
        for i, (label, y_values) in enumerate(series):
            y_values = np.asarray(y_values, dtype=np.float64)
            ax.plot(
                x_values,
                y_values,
                marker=markers[i % len(markers)],
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_values)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
