import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path


traj_id = 0
data_dir = Path(__file__).parent / "pku_ABCP_all"
axis_labels = ["action x", "action y", "action z", "action average"]

# ==================================
# 样式
# ==================================

# 主视觉：蓝/橙 action；其余元素降饱和、降透明度，为主线让路
COLORS = {
    "gt": "#1A5276",
    "pred": "#D95D39",
    "motion": "#6E6A7A",
    "gt_slope": "#C5DBEA",
    "pred_slope": "#F3D4B8",
}
BG_HIGH = "#FFECE8"
BG_LOW = "#EDF4FA"
BG_ALPHA = 0.30
BG = "#F8F9FA"

MOTION_ALPHA = 0.62
MOTION_LW = 1.35
SLOPE_BAR_ALPHA = 0.72
SLOPE_BAR_EDGE_LW = 1.0
MIN_SPAN_LEN = 5
MIN_SLOPE_BAR_RATIO = 0.06  # 区间步数占比低于此值不画斜率柱

GRID = "#DEE2E6"
SPINE = "#ADB5BD"

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 17,
        "axes.titlesize": 21,
        "axes.titleweight": "bold",
        "legend.fontsize": 16,
        "figure.facecolor": BG,
        "axes.facecolor": "white",
    }
)

state_cols = ["state_1"]
state_labels = {"state_1": "Tissue respiratory motion"}

# ==================================
# 读取数据
# ==================================

df = pd.read_csv(
    data_dir / f"trajectory_{traj_id}_plot.csv"
)

gt_cols = sorted(
    [c for c in df.columns if c.startswith("gt_action_")]
)

pred_cols = sorted(
    [c for c in df.columns if c.startswith("pred_action_")]
)

action_dim = len(gt_cols)
steps = len(df)
state_values = df[state_cols[0]].values

gt_all = df[gt_cols].values
pred_all = df[pred_cols].values
gt_avg = gt_all.mean(axis=1)
pred_avg = pred_all.mean(axis=1)


def abs_slope(values):
    return np.abs(np.gradient(values))


motion_median = np.median(state_values)


def motion_spans(motion, median):
    above = motion > median
    n = len(motion)
    spans = []
    if n == 0:
        return spans

    start = 0
    for i in range(1, n):
        if above[i] != above[start]:
            spans.append((start, i, above[start]))
            start = i
    spans.append((start, n, above[start]))
    return spans


def merge_short_spans(spans, min_len=MIN_SPAN_LEN):
    spans = list(spans)
    if len(spans) <= 1:
        return spans

    while True:
        lengths = [end - start for start, end, _ in spans]
        min_length = min(lengths)
        if min_length >= min_len:
            break

        idx = lengths.index(min_length)
        start, end, _ = spans[idx]

        if idx == 0:
            next_start, next_end, next_high = spans[1]
            spans[1] = (start, next_end, next_high)
            spans.pop(0)
        elif idx == len(spans) - 1:
            prev_start, _, prev_high = spans[idx - 1]
            spans[idx - 1] = (prev_start, end, prev_high)
            spans.pop(idx)
        else:
            prev_len = spans[idx - 1][1] - spans[idx - 1][0]
            next_len = spans[idx + 1][1] - spans[idx + 1][0]
            if prev_len >= next_len:
                prev_start, _, prev_high = spans[idx - 1]
                spans[idx - 1] = (prev_start, end, prev_high)
                spans.pop(idx)
            else:
                next_start, next_end, next_high = spans[idx + 1]
                spans[idx + 1] = (start, next_end, next_high)
                spans.pop(idx)

    return spans


def merge_adjacent_spans(spans):
    if not spans:
        return spans

    merged = [spans[0]]
    for start, end, is_high in spans[1:]:
        prev_start, prev_end, prev_high = merged[-1]
        if is_high == prev_high:
            merged[-1] = (prev_start, end, prev_high)
        else:
            merged.append((start, end, is_high))
    return merged


def spans_for_slope_bars(spans, total_steps):
    if not spans:
        return spans

    filtered = []
    for start, end, is_high in spans:
        span_len = end - start
        if span_len < MIN_SPAN_LEN:
            continue
        if span_len / total_steps < MIN_SLOPE_BAR_RATIO:
            continue
        filtered.append((start, end, is_high))
    return filtered


MOTION_SPANS = merge_adjacent_spans(
    merge_short_spans(
        motion_spans(state_values, motion_median)
    )
)


def draw_motion_background(ax, spans):
    for start, end, is_high in spans:
        color = BG_HIGH if is_high else BG_LOW
        ax.axvspan(
            start - 0.5,
            end - 0.5,
            facecolor=color,
            alpha=BG_ALPHA,
            zorder=0,
            linewidth=0,
        )


def draw_slope_bars(ax, gt_slope, pred_slope, spans, with_labels):
    gt_label_done = False
    pred_label_done = False

    spans_to_draw = spans_for_slope_bars(spans, steps)

    gt_means = [np.mean(gt_slope[s:e]) for s, e, _ in spans_to_draw]
    pred_means = [np.mean(pred_slope[s:e]) for s, e, _ in spans_to_draw]
    max_val = max(max(gt_means, default=0), max(pred_means, default=0), 1e-12)

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    bar_zone = y_range * 0.28
    bar_base = ymin

    for idx, (start, end, _) in enumerate(spans_to_draw):
        span_width = end - start
        gt_mean = gt_means[idx]
        pred_mean = pred_means[idx]

        gap = span_width * 0.10
        bar_width = span_width * 0.18
        group_width = bar_width * 2 + gap
        x_base = start - 0.5 + (span_width - group_width) / 2

        gt_height = (gt_mean / max_val) * bar_zone
        pred_height = (pred_mean / max_val) * bar_zone

        gt_label = (
            "gt action slope"
            if with_labels and not gt_label_done
            else "_nolegend_"
        )
        pred_label = (
            "pred action slope"
            if with_labels and not pred_label_done
            else "_nolegend_"
        )

        ax.bar(
            x_base,
            gt_height,
            width=bar_width,
            bottom=bar_base,
            align="edge",
            color=COLORS["gt_slope"],
            alpha=SLOPE_BAR_ALPHA,
            edgecolor=COLORS["gt"],
            linewidth=SLOPE_BAR_EDGE_LW,
            zorder=2,
            label=gt_label,
        )
        ax.bar(
            x_base + bar_width + gap,
            pred_height,
            width=bar_width,
            bottom=bar_base,
            align="edge",
            color=COLORS["pred_slope"],
            alpha=SLOPE_BAR_ALPHA,
            edgecolor=COLORS["pred"],
            linewidth=SLOPE_BAR_EDGE_LW,
            zorder=2,
            label=pred_label,
        )

        if gt_label != "_nolegend_":
            gt_label_done = True
        if pred_label != "_nolegend_":
            pred_label_done = True


def style_axis(ax, ax2):
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.55)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        length=3,
        color=SPINE,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(SPINE)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["bottom"].set_visible(False)
    ax2.tick_params(
        axis="y",
        which="both",
        labelright=False,
        length=3,
        color=SPINE,
    )

    ax.set_xlim(-0.5, steps - 0.5)
    ax.margins(x=0)


def expand_ylim_for_slope_bars(ax, bottom_ratio=0.32):
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    extra = y_range * bottom_ratio / (1 - bottom_ratio)
    ax.set_ylim(ymin - extra, ymax)


def plot_subplot(ax, gt, pred, title, with_labels=False):
    def lbl(name):
        return name if with_labels else "_nolegend_"

    draw_motion_background(ax, MOTION_SPANS)

    ax.plot(
        gt,
        color=COLORS["gt"],
        linewidth=2.4,
        label=lbl("gt action"),
        zorder=5,
    )
    ax.plot(
        pred,
        color=COLORS["pred"],
        linewidth=2.2,
        alpha=0.95,
        label=lbl("pred action"),
        zorder=4,
    )

    gt_slope = abs_slope(gt)
    pred_slope = abs_slope(pred)

    ax2 = ax.twinx()

    for state_col in state_cols:
        ax2.plot(
            state_values,
            color=COLORS["motion"],
            linewidth=MOTION_LW,
            linestyle=(0, (5, 4)),
            alpha=MOTION_ALPHA,
            label=lbl(state_labels.get(state_col, state_col)),
            zorder=2,
        )

    expand_ylim_for_slope_bars(ax)
    draw_slope_bars(ax, gt_slope, pred_slope, MOTION_SPANS, with_labels)

    ax.set_title(title, loc="left", pad=6, color="#343A40")
    style_axis(ax, ax2)
    return ax2


def gt_pred_for_spec(axis_key):
    if axis_key == "avg":
        return gt_avg, pred_avg
    return df[f"gt_action_{axis_key}"].values, df[f"pred_action_{axis_key}"].values


def render_figure(
    plot_specs,
    figsize,
    output_path,
    top_margin=0.88,
    hspace=0.18,
    legend_below=False,
    legend_pos="top",
):
    n_plots = len(plot_specs)

    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=figsize,
        facecolor=BG,
    )
    if n_plots == 1:
        axes = [axes]

    ax2_for_legend = None
    for idx, (axis_key, title) in enumerate(plot_specs):
        gt, pred = gt_pred_for_spec(axis_key)
        ax2 = plot_subplot(
            axes[idx],
            gt,
            pred,
            title,
            with_labels=(idx == 0),
        )
        if idx == 0:
            ax2_for_legend = ax2

    legend_handles = []
    legend_labels = []
    for legend_ax in [axes[0], ax2_for_legend]:
        handles, labels = legend_ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label != "_nolegend_":
                legend_handles.append(handle)
                legend_labels.append(label)

    if legend_pos == "top_outside":
        legend_loc = "lower center"
        legend_ncol = 3
        legend_anchor = (0.5, 0.90) if n_plots > 1 else (0.5, 0.97)
    elif legend_below:
        legend_anchor = (0.5, -0.06)
        legend_loc = "upper center"
        legend_ncol = 3
    else:
        legend_anchor = (0.5, 0.985)
        legend_loc = "upper center"
        legend_ncol = 3

    fig.legend(
        legend_handles,
        legend_labels,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        ncol=legend_ncol,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor=GRID,
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.6,
    ).get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.3)
    if legend_pos == "top_outside":
        if n_plots > 1:
            plt.subplots_adjust(top=0.88, bottom=0.06, hspace=hspace)
        else:
            plt.subplots_adjust(top=0.80, bottom=0.10)
    elif legend_below:
        plt.subplots_adjust(top=0.94, bottom=0.24)
    elif n_plots > 1:
        plt.subplots_adjust(hspace=hspace, top=top_margin)
    else:
        plt.subplots_adjust(top=top_margin)

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor=BG,
    )
    plt.close(fig)


# ==================================
# 画图
# ==================================

all_plot_specs = [
    (0, axis_labels[0]),
    (1, axis_labels[1]),
    (2, axis_labels[2]),
    ("avg", axis_labels[3]),
]

action_x_plot_specs = [
    (0, "action"),
]

render_figure(
    all_plot_specs,
    figsize=(10, 2.6 * len(all_plot_specs)),
    output_path=data_dir / f"trajectory_{traj_id}_replot.png",
    legend_pos="top_outside",
    hspace=0.18,
)

render_figure(
    action_x_plot_specs,
    figsize=(14, 4.4),
    output_path=data_dir / f"trajectory_{traj_id}_action_replot.png",
    legend_pos="top_outside",
)
