import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

TRAJECTORY_IDS = (0, 1, 17, 18, 19,)
DATA_DIR = Path(__file__).parent / "trajectory"
ACTION_X_AXIS = 0

# ==================================
# 样式
# ==================================

COLORS = {
    "gt": "#1A5276",
    "pred": "#D95D39",
    "pred_compare": "#2E7D32",
    "motion": "#6E6A7A",
    "gt_slope": "#C5DBEA",
    "pred_slope": "#F3D4B8",
    "pred_compare_slope": "#B7DFC5",
}
BG_HIGH = "#FFECE8"
BG_LOW = "#EDF4FA"
BG_ALPHA = 0.30
BG = "#F8F9FA"

MOTION_ALPHA = 0.62
MOTION_LW = 1.35
SLOPE_BAR_ALPHA = 0.72
SLOPE_BAR_EDGE_LW = 1.0
SLOPE_BAR_WIDTH = 6.0
SLOPE_BAR_GAP = 1.4
MIN_SPAN_LEN = 5
MIN_SLOPE_BAR_RATIO = 0.06

GRID = "#DEE2E6"
SPINE = "#ADB5BD"

STATE_LABELS = {
    "state_0": "Tissue state 0",
    "state_1": "Tissue respiratory motion",
}

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


DEFAULT_COMPARE: dict[int, int] = {}


@dataclass
class TrajectoryData:
    traj_id: int
    df: pd.DataFrame
    steps: int
    gt: np.ndarray
    pred: np.ndarray
    state_cols: list[str]
    motion_values: np.ndarray
    motion_spans: list[tuple[int, int, bool]]
    compare_traj_id: int | None = None
    pred_compare: np.ndarray | None = None


def csv_path_for_traj(traj_id: int) -> Path:
    return DATA_DIR / f"trajectory_{traj_id}_plot.csv"


def output_path_for_traj(traj_id: int) -> Path:
    return DATA_DIR / f"trajectory_{traj_id}_action_x.png"


def abs_slope(values):
    return np.abs(np.gradient(values))


def local_extrema(motion, min_distance=MIN_SPAN_LEN):
    n = len(motion)
    if n < 3:
        return []

    candidates = []
    for i in range(1, n - 1):
        if motion[i] >= motion[i - 1] and motion[i] > motion[i + 1]:
            candidates.append((i, "peak"))
        elif motion[i] <= motion[i - 1] and motion[i] < motion[i + 1]:
            candidates.append((i, "valley"))

    if not candidates:
        return []

    filtered = [candidates[0]]
    for idx, kind in candidates[1:]:
        prev_idx, prev_kind = filtered[-1]
        if idx - prev_idx < min_distance:
            if kind == "peak" and motion[idx] >= motion[prev_idx]:
                filtered[-1] = (idx, kind)
            elif kind == "valley" and motion[idx] <= motion[prev_idx]:
                filtered[-1] = (idx, kind)
        elif kind != prev_kind:
            filtered.append((idx, kind))
        elif kind == "peak" and motion[idx] > motion[prev_idx]:
            filtered[-1] = (idx, kind)
        elif kind == "valley" and motion[idx] < motion[prev_idx]:
            filtered[-1] = (idx, kind)

    return filtered


def motion_spans_from_extrema(motion, min_distance=MIN_SPAN_LEN):
    """按波峰/波谷切分区间：上升段为高运动，下降段为低运动。"""
    n = len(motion)
    if n == 0:
        return []

    extrema = local_extrema(motion, min_distance=min_distance)
    boundaries = [0] + [idx for idx, _ in extrema] + [n]
    boundaries = sorted(set(boundaries))

    spans = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start:
            continue
        is_high = motion[end - 1] > motion[start]
        spans.append((start, end, is_high))
    return spans


def merge_island_spans(spans, max_island_len=15):
    """合并被同类区间夹住的短异类区间（如上升过程中的小回落）。"""
    spans = list(spans)
    if len(spans) < 3:
        return spans

    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(spans):
            if (
                i + 2 < len(spans)
                and spans[i][2] == spans[i + 2][2]
                and spans[i + 1][2] != spans[i][2]
                and spans[i + 1][1] - spans[i + 1][0] <= max_island_len
            ):
                start, _, is_high = spans[i]
                end = spans[i + 2][1]
                merged.append((start, end, is_high))
                i += 3
                changed = True
            else:
                merged.append(spans[i])
                i += 1
        spans = merged
    return spans


def finalize_motion_spans(spans):
    spans = merge_short_spans(spans)
    spans = merge_adjacent_spans(spans)
    spans = merge_island_spans(spans)
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
            _, next_end, next_high = spans[1]
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
        prev_start, _, prev_high = merged[-1]
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


def primary_motion_col(state_cols: list[str]) -> str:
    if "state_1" in state_cols:
        return "state_1"
    return state_cols[0]


def trim_trailing_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """去掉末尾无效行（如 NaN 填充），避免右侧出现空白绘图区域。"""
    required_cols = [
        f"gt_action_{ACTION_X_AXIS}",
        f"pred_action_{ACTION_X_AXIS}",
        *sorted(c for c in df.columns if c.startswith("state_")),
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {', '.join(missing)}")

    valid = np.ones(len(df), dtype=bool)
    for col in required_cols:
        valid &= np.isfinite(df[col].values)

    if not valid.any():
        raise ValueError("CSV 中没有可用的有效数据行")

    last_valid = int(np.where(valid)[0][-1])
    return df.iloc[: last_valid + 1].reset_index(drop=True)


def load_trajectory(traj_id: int, compare_traj_id: int | None = None) -> TrajectoryData:
    csv_path = csv_path_for_traj(traj_id)
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    df = trim_trailing_invalid_rows(pd.read_csv(csv_path))
    state_cols = sorted(c for c in df.columns if c.startswith("state_"))
    if not state_cols:
        raise ValueError(f"{csv_path} 中缺少 state_* 列")

    motion_col = primary_motion_col(state_cols)
    motion_values = df[motion_col].values
    motion_spans = finalize_motion_spans(
        motion_spans_from_extrema(motion_values)
    )

    pred_compare = None
    if compare_traj_id is not None:
        compare_path = csv_path_for_traj(compare_traj_id)
        if not compare_path.is_file():
            raise FileNotFoundError(f"未找到对比数据文件: {compare_path}")
        compare_df = trim_trailing_invalid_rows(pd.read_csv(compare_path))
        pred_compare = compare_df[f"pred_action_{ACTION_X_AXIS}"].values

    return TrajectoryData(
        traj_id=traj_id,
        df=df,
        steps=len(df),
        gt=df[f"gt_action_{ACTION_X_AXIS}"].values,
        pred=df[f"pred_action_{ACTION_X_AXIS}"].values,
        state_cols=state_cols,
        motion_values=motion_values,
        motion_spans=motion_spans,
        compare_traj_id=compare_traj_id,
        pred_compare=pred_compare,
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


def mean_slope_in_span(slope, start, end):
    if start >= len(slope):
        return None
    return float(np.mean(slope[start:min(end, len(slope))]))


def draw_slope_bars(ax, gt_slope, pred_series, spans, total_steps, with_labels):
    spans_to_draw = spans_for_slope_bars(spans, total_steps)

    gt_means = [mean_slope_in_span(gt_slope, s, e) for s, e, _ in spans_to_draw]
    pred_means_list = [
        [mean_slope_in_span(slope, s, e) for s, e, _ in spans_to_draw]
        for slope, _, _, _ in pred_series
    ]

    all_means = [v for v in gt_means if v is not None]
    for pred_means in pred_means_list:
        all_means.extend(v for v in pred_means if v is not None)
    max_val = max(all_means, default=0)
    max_val = max(max_val, 1e-12)

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    bar_zone = y_range * 0.28
    bar_base = ymin

    label_done = {label: False for _, _, _, label in pred_series}
    gt_label_done = False

    n_pred = len(pred_series)
    bar_width = SLOPE_BAR_WIDTH
    gap = SLOPE_BAR_GAP
    group_width = bar_width * (1 + n_pred) + gap * n_pred

    for idx, (start, end, _) in enumerate(spans_to_draw):
        span_width = end - start
        span_center = start - 0.5 + span_width / 2
        x_base = span_center - group_width / 2

        gt_mean = gt_means[idx]
        if gt_mean is not None:
            gt_label = (
                "gt action slope"
                if with_labels and not gt_label_done
                else "_nolegend_"
            )
            ax.bar(
                x_base,
                (gt_mean / max_val) * bar_zone,
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
            if gt_label != "_nolegend_":
                gt_label_done = True

        cursor = x_base + bar_width + gap
        for pred_idx, (slope, fill_color, edge_color, label) in enumerate(pred_series):
            pred_mean = pred_means_list[pred_idx][idx]
            if pred_mean is None:
                cursor += bar_width + gap
                continue

            pred_label = (
                label
                if with_labels and not label_done[label]
                else "_nolegend_"
            )
            ax.bar(
                cursor,
                (pred_mean / max_val) * bar_zone,
                width=bar_width,
                bottom=bar_base,
                align="edge",
                color=fill_color,
                alpha=SLOPE_BAR_ALPHA,
                edgecolor=edge_color,
                linewidth=SLOPE_BAR_EDGE_LW,
                zorder=2,
                label=pred_label,
            )
            if pred_label != "_nolegend_":
                label_done[label] = True
            cursor += bar_width + gap


def style_axis(ax, ax2, steps):
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


def pred_label_for(traj_id: int, role: str, with_traj_suffix: bool) -> str:
    if not with_traj_suffix:
        return "pred action" if role == "action" else "pred action slope"
    suffix = f" ({traj_id})"
    return f"pred action{suffix}" if role == "action" else f"pred action slope{suffix}"


def plot_action_x(
    ax,
    data: TrajectoryData,
    title: str,
    motion_spans,
    with_labels=True,
):
    def lbl(name):
        return name if with_labels else "_nolegend_"

    draw_motion_background(ax, motion_spans)

    ax.plot(
        data.gt,
        color=COLORS["gt"],
        linewidth=2.4,
        label=lbl("gt action"),
        zorder=5,
    )
    has_compare = data.pred_compare is not None and data.compare_traj_id is not None

    ax.plot(
        data.pred,
        color=COLORS["pred"],
        linewidth=2.2,
        alpha=0.95,
        label=lbl(pred_label_for(data.traj_id, "action", has_compare)),
        zorder=4,
    )
    if has_compare:
        compare_len = len(data.pred_compare)
        ax.plot(
            np.arange(compare_len),
            data.pred_compare,
            color=COLORS["pred_compare"],
            linewidth=2.2,
            alpha=0.95,
            label=lbl(pred_label_for(data.compare_traj_id, "action", True)),
            zorder=4,
        )

    gt_slope = abs_slope(data.gt)
    pred_series = [
        (
            abs_slope(data.pred),
            COLORS["pred_slope"],
            COLORS["pred"],
            pred_label_for(data.traj_id, "slope", has_compare),
        )
    ]
    if has_compare:
        pred_series.append(
            (
                abs_slope(data.pred_compare),
                COLORS["pred_compare_slope"],
                COLORS["pred_compare"],
                pred_label_for(data.compare_traj_id, "slope", True),
            )
        )

    ax2 = ax.twinx()

    for state_col in data.state_cols:
        ax2.plot(
            data.df[state_col].values,
            color=COLORS["motion"],
            linewidth=MOTION_LW,
            linestyle=(0, (5, 4)),
            alpha=MOTION_ALPHA,
            label=lbl(STATE_LABELS.get(state_col, state_col)),
            zorder=2,
        )

    expand_ylim_for_slope_bars(ax)
    draw_slope_bars(
        ax,
        gt_slope,
        pred_series,
        motion_spans,
        data.steps,
        with_labels,
    )

    ax.set_title(title, loc="left", pad=6, color="#343A40")
    style_axis(ax, ax2, data.steps)
    return ax2


def render_figure(data: TrajectoryData, output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 4.4), facecolor=BG)
    ax2 = plot_action_x(ax, data, "action", data.motion_spans, with_labels=True)

    legend_handles = []
    legend_labels = []
    for legend_ax in [ax, ax2]:
        handles, labels = legend_ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label != "_nolegend_":
                legend_handles.append(handle)
                legend_labels.append(label)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor=GRID,
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.6,
    ).get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(top=0.80, bottom=0.10)

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor=BG,
    )
    plt.close(fig)


def parse_args():
    id_help = ", ".join(str(t) for t in TRAJECTORY_IDS)
    parser = argparse.ArgumentParser(
        description="根据轨迹编号绘制 trajectory action x 图"
    )
    parser.add_argument(
        "traj_id",
        type=int,
        choices=TRAJECTORY_IDS,
        help=f"轨迹编号，可选: {id_help}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 PNG 路径（默认写入 trajectory 目录）",
    )
    parser.add_argument(
        "--compare",
        type=int,
        default=None,
        choices=TRAJECTORY_IDS,
        help="叠加对比轨迹的 pred action",
    )
    return parser.parse_args()


def resolve_compare_traj_id(traj_id: int, compare_arg: int | None) -> int | None:
    if compare_arg is None:
        return DEFAULT_COMPARE.get(traj_id)
    if compare_arg == traj_id:
        raise ValueError("对比轨迹不能与主轨迹相同")
    return compare_arg


def main():
    args = parse_args()
    compare_traj_id = resolve_compare_traj_id(args.traj_id, args.compare)
    data = load_trajectory(args.traj_id, compare_traj_id=compare_traj_id)
    output_path = args.output or output_path_for_traj(args.traj_id)
    render_figure(data, output_path)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
