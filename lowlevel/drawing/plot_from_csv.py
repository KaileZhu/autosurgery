import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
TRAJECTORY_IDS = (17, 18, 19)
MODELS = ("ABCP",)
DATA_DIR = Path(__file__).parent / "trajectory"
ACTION_X_AXIS = 0
FUSION_K = 0.03

# ==================================
# 样式
# ==================================

COLORS = {
    "gt": "#1A5276",
    "pred_csv": "#D95D39",
    "pred_fused": "#2E7D32",
    "motion": "#6E6A7A",
    "gt_slope": "#C5DBEA",
    "pred_csv_slope": "#F3D4B8",
    "pred_fused_slope": "#B7DFC5",
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

MOTION_LABELS = {
    "motion": "Tissue respiratory motion",
    "motion_0": "Tissue motion 0",
    "motion_1": "Tissue respiratory motion",
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


@dataclass
class TrajectoryData:
    label: str
    df: pd.DataFrame
    steps: int
    gt: np.ndarray
    pred_csv: np.ndarray
    signal_cols: list[str]
    motion_values: np.ndarray
    motion_spans: list[tuple[int, int, bool]]
    pred_fused: np.ndarray | None = None
    fused_only: bool = False


def csv_path_for_run(traj_id: int, model: str) -> Path:
    return DATA_DIR / str(traj_id) / f"{model}.csv"


def output_path_for_run(traj_id: int, model: str) -> Path:
    return DATA_DIR / str(traj_id) / f"{model}.png"


def npy_path_for_run(traj_id: int, model: str) -> Path:
    return DATA_DIR / str(traj_id) / f"{model}.npy"


def pred_from_chunk_fusion(chunk_data: np.ndarray, k: float = FUSION_K) -> np.ndarray:
    """对 chunk 预测做指数加权时间融合，返回 (steps, action_dim)。"""
    steps, chunk_horizon, action_dim = chunk_data.shape
    all_time_actions = np.zeros(
        (steps, steps + chunk_horizon, action_dim),
        dtype=chunk_data.dtype,
    )
    for step_count in range(steps):
        all_time_actions[
            step_count, step_count : step_count + chunk_horizon
        ] = chunk_data[step_count]

    pred_action_across_time = []
    for step_count in range(steps):
        actions_for_curr_step = all_time_actions[:, step_count]
        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        if len(actions_for_curr_step) == 0:
            raise ValueError(f"时间步 {step_count} 没有可用的 chunk 动作")

        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        raw_action = np.sum(
            actions_for_curr_step * exp_weights.reshape(-1, 1),
            axis=0,
        )
        pred_action_across_time.append(raw_action)

    return np.array(pred_action_across_time)


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


def signal_plot_cols(df: pd.DataFrame) -> list[str]:
    if "motion" in df.columns:
        return ["motion"]
    motion_cols = sorted(c for c in df.columns if c.startswith("motion_"))
    if motion_cols:
        return motion_cols
    state_cols = sorted(c for c in df.columns if c.startswith("state_"))
    if state_cols:
        return state_cols
    return []


def primary_span_col(signal_cols: list[str]) -> str:
    for name in ("motion_1", "motion", "state_1", "state_0"):
        if name in signal_cols:
            return name
    return signal_cols[0]


def trim_trailing_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """去掉末尾无效行（如 NaN 填充），避免右侧出现空白绘图区域。"""
    signal_cols = signal_plot_cols(df)
    if not signal_cols:
        raise ValueError("CSV 缺少 motion 或 state_* 列")

    required_cols = [
        f"gt_action_{ACTION_X_AXIS}",
        f"pred_action_{ACTION_X_AXIS}",
        *signal_cols,
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


def build_trajectory_data(
    df: pd.DataFrame,
    csv_path: Path,
    label: str,
    use_fusion: bool,
    npy_path: Path | None,
    fused_only: bool = False,
) -> TrajectoryData:
    signal_cols = signal_plot_cols(df)
    if not signal_cols:
        raise ValueError(f"{csv_path} 中缺少 motion 或 state_* 列")

    pred_fused = None
    if use_fusion or fused_only:
        if npy_path is None:
            raise ValueError("启用 --fuse / --fused-only 时必须提供对应的 .npy 文件")
        if not npy_path.is_file():
            raise FileNotFoundError(f"未找到 chunk 数据文件: {npy_path}")

        chunk_data = np.load(npy_path)
        if chunk_data.shape[0] != len(df):
            raise ValueError(
                f"CSV 行数 ({len(df)}) 与 npy 步数 ({chunk_data.shape[0]}) 不一致: "
                f"{csv_path.name} / {npy_path.name}"
            )

        pred_fused_all = pred_from_chunk_fusion(chunk_data)
        if pred_fused_all.shape[1] <= ACTION_X_AXIS:
            raise ValueError(f"{npy_path} 的 action 维度不足以绘制 action x")
        pred_fused = pred_fused_all[:, ACTION_X_AXIS]

    span_col = primary_span_col(signal_cols)
    motion_values = df[span_col].values
    motion_spans = finalize_motion_spans(
        motion_spans_from_extrema(motion_values)
    )

    return TrajectoryData(
        label=label,
        df=df,
        steps=len(df),
        gt=df[f"gt_action_{ACTION_X_AXIS}"].values,
        pred_csv=df[f"pred_action_{ACTION_X_AXIS}"].values,
        signal_cols=signal_cols,
        motion_values=motion_values,
        motion_spans=motion_spans,
        pred_fused=pred_fused,
        fused_only=fused_only,
    )


def load_trajectory(
    traj_id: int,
    model: str,
    use_fusion: bool = False,
    fused_only: bool = False,
) -> TrajectoryData:
    csv_path = csv_path_for_run(traj_id, model)
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    df = trim_trailing_invalid_rows(pd.read_csv(csv_path))
    need_npy = use_fusion or fused_only
    npy_path = npy_path_for_run(traj_id, model) if need_npy else None
    return build_trajectory_data(
        df, csv_path, model, use_fusion, npy_path, fused_only=fused_only
    )


def infer_npy_path_for_csv(csv_path: Path) -> Path | None:
    """推断与 plot CSV 配对的 chunk npy，如 trajectory_1_plot.csv -> trajectory_1_chunk.npy。"""
    stem = csv_path.stem
    candidates = []
    if stem.endswith("_plot"):
        candidates.append(csv_path.with_name(stem.replace("_plot", "_chunk") + ".npy"))
    candidates.extend(
        [
            csv_path.with_name(stem + "_chunk.npy"),
            csv_path.with_suffix(".npy"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_trajectory_csv(
    csv_path: Path,
    use_fusion: bool = False,
    npy_path: Path | None = None,
    fused_only: bool = False,
) -> TrajectoryData:
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    if (use_fusion or fused_only) and npy_path is None:
        npy_path = infer_npy_path_for_csv(csv_path)

    df = trim_trailing_invalid_rows(pd.read_csv(csv_path))
    return build_trajectory_data(
        df, csv_path, csv_path.stem, use_fusion, npy_path, fused_only=fused_only
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

    if data.fused_only:
        if data.pred_fused is None:
            raise ValueError("fused_only 模式需要融合 pred，请使用 --fuse 或 --fused-only")
        ax.plot(
            data.pred_fused,
            color=COLORS["pred_csv"],
            linewidth=2.2,
            alpha=0.95,
            label=lbl("pred action"),
            zorder=4,
        )
        pred_series = [
            (
                abs_slope(data.pred_fused),
                COLORS["pred_csv_slope"],
                COLORS["pred_csv"],
                "pred action slope",
            ),
        ]
    else:
        ax.plot(
            data.pred_csv,
            color=COLORS["pred_csv"],
            linewidth=2.2,
            alpha=0.95,
            label=lbl(
                "pred action (csv)" if data.pred_fused is not None else "pred action"
            ),
            zorder=4,
        )
        pred_series = [
            (
                abs_slope(data.pred_csv),
                COLORS["pred_csv_slope"],
                COLORS["pred_csv"],
                "pred action slope (csv)"
                if data.pred_fused is not None
                else "pred action slope",
            ),
        ]
        if data.pred_fused is not None:
            ax.plot(
                data.pred_fused,
                color=COLORS["pred_fused"],
                linewidth=2.2,
                alpha=0.95,
                label=lbl("pred action (fused)"),
                zorder=4,
            )
            pred_series.append(
                (
                    abs_slope(data.pred_fused),
                    COLORS["pred_fused_slope"],
                    COLORS["pred_fused"],
                    "pred action slope (fused)",
                )
            )

    gt_slope = abs_slope(data.gt)

    ax2 = ax.twinx()

    for signal_col in data.signal_cols:
        ax2.plot(
            data.df[signal_col].values,
            color=COLORS["motion"],
            linewidth=MOTION_LW,
            linestyle=(0, (5, 4)),
            alpha=MOTION_ALPHA,
            label=lbl(MOTION_LABELS.get(signal_col, signal_col)),
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

    legend_ncol = min(len(legend_labels), 5)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
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
    model_help = ", ".join(MODELS)
    parser = argparse.ArgumentParser(
        description="绘制 action x 图；默认仅用 CSV pred，--fuse 时叠加 chunk k 融合 pred"
    )
    parser.add_argument(
        "traj_id",
        nargs="?",
        type=int,
        choices=TRAJECTORY_IDS,
        help=f"轨迹编号（与 model 搭配），可选: {id_help}",
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=MODELS,
        help=f"模型名称（与 traj_id 搭配），可选: {model_help}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="直接指定 CSV 路径，如 trajectory/19/trajectory_1_plot.csv",
    )
    parser.add_argument(
        "--npy",
        type=Path,
        default=None,
        help="chunk npy 路径；--fuse 时可省略，将自动查找同目录 *_chunk.npy",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="从 npy 做 k 指数权重融合，并与 CSV pred 同时绘制",
    )
    parser.add_argument(
        "--fused-only",
        action="store_true",
        help="仅绘制 k 融合 pred（不画 CSV pred）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 PNG 路径",
    )
    return parser.parse_args()


def resolve_input(args):
    use_fusion = args.fuse or args.fused_only
    if args.fused_only and args.fuse:
        raise SystemExit("--fuse 与 --fused-only 请二选一")

    if args.csv is not None:
        csv_path = args.csv
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        output_path = args.output or csv_path.with_suffix(".png")
        data = load_trajectory_csv(
            csv_path,
            use_fusion=use_fusion,
            npy_path=args.npy,
            fused_only=args.fused_only,
        )
        return data, output_path

    if args.traj_id is None or args.model is None:
        raise SystemExit("请指定 traj_id 和 model，或使用 --csv 指定 CSV 文件")

    output_path = args.output or output_path_for_run(args.traj_id, args.model)
    data = load_trajectory(
        args.traj_id,
        args.model,
        use_fusion=use_fusion,
        fused_only=args.fused_only,
    )
    return data, output_path


def main():
    args = parse_args()
    data, output_path = resolve_input(args)
    render_figure(data, output_path)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
