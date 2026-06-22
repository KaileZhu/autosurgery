import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path


traj_id = 0
action_horizon = 16
data_dir = Path(__file__).parent / "pku_ABCP_all"
axis_labels = ["action x", "action y", "action z", "action average"]

# ==================================
# 样式
# ==================================

COLORS = {
    "gt": "#1B4965",
    "pred": "#E76F51",
    "motion": "#7B61FF",
    "gt_slope": "#5FA8D3",
    "pred_slope": "#F4A261",
    "inference": "#C1121F",
    "chunk": "#6C757D",
}
BG = "#F8F9FA"
GRID = "#DEE2E6"
SPINE = "#ADB5BD"

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "legend.fontsize": 14,
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

chunk_data = np.load(
    data_dir / f"trajectory_{traj_id}_chunk.npy"
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

# ==================================
# 画图
# ==================================

n_plots = action_dim + 1

fig, axes = plt.subplots(
    nrows=n_plots,
    ncols=1,
    figsize=(10, 2.5 * n_plots),
    facecolor=BG,
)

if n_plots == 1:
    axes = [axes]


def abs_slope(values):
    return np.abs(np.gradient(values))


def style_axis(ax, ax2, ax3):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
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

    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.tick_params(
        axis="y",
        which="both",
        labelright=False,
        length=0,
    )


def plot_subplot(ax, gt, pred, chunk_values_fn, title, with_labels=False):
    def lbl(name):
        return name if with_labels else "_nolegend_"

    ax.plot(
        gt,
        color=COLORS["gt"],
        linewidth=2.2,
        label=lbl("gt action"),
        zorder=4,
    )
    ax.plot(
        pred,
        color=COLORS["pred"],
        linewidth=2.0,
        alpha=0.92,
        label=lbl("pred action"),
        zorder=3,
    )

    gt_slope = abs_slope(gt)
    pred_slope = abs_slope(pred)

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 0))

    for state_col in state_cols:
        ax2.plot(
            state_values,
            color=COLORS["motion"],
            linewidth=1.6,
            linestyle=(0, (6, 3)),
            alpha=0.85,
            label=lbl(state_labels.get(state_col, state_col)),
            zorder=2,
        )

    ax3.plot(
        gt_slope,
        color=COLORS["gt_slope"],
        linewidth=1.4,
        linestyle=(0, (1, 2)),
        alpha=0.75,
        label=lbl("gt action slope"),
        zorder=1,
    )
    ax3.plot(
        pred_slope,
        color=COLORS["pred_slope"],
        linewidth=1.4,
        linestyle=(0, (1, 2)),
        alpha=0.75,
        label=lbl("pred action slope"),
        zorder=1,
    )

    for j in range(0, steps, action_horizon):
        if j == 0:
            ax.scatter(
                j,
                gt[j],
                s=36,
                color=COLORS["inference"],
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
                label=lbl("inference point"),
            )
        else:
            ax.scatter(
                j,
                gt[j],
                s=28,
                color=COLORS["inference"],
                edgecolors="white",
                linewidths=0.6,
                zorder=6,
            )

    for k in range(0, steps, 3):
        y = chunk_values_fn(k)
        x = np.full(len(y), k)

        if k == 0:
            ax.scatter(
                x,
                y,
                s=4,
                color=COLORS["chunk"],
                alpha=0.55,
                linewidths=0,
                zorder=5,
                label=lbl("16-steps pred action"),
            )
        else:
            ax.scatter(
                x,
                y,
                s=4,
                color=COLORS["chunk"],
                alpha=0.55,
                linewidths=0,
                zorder=5,
            )

    ax.set_title(title, loc="left", pad=6, color="#343A40")
    style_axis(ax, ax2, ax3)
    return ax2, ax3


gt = df[f"gt_action_{0}"].values
pred = df[f"pred_action_{0}"].values
ax2_ref, ax3_ref = plot_subplot(
    axes[0],
    gt,
    pred,
    lambda k, axis=0: chunk_data[k, :, axis],
    axis_labels[0],
    with_labels=True,
)

for i in range(1, action_dim):
    gt = df[f"gt_action_{i}"].values
    pred = df[f"pred_action_{i}"].values
    plot_subplot(
        axes[i],
        gt,
        pred,
        lambda k, axis=i: chunk_data[k, :, axis],
        axis_labels[i],
    )

plot_subplot(
    axes[action_dim],
    gt_avg,
    pred_avg,
    lambda k: chunk_data[k, :, :].mean(axis=1),
    axis_labels[action_dim],
)

legend_handles = []
legend_labels = []
for legend_ax in [axes[0], ax2_ref, ax3_ref]:
    handles, labels = legend_ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label != "_nolegend_":
            legend_handles.append(handle)
            legend_labels.append(label)

legend = fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=4,
    frameon=True,
    framealpha=0.95,
    facecolor="white",
    edgecolor=GRID,
    columnspacing=1.4,
    handlelength=2.2,
    handletextpad=0.6,
)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(hspace=0.18, top=0.88)

plt.savefig(
    data_dir / f"trajectory_{traj_id}_replot.png",
    dpi=180,
    bbox_inches="tight",
    pad_inches=0.08,
    facecolor=BG,
)
