import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


traj_id = 0
action_horizon = 16

# ==================================
# 读取数据
# ==================================

df = pd.read_csv(
    f"trajectory_{traj_id}_plot.csv"
)

chunk_data = np.load(
    f"trajectory_{traj_id}_chunk.npy"
)

# ==================================
# 自动统计维度
# ==================================

gt_cols = sorted(
    [c for c in df.columns if c.startswith("gt_action_")]
)

pred_cols = sorted(
    [c for c in df.columns if c.startswith("pred_action_")]
)

state_cols = sorted(
    [c for c in df.columns if c.startswith("state_")]
)

action_dim = len(gt_cols)

steps = len(df)

# ==================================
# 画图
# ==================================

fig, axes = plt.subplots(
    nrows=action_dim,
    ncols=1,
    figsize=(8, 4 * action_dim)
)

if action_dim == 1:
    axes = [axes]

for i, ax in enumerate(axes):

    gt = df[f"gt_action_{i}"].values
    pred = df[f"pred_action_{i}"].values

    ax.plot(
        gt,
        label="gt action"
    )

    ax.plot(
        pred,
        label="pred action"
    )

    # --------------------------------
    # state
    # --------------------------------

    ax2 = ax.twinx()

    for state_col in state_cols:

        ax2.plot(
            df[state_col].values,
            "--",
            alpha=0.8
        )

    # --------------------------------
    # inference point
    # --------------------------------

    for j in range(
        0,
        steps,
        action_horizon
    ):

        if j == 0:
            ax.plot(
                j,
                gt[j],
                "ro",
                label="inference point"
            )
        else:
            ax.plot(
                j,
                gt[j],
                "ro"
            )

    # --------------------------------
    # chunk point
    # --------------------------------

    for k in range(
        0,
        steps,
        3
    ):

        y = chunk_data[k, :, i]
        x = np.full(
            len(y),
            k
        )

        if k == 0:
            ax.plot(
                x,
                y,
                "ko",
                ms=1,
                label="chunk"
            )
        else:
            ax.plot(
                x,
                y,
                "ko",
                ms=1
            )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(
        lines1 + lines2,
        labels1 + labels2
    )

    ax.set_title(
        f"Action {i}"
    )

plt.tight_layout()

plt.savefig(
    f"trajectory_{traj_id}_replot.png"
)

plt.show()