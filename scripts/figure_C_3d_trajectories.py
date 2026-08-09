"""Generate Figure C: 3D action trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from PIL import Image, ImageChops
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import LinearLocator, MaxNLocator, NullFormatter
from mpl_toolkits.mplot3d import proj3d


def resolve_sans_family(
    preferred: tuple[str, ...] = (
        "Arial",
        "Helvetica",
        "Arimo",
        "Liberation Sans",
        "DejaVu Sans",
    )
) -> str:
    """Return the first installed family from ``preferred``.

    Arial is absent on most Linux boxes.  Arimo and Liberation Sans are
    metric-compatible with it, so falling back keeps every label the same
    width and the tuned inch-based layouts keep working untouched.
    """
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


SANS = resolve_sans_family()


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "trials"
TRIAL_IDS = (17, 18, 19)


def trial_inputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "inputs"


def trial_outputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "outputs"
MODEL_FILES = {
    "zeroshot": "zeroshot",
    "AB": "AB",
    "ABC": "ABC",
    "ABCP": "ABCP",
}
DISPLAY_LABELS = {
    "zeroshot": "Zeroshot",
    "AB": "Ex vivo",
    "ABC": "In vivo",
    "ABCP": "In vivo-P",
}
# Four desaturated hues, well separated but none of them bright: the quiet
# look Nature figures tend to have once they are printed.  In vivo-P takes the
# wine red so it stays legible where it rides on top of the near-black
# demonstration.  This dict is the single source of truth -- figure_DE and
# figure_CE import it, so every panel of the paper agrees.  figure_DE keeps
# the rejected alternates behind its --palette flag.
COLORS = {
    "GT": "#1A1A1A",
    "zeroshot": "#CBC6BC",
    "AB": "#A2998B",
    "ABC": "#776D5F",
    "ABCP": "#1F6F6B",
}
# A low camera keeps the box floor almost edge-on, which is what used to eat
# the lower-left corner as an empty wedge.  The trajectory still runs from the
# lower right up to the left, leaving the upper-right quadrant clear.
DEFAULT_ELEV = 8.0
DEFAULT_AZIM = -56.0
LINEWIDTHS = {"GT": 1.95, "zeroshot": 0.95, "AB": 0.95, "ABC": 1.1, "ABCP": 1.4}

# The panel is 8.8 in wide, so what reads as a comfortable size here is larger
# than the 8.5 pt figure_DE uses on its 7 in canvas.
FONT_SIZE = 11.5

# The tolerance sphere is centred on the demonstration endpoint and its radius
# is the In vivo endpoint error, so it borrows that model's colour and follows
# whatever palette is in force.  The hue is tinted towards white first: a dark
# palette colour at a low alpha still stacks into a heavy, solid-looking ball,
# whereas tinting keeps the sphere equally airy whichever palette is active.
SPHERE_TINT = 0.62
SPHERE_FACE_ALPHA = 0.20
SPHERE_EDGE_ALPHA = 0.42


def tint(color: str, amount: float) -> str:
    """Blend ``color`` towards white by ``amount`` in [0, 1]."""
    rgb = np.asarray(mcolors.to_rgb(color))
    return mcolors.to_hex(rgb + (1.0 - rgb) * amount)


def sphere_color() -> str:
    return tint(COLORS["ABC"], SPHERE_TINT)

rcParams.update(
    {
        "font.family": SANS,
        "font.sans-serif": [SANS, "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": SANS,
        "mathtext.it": f"{SANS}:italic",
        "mathtext.bf": f"{SANS}:bold",
        # The custom fontset also resolves cal/sf/tt; leaving them at their
        # defaults makes matplotlib hunt for a missing 'cursive' family.
        "mathtext.cal": f"{SANS}:italic",
        "mathtext.sf": SANS,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.linewidth": 0.65,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.fontsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def fuse_chunks(chunks: np.ndarray, k: float = 0.03) -> np.ndarray:
    """Exponentially fuse all chunk predictions that cover each time step.

    This reproduces the temporal aggregation used in ``plot_from_csv.py``:
    predictions are ordered from the oldest chunk to the newest and weighted
    with exp(-k * i).
    """
    if chunks.ndim != 3:
        raise ValueError(f"Expected (steps, chunk_horizon, xyz), got {chunks.shape}")
    steps, horizon, action_dim = chunks.shape
    if action_dim < 3:
        raise ValueError(f"At least three action dimensions are required, got {action_dim}")

    fused = np.empty((steps, action_dim), dtype=np.float64)
    for t in range(steps):
        first_chunk = max(0, t - horizon + 1)
        chunk_indices = np.arange(first_chunk, t + 1)
        horizon_indices = t - chunk_indices
        candidates = chunks[chunk_indices, horizon_indices]
        weights = np.exp(-k * np.arange(len(candidates), dtype=np.float64))
        weights /= weights.sum()
        fused[t] = np.sum(candidates * weights[:, None], axis=0)
    return fused


def load_gt(csv_path: Path, steps: int | None = None) -> np.ndarray:
    df = pd.read_csv(csv_path)
    columns = [f"gt_action_{axis}" for axis in range(3)]
    missing = [column for column in columns if column not in df]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    values = df[columns].to_numpy(dtype=np.float64)
    return values if steps is None else values[:steps]


def style_3d_axes(ax: plt.Axes, trajectories: list[np.ndarray]) -> None:
    points = np.concatenate([trajectory[:, :3] for trajectory in trajectories], axis=0)
    low, high = points.min(axis=0), points.max(axis=0)
    spans = np.maximum(high - low, np.finfo(float).eps)
    padding = spans * 0.055
    ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
    ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
    ax.set_zlim(low[2] - padding[2], high[2] + padding[2])

    # Box dimensions proportional to data spans preserve equal physical scale
    # per action unit; the floor keeps shorter axes readable without flattening z.
    box = spans / spans.max()
    ax.set_box_aspect(np.maximum(box, 0.48))
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#B8B8B8")
        axis.pane.set_alpha(0.0)
        axis._axinfo["grid"].update(
            {"color": (0.78, 0.78, 0.78, 0.42), "linewidth": 0.45}
        )
        axis._axinfo["axisline"].update(
            {"color": (0.0, 0.0, 0.0, 1), "linewidth": 0.65}
        )
        axis._axinfo["tick"].update(
            {"inward_factor": 0.0, "outward_factor": 0.18}
        )


def draw_black_box_silhouette(ax: plt.Axes) -> None:
    """Draw only the projected outer silhouette of the 3D plotting box."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()
    corners = [
        np.array((x, y, z))
        for x in (x0, x1)
        for y in (y0, y1)
        for z in (z0, z1)
    ]
    projected = []
    for index, corner in enumerate(corners):
        px, py, _ = proj3d.proj_transform(*corner, ax.get_proj())
        projected.append((float(px), float(py), index))

    def cross(origin, a, b) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    points = sorted(projected)
    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]

    for current, following in zip(hull, hull[1:] + hull[:1]):
        start = corners[current[2]]
        end = corners[following[2]]
        ax.plot(
            (start[0], end[0]),
            (start[1], end[1]),
            (start[2], end[2]),
            color="black",
            lw=0.65,
            zorder=0.5,
        )


def _project_to_axes_fraction(ax: plt.Axes, point: np.ndarray) -> np.ndarray:
    x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
    display = ax.transData.transform((x2, y2))
    return ax.transAxes.inverted().transform(display)


def _axis_screen_rotation(ax: plt.Axes, start: np.ndarray, end: np.ndarray) -> float:
    start_frac = _project_to_axes_fraction(ax, start)
    end_frac = _project_to_axes_fraction(ax, end)
    delta = end_frac - start_frac
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _perpendicular_offset(
    start_frac: np.ndarray, end_frac: np.ndarray, distance: float
) -> np.ndarray:
    delta = end_frac - start_frac
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        return np.array([0.0, 0.0])
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    return normal * distance


def _readable_rotation(rotation: float) -> float:
    if rotation > 90.0 or rotation < -90.0:
        return rotation + 180.0
    return rotation


def _outward_axis_offset(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    anchor: np.ndarray,
    distance: float,
) -> np.ndarray:
    start_frac = _project_to_axes_fraction(ax, start)
    end_frac = _project_to_axes_fraction(ax, end)
    anchor_frac = _project_to_axes_fraction(ax, anchor)
    offset = _perpendicular_offset(start_frac, end_frac, distance)
    center_frac = np.array([0.5, 0.5])
    if np.dot(offset, anchor_frac - center_frac) < 0.0:
        offset = -offset
    return offset


def _y_axis_label_segment(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the front-right y edge so the label sits on the visible receding axis."""
    candidates = (
        (np.array([x1, y0, z0]), np.array([x1, y1, z0])),
        (np.array([x0, y0, z0]), np.array([x0, y1, z0])),
        (np.array([x1, y0, z1]), np.array([x1, y1, z1])),
        (np.array([x0, y0, z1]), np.array([x0, y1, z1])),
    )

    def segment_score(segment: tuple[np.ndarray, np.ndarray]) -> float:
        start_frac = _project_to_axes_fraction(ax, segment[0])
        end_frac = _project_to_axes_fraction(ax, segment[1])
        mid_frac = 0.5 * (start_frac + end_frac)
        length = float(np.linalg.norm(end_frac - start_frac))
        return length + 0.45 * (mid_frac[0] - mid_frac[1])

    return max(candidates, key=segment_score)


def _rightmost_z_axis_segment(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the vertical box edge that appears farthest right in the current view."""
    candidates = (
        (np.array([x0, y0, z0]), np.array([x0, y0, z1])),
        (np.array([x0, y1, z0]), np.array([x0, y1, z1])),
        (np.array([x1, y0, z0]), np.array([x1, y0, z1])),
        (np.array([x1, y1, z0]), np.array([x1, y1, z1])),
    )
    return max(
        candidates,
        key=lambda segment: 0.5
        * (
            _project_to_axes_fraction(ax, segment[0])[0]
            + _project_to_axes_fraction(ax, segment[1])[0]
        ),
    )


def add_view_aligned_axis_labels(
    ax: plt.Axes,
    labels: tuple[str, str, str] = (
        r"$x$ (m)",
        r"$y$ (m)",
        r"$z$ (m)",
    ),
    fontsize: float = FONT_SIZE,
) -> None:
    """Place axis labels near the visible 3D axis edges for the current view."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()

    y_axis_start, y_axis_end = _y_axis_label_segment(ax, x0, x1, y0, y1, z0, z1)
    y_axis_anchor = y_axis_start + 0.68 * (y_axis_end - y_axis_start)
    y_rotation = _readable_rotation(
        _axis_screen_rotation(ax, y_axis_start, y_axis_end)
    )
    y_offset = _outward_axis_offset(
        ax, y_axis_start, y_axis_end, y_axis_anchor, 0.026
    )

    x_axis_start = np.array([x0, y0, z0])
    x_axis_end = np.array([x1, y0, z0])
    x_axis_anchor = x_axis_start + 0.55 * (x_axis_end - x_axis_start)
    x_rotation = _readable_rotation(
        _axis_screen_rotation(ax, x_axis_start, x_axis_end)
    )
    x_offset = _outward_axis_offset(
        ax, x_axis_start, x_axis_end, x_axis_anchor, 0.026
    )

    z_axis_start, z_axis_end = _rightmost_z_axis_segment(
        ax, x0, x1, y0, y1, z0, z1
    )
    z_axis_anchor = z_axis_start + 0.67 * (z_axis_end - z_axis_start)
    z_rotation = 90.0
    z_offset = np.array([0.035, 0.0])

    label_specs = (
        (x_axis_anchor, labels[0], x_offset, x_rotation),
        (y_axis_anchor, labels[1], y_offset, y_rotation),
        (z_axis_anchor, labels[2], z_offset, z_rotation),
    )
    for anchor, label, offset, rotation in label_specs:
        frac = _project_to_axes_fraction(ax, anchor)
        frac = frac + offset
        ax.text2D(
            frac[0],
            frac[1],
            label,
            transform=ax.transAxes,
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#333333",
        )


VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D trajectory viewer</title>
<style>
  html, body { margin: 0; width: 100%; height: 100%; font-family: Arial, sans-serif; background: #fff; color: #222; }
  #wrap { display: grid; grid-template-rows: auto 1fr; height: 100%; }
  #toolbar { display: flex; align-items: center; gap: 18px; padding: 12px 16px; border-bottom: 1px solid #ddd; flex-wrap: wrap; }
  #toolbar label { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; }
  #toolbar input[type="range"] { width: 190px; }
  #readout { margin-left: auto; font-size: 14px; color: #333; }
  #canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  #canvas.dragging { cursor: grabbing; }
  .hint { color: #666; font-size: 13px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <label>Azim <input id="azim" type="range" min="-180" max="180" step="1" value="-61"></label>
    <label>Elev <input id="elev" type="range" min="-90" max="90" step="1" value="19"></label>
    <label>Zoom <input id="zoom" type="range" min="0.5" max="3" step="0.01" value="1"></label>
    <button id="reset" type="button">Reset</button>
    <span class="hint">Drag to rotate, wheel to zoom. Use the values on the right for Matplotlib view_init.</span>
    <span id="readout"></span>
  </div>
  <canvas id="canvas"></canvas>
</div>
<script>
const DATA = __DATA__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const azimInput = document.getElementById('azim');
const elevInput = document.getElementById('elev');
const zoomInput = document.getElementById('zoom');
const readout = document.getElementById('readout');
let azim = Number(azimInput.value);
let elev = Number(elevInput.value);
let zoom = Number(zoomInput.value);
let dragging = false;
let last = null;

const allPoints = DATA.series.flatMap(s => s.points);
const bounds = [0, 1, 2].map(i => [Math.min(...allPoints.map(p => p[i])), Math.max(...allPoints.map(p => p[i]))]);
const center = bounds.map(b => (b[0] + b[1]) / 2);
const span = Math.max(...bounds.map(b => Math.max(b[1] - b[0], 1e-12)));

function resize() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  draw();
}

function rotate(point) {
  const a = azim * Math.PI / 180;
  const e = elev * Math.PI / 180;
  const x = (point[0] - center[0]) / span;
  const y = (point[1] - center[1]) / span;
  const z = (point[2] - center[2]) / span;
  const ca = Math.cos(a), sa = Math.sin(a);
  const ce = Math.cos(e), se = Math.sin(e);
  const x1 = ca * x - sa * y;
  const y1 = sa * x + ca * y;
  const z1 = z;
  return [x1, ce * y1 - se * z1, se * y1 + ce * z1];
}

function project(point) {
  const [x, y, z] = rotate(point);
  const size = Math.min(canvas.width, canvas.height) * 0.82 * zoom;
  return [canvas.width / 2 + x * size, canvas.height / 2 - y * size, z];
}

function drawGrid() {
  ctx.save();
  ctx.strokeStyle = 'rgba(180,180,180,0.35)';
  ctx.lineWidth = 1;
  const ticks = 5;
  for (let axis = 0; axis < 3; axis++) {
    for (let i = 0; i < ticks; i++) {
      const t = bounds[axis][0] + (bounds[axis][1] - bounds[axis][0]) * i / (ticks - 1);
      const p1 = [bounds[0][0], bounds[1][0], bounds[2][0]];
      const p2 = [bounds[0][0], bounds[1][0], bounds[2][0]];
      p1[axis] = t; p2[axis] = t;
      const other = (axis + 1) % 3;
      p2[other] = bounds[other][1];
      const a = project(p1), b = project(p2);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
  }
  ctx.restore();
}

function drawSphere() {
  if (!DATA.sphere || DATA.sphere.radius <= 0) return;
  const c = DATA.sphere.center;
  const r = DATA.sphere.radius;
  ctx.save();
  ctx.strokeStyle = 'rgba(92,145,181,0.35)';
  ctx.fillStyle = 'rgba(140,187,217,0.10)';
  ctx.lineWidth = 1;
  for (let ring = 0; ring < 3; ring++) {
    ctx.beginPath();
    for (let i = 0; i <= 96; i++) {
      const t = i / 96 * Math.PI * 2;
      const p = c.slice();
      if (ring === 0) { p[0] += r * Math.cos(t); p[1] += r * Math.sin(t); }
      if (ring === 1) { p[0] += r * Math.cos(t); p[2] += r * Math.sin(t); }
      if (ring === 2) { p[1] += r * Math.cos(t); p[2] += r * Math.sin(t); }
      const q = project(p);
      if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawSeries(series) {
  const projected = series.points.map(project);
  ctx.save();
  ctx.strokeStyle = series.color;
  ctx.lineWidth = series.width;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  projected.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.stroke();
  const end = projected[projected.length - 1];
  ctx.fillStyle = series.color;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(end[0], end[1], series.name === 'In vivo-P' ? 6 : 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.restore();
}

function drawLegend() {
  ctx.save();
  ctx.font = `${15 * (window.devicePixelRatio || 1)}px Arial`;
  ctx.textBaseline = 'middle';
  let x = 28 * (window.devicePixelRatio || 1), y = 30 * (window.devicePixelRatio || 1);
  for (const s of DATA.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 34, y); ctx.stroke();
    ctx.fillStyle = '#111'; ctx.fillText(s.name, x + 44, y);
    x += ctx.measureText(s.name).width + 92;
  }
  ctx.restore();
}

function drawLabels() {
  const gt = DATA.series[0].points;
  const start = project(gt[0]);
  const end = project(gt[gt.length - 1]);
  ctx.save();
  ctx.font = `${16 * (window.devicePixelRatio || 1)}px Arial`;
  ctx.fillStyle = '#333';
  ctx.strokeStyle = '#1A1A1A';
  ctx.lineWidth = 2;
  ctx.fillText('Start', start[0] + 8, start[1]);
  ctx.fillText('End', end[0] + 8, end[1]);
  ctx.restore();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  drawSphere();
  const ordered = [...DATA.series].sort((a, b) => a.order - b.order);
  ordered.forEach(drawSeries);
  drawLabels();
  drawLegend();
  readout.textContent = `elev=${elev.toFixed(0)}, azim=${azim.toFixed(0)}, zoom=${zoom.toFixed(2)}`;
}

function syncInputs() {
  azimInput.value = String(Math.round(azim));
  elevInput.value = String(Math.round(elev));
  zoomInput.value = String(zoom.toFixed(2));
  draw();
}

azimInput.addEventListener('input', () => { azim = Number(azimInput.value); draw(); });
elevInput.addEventListener('input', () => { elev = Number(elevInput.value); draw(); });
zoomInput.addEventListener('input', () => { zoom = Number(zoomInput.value); draw(); });
document.getElementById('reset').addEventListener('click', () => { azim = -61; elev = 19; zoom = 1; syncInputs(); });
canvas.addEventListener('pointerdown', e => { dragging = true; last = [e.clientX, e.clientY]; canvas.classList.add('dragging'); canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (!dragging) return;
  const dx = e.clientX - last[0], dy = e.clientY - last[1];
  azim = Math.max(-180, Math.min(180, azim + dx * 0.45));
  elev = Math.max(-90, Math.min(90, elev - dy * 0.35));
  last = [e.clientX, e.clientY];
  syncInputs();
});
canvas.addEventListener('pointerup', e => { dragging = false; canvas.classList.remove('dragging'); canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom = Math.max(0.5, Math.min(3, zoom * (e.deltaY < 0 ? 1.08 : 0.92))); syncInputs(); }, { passive: false });
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def save_interactive_plot(
    gt: np.ndarray,
    trajectories: dict[str, np.ndarray],
    output: Path,
) -> None:
    sphere_center = gt[-1, :3]
    sphere_radius = float(
        np.linalg.norm(trajectories["ABC"][-1, :3] - sphere_center)
    )
    series = [
        {
            "name": "Demonstration",
            "points": gt[:, :3].tolist(),
            "color": COLORS["GT"],
            "width": 5.0,
            "order": 10,
        }
    ]
    for label, trajectory in trajectories.items():
        series.append(
            {
                "name": DISPLAY_LABELS[label],
                "points": trajectory[:, :3].tolist(),
                "color": COLORS[label],
                "width": 3.5 if label != "ABCP" else 4.5,
                "order": 5 if label == "ABCP" else 3,
            }
        )

    payload = {
        "series": series,
        "sphere": {"center": sphere_center.tolist(), "radius": sphere_radius},
    }
    html = VIEWER_TEMPLATE.replace("__DATA__", json.dumps(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved interactive: {output}")


def trim_png_whitespace(path: Path, padding: int = 28) -> None:
    """Crop excess white margins from a saved PNG while keeping a small border."""
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    bbox = ImageChops.difference(image, background).getbbox()
    if bbox is None:
        return
    left, top, right, bottom = bbox
    left = max(left - padding, 0)
    top = max(top - padding, 0)
    right = min(right + padding, image.width)
    bottom = min(bottom + padding, image.height)
    image.crop((left, top, right, bottom)).save(path)

def plot(
    trial_dir: Path,
    output: Path,
    k: float,
    dpi: int,
    interactive_output: Path | None = None,
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
) -> dict[str, tuple[float, float]]:
    npy_paths = [trial_dir / f"{stem}.npy" for stem in MODEL_FILES.values()]
    step_counts = [np.load(path, mmap_mode="r", allow_pickle=False).shape[0] for path in npy_paths]
    if len(set(step_counts)) != 1:
        raise ValueError(f"Prediction NPY files have different lengths: {step_counts}")
    steps = step_counts[0]
    gt = load_gt(trial_dir / "ABCP.csv", steps=steps)
    trajectories: dict[str, np.ndarray] = {}

    for label, stem in MODEL_FILES.items():
        csv_path = trial_dir / f"{stem}.csv"
        npy_path = trial_dir / f"{stem}.npy"
        if not csv_path.is_file() or not npy_path.is_file():
            raise FileNotFoundError(f"Missing input pair: {csv_path.name}, {npy_path.name}")

        model_gt = load_gt(csv_path, steps=steps)
        if model_gt.shape != gt.shape or not np.allclose(model_gt, gt, atol=1e-8, rtol=0):
            raise ValueError(f"{csv_path.name} does not contain the same GT trajectory")

        chunks = np.load(npy_path, allow_pickle=False)
        if chunks.shape[0] != len(gt):
            raise ValueError(
                f"{npy_path.name}: {chunks.shape[0]} steps, but GT has {len(gt)}"
            )
        trajectories[label] = fuse_chunks(chunks, k=k)

    fig = plt.figure(figsize=(8.8, 3.05))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_position([0.015, 0.02, 0.97, 0.84])

    # Endpoint tolerance sphere: its radius is the ABC endpoint error, so ABC
    # lies on the boundary and better/worse endpoints fall inside/outside.
    sphere_center = gt[-1, :3]
    sphere_radius = float(
        np.linalg.norm(trajectories["ABC"][-1, :3] - sphere_center)
    )
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    sphere_x = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = sphere_center[2] + sphere_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color=sphere_color(),
        alpha=SPHERE_FACE_ALPHA,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=1,
    )
    ax.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=6,
        cstride=6,
        color=sphere_color(),
        alpha=SPHERE_EDGE_ALPHA,
        linewidth=0.35,
        zorder=2,
    )

    ax.plot(
        *gt.T,
        color=COLORS["GT"],
        lw=LINEWIDTHS["GT"],
        label="Demonstration",
        zorder=10,
    )
    for label, trajectory in trajectories.items():
        ax.plot(
            *trajectory[:, :3].T,
            color=COLORS[label],
            lw=LINEWIDTHS[label],
            label=DISPLAY_LABELS[label],
            alpha=0.96,
            zorder=5 if label == "ABCP" else 3,
        )
        ax.scatter(
            *trajectory[-1, :3],
            s=18 if label == "ABCP" else 14,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.45,
            depthshade=False,
            zorder=11,
        )

    ax.scatter(*gt[0], color="white", edgecolor=COLORS["GT"], linewidth=0.9,
               s=22, marker="o", depthshade=False, zorder=12)
    ax.scatter(*gt[-1], color=COLORS["GT"], edgecolor="white", linewidth=0.55,
               s=24, marker="o", depthshade=False, zorder=12)
    ax.text(*(gt[-1] + np.array([0.00018, -0.00005, 0.00002])), "End", fontsize=FONT_SIZE, color="#333333", zorder=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.view_init(elev=elev, azim=azim)
    sphere_extent = np.vstack(
        [sphere_center - sphere_radius, sphere_center + sphere_radius]
    )
    style_3d_axes(ax, [gt, *trajectories.values(), sphere_extent])
    draw_black_box_silhouette(ax)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(LinearLocator(5))
        axis.set_major_formatter(NullFormatter())
    add_view_aligned_axis_labels(ax)
    start_frac = _project_to_axes_fraction(ax, gt[0])
    ax.text2D(
        start_frac[0] + 0.012,
        start_frac[1] + 0.018,
        "Start",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        color="#333333",
        zorder=13,
    )

    handles = [
        Line2D([0], [0], color=COLORS["GT"], lw=LINEWIDTHS["GT"], label="Demonstration"),
        Line2D([0], [0], color=COLORS["zeroshot"], lw=LINEWIDTHS["zeroshot"], label=DISPLAY_LABELS["zeroshot"]),
        Line2D([0], [0], color=COLORS["AB"], lw=LINEWIDTHS["AB"], label=DISPLAY_LABELS["AB"]),
        Line2D([0], [0], color=COLORS["ABC"], lw=LINEWIDTHS["ABC"], label=DISPLAY_LABELS["ABC"]),
        Line2D([0], [0], color=COLORS["ABCP"], lw=LINEWIDTHS["ABCP"], label=DISPLAY_LABELS["ABCP"]),
    ]
    legend_kwargs = dict(
        loc="upper center",
        frameon=False,
        handlelength=1.65,
        handletextpad=0.42,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    # Two rows rather than one: at 11.5 pt a single row of five entries is the
    # widest thing on the canvas and stretches the whole figure.  Split 3 + 2
    # with two separate legends -- one legend with ncol=3 would fill column by
    # column and scramble the model order.
    first_row = fig.legend(
        handles=handles[:3],
        bbox_to_anchor=(0.52, 0.875),
        ncol=3,
        **legend_kwargs,
    )
    fig.add_artist(first_row)
    fig.legend(
        handles=handles[3:],
        bbox_to_anchor=(0.52, 0.805),
        ncol=2,
        **legend_kwargs,
    )
    fig.subplots_adjust(left=0.005, right=0.99, bottom=0.01, top=0.99)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    trim_png_whitespace(output)
    print(f"Saved: {output}")
    if interactive_output is not None:
        save_interactive_plot(gt, trajectories, interactive_output)
    return {
        label: (
            float(np.linalg.norm(trajectory[0, :3] - gt[0])),
            float(np.linalg.norm(trajectory[-1, :3] - gt[-1])),
        )
        for label, trajectory in trajectories.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=0.03, help="chunk fusion decay")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: each trajectory's own folder)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="only save the static PNG, without the rotatable HTML viewer",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=DEFAULT_ELEV,
        help=f"camera elevation in degrees; default: {DEFAULT_ELEV}",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=DEFAULT_AZIM,
        help=f"camera azimuth in degrees; default: {DEFAULT_AZIM}",
    )
    parser.add_argument(
        "--trial",
        type=int,
        choices=TRIAL_IDS,
        default=None,
        help="plot one trial only; by default all trial ids are plotted",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_metrics: dict[int, dict[str, tuple[float, float]]] = {}
    for trial_id in (args.trial,) if args.trial else TRIAL_IDS:
        source_dir = trial_inputs_dir(trial_id)
        output_dir = trial_outputs_dir(trial_id)
        destination = (
            args.output / f"figure_C_3d_trajectory_{trial_id}.png"
            if args.output
            else output_dir / "figure_C_3d_trajectory.png"
        )
        interactive_destination = None
        if not args.no_interactive:
            interactive_destination = (
                args.output / f"figure_C_3d_trajectory_{trial_id}_interactive.html"
                if args.output
                else output_dir / "figure_C_3d_trajectory_interactive.html"
            )
        all_metrics[trial_id] = plot(
            source_dir,
            destination,
            k=args.k,
            dpi=args.dpi,
            interactive_output=interactive_destination,
            elev=args.elev,
            azim=args.azim,
        )

    print("\nEuclidean distance to GT (3D action space)")
    print(f"{'Trial':<7}{'Model':<12}{'Start':>14}{'End':>14}")
    for trial_id, model_metrics in all_metrics.items():
        for model, (start_distance, end_distance) in model_metrics.items():
            print(
                f"{trial_id:<7}{model:<12}"
                f"{start_distance:>14.9f}{end_distance:>14.9f}"
            )

