# autosurgery

自主手术相关论文图表、专利材料与试验数据仓库。

## 目录结构

```
autosurgery/
├── papers/                 # 参考文献与论文稿
├── patent/                 # 专利文稿与 TikZ 图
├── figures/                # 成图与素材
│   ├── figure.pptx         # 主图稿
│   └── anomaly/            # 异常图解（按类型分子目录）
├── scripts/                # 绘图脚本（Figure B–E）
└── data/
    ├── docs/               # 数据说明文档
    ├── trials/             # 轨迹试验（trial_17 / 18 / 19）
    │   └── trial_XX/
    │       ├── inputs/     # 模型 CSV / NPY
    │       ├── outputs/    # 生成的图
    │       └── video/      # 对应视频
    └── sequences/          # 图像序列（帧 + 视频）
```

## 命名约定

| 类型 | 规则 | 示例 |
|------|------|------|
| 目录 | 小写英文 + 下划线 | `data/trials/trial_17` |
| 脚本 | `figure_<字母>_<简述>.py` | `figure_B_tissue_motion.py` |
| 试验输出 | `figure_<字母>_<简述>.png` | `figure_E_actionx_models.png` |
| 异常素材 | `{type}_{stage}_{view}.jpg` | `gripper_slip_anomaly_ui.jpg` |

异常类型：`dissection` / `tissue_contact` / `gripper_slip` / `progress_stall` / `unknown`  
阶段：`normal` / `anomaly` / `recovery` / `return_normal`  
视图：`ui` / `image`

## 绘图脚本

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/run_figures.py
```

统一入口默认依次生成 Figure B、C、DE、D、E，并覆盖 trial 17、18、19。也可以只运行
指定图或 trial；`--list` 仅打印将执行的命令，适合提交前快速核对：

```bash
python scripts/run_figures.py --figures B C DE --trial 18
python scripts/run_figures.py --list
```

需要精细调整单张图时，仍可直接执行原脚本：

```bash
python scripts/figure_B_tissue_motion.py --all
python scripts/figure_C_3d_trajectories.py
python scripts/figure_DE_actionx_phase_velocity.py --all
python scripts/figure_D_motion_phase_velocity.py --all
python scripts/figure_E_actionx_pairwise.py --all
python scripts/figure_CE_trajectory_actionx_1x2.py --trial 18
```

默认从 `data/trials/trial_*/inputs/` 读入，写出到同 trial 的 `outputs/`。
**改完任何绘图脚本都要跑全部三个 trial（17 / 18 / 19），不要只出一个。**

主图是两张分开的图：

- `figure_C_3d_trajectories.py` —— 3D 轨迹，演示轨迹 + 四个模型 + 端点容差球。
- `figure_DE_actionx_phase_velocity.py` —— Action-X 曲线、呼吸相底色、相位平均
  速度柱合并在一个坐标系里。曲线在左轴（m），速度柱在右轴（m/s）。横轴不标秒数，
  改标呼吸相序数，刻度落在各段中心。

`figure_CE_trajectory_actionx_1x2.py` 是这两张图的 1×2 合版，保留备用。

横轴标签的形式是**全图统一**自动选的：按相邻标签中心间距挑出互不碰撞的最长形式，
所以 trial 17（7 段）用 `First / Inspiration` 全称，trial 18（11 段）自动降为
`1st / Insp.`。绝不会在同一行里混用两种形式。

模型配色的唯一来源是 `figure_C_3d_trajectories.COLORS`，`figure_DE` / `figure_CE`
都从它导入，改一处三张图同步。当前用的是 `jewel`（灰翠蓝 / 柔金 / 青瓷 / 酒红，
四个降饱和色）。`figure_DE --palette {jewel,npg,slate,neutral}` 可以现场比选，
选定后把色值写回 `figure_C.COLORS` 即可全局生效。

字体走 `resolve_sans_family()`：优先 Arial，缺失时回退到与其等宽等形的 Arimo /
Liberation Sans，排版不会变形。装了真 Arial 后脚本会自动改用它，无需改代码：

```bash
sudo apt install -y ttf-mscorefonts-installer && fc-cache -f && rm -f ~/.cache/matplotlib/fontlist-*.json
```

`figure_B` / `figure_D` / `figure_DE` / `figure_CE` 都支持 `--height`（英寸），只压
绘图框、不动字号。`figure_DE` 另有 `--caption` 开关，默认关闭以省高度。

## 脚本职责

| 脚本 | 用途 |
|------|------|
| `run_figures.py` | 统一批量入口与失败即停的执行检查 |
| `extract_tissue_motion.py` | 从视频提取呼吸组织位移曲线 |
| `figure_B_tissue_motion.py` | Figure B 组织运动曲线 |
| `figure_C_3d_trajectories.py` | Figure C 三维轨迹 |
| `figure_D_motion_phase_velocity.py` | Figure D 呼吸相与速度 |
| `figure_DE_actionx_phase_velocity.py` | Action-X 与相位速度合图 |
| `figure_E_actionx_pairwise.py` | Figure E 模型两两比较 |
| `figure_CE_trajectory_actionx_1x2.py` | C/DE 的 1×2 备用合版 |
| `figure_anomaly_probability.py` | 异常分类概率图 |
| `figure_output_probability.py` | CVS 输出概率图 |
| `annotate_respiratory_phase_video.py` | 给视频叠加呼吸相标签 |
| `overlay_tissue_motion_video.py` | 给视频叠加组织运动曲线 |

## 仓库体积约定

- `data/trials/*/outputs/`、`figures/palette_preview/` 和派生视频均可由脚本重新生成，默认不提交。
- PPTX、PDF、MP4 使用 Git LFS；首次 clone 后需要安装 Git LFS 并运行 `git lfs install`。
- 不要反复把同一个大型二进制文件作为普通 Git blob 提交。普通 Git 会永久保留每一版，
  即使工作区最终文件不大，也会让 clone 下载完整历史。
- 如只需代码，可使用 `git clone --filter=blob:none --no-checkout <repo-url>`，再按需检出。
