"""
exp04.py — Experiment 4: 損失 landscape の可視化

目的:
  θ = (θ₁, θ₂) の 2D グリッドで J(θ) と J_ε(θ) を評価し、
  平滑化が landscape を滑らかにすることを視覚的に示す。

設定:
  - grid_n=50, grid_range=[-3, 3]
  - dynamics_seed=0 固定
  - 2D ヒートマップ (log scale) + 3D サーフェス

出力:
  results/exp04/exp4_grid.csv           : グリッド全点の J, J_ε 値
  results/exp04/exp4_heatmap.png        : 2D ヒートマップ並列比較
  results/exp04/exp4_surface.png        : 3D サーフェス並列比較
"""

import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

from pathlib import Path

from common import (
    Dynamics,
    build_observer,
    build_target,
    cfg,
    device,
    loss_raw,
    loss_smooth,
)

RESULTS_DIR = Path(__file__).parent / "results" / "exp04"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

d           = cfg["state_dim"]
T           = cfg["T"]
grid_n      = cfg["grid_n"]
grid_range  = cfg["grid_range"]

THETA_STAR = torch.tensor(cfg["theta_star"], device=device)

# ============================================================
# モデル・ターゲット軌跡の構築
# ============================================================

torch.manual_seed(cfg["dynamics_seed"])
dyn = Dynamics(d).to(device)
obs_module = build_observer(d)
observe    = obs_module

p0_fixed, zs_target, E1, E2, a, M_base, schedule, kernel = build_target(dyn, observe)

# ============================================================
# グリッド定義
# ============================================================

theta1_vals = np.linspace(-grid_range, grid_range, grid_n)
theta2_vals = np.linspace(-grid_range, grid_range, grid_n)

J_grid     = np.zeros((grid_n, grid_n))
Jeps_grid  = np.zeros((grid_n, grid_n))

# ============================================================
# グリッド評価（勾配不要）
# ============================================================

print(f"Evaluating {grid_n}x{grid_n} grid ...")

with torch.no_grad():
    for i, t1 in enumerate(theta1_vals):
        for j, t2 in enumerate(theta2_vals):
            theta = torch.tensor([t1, t2], dtype=torch.float32, device=device)
            J_grid[i, j]    = loss_raw(
                theta, p0_fixed, zs_target, dyn, schedule, T, observe
            ).item()
            Jeps_grid[i, j] = loss_smooth(
                theta, p0_fixed, zs_target, dyn, schedule, T, observe, kernel
            ).item()

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{grid_n} rows done")

print("Grid evaluation complete.")

# ============================================================
# CSV 保存
# ============================================================

grid_path = RESULTS_DIR / "exp4_grid.csv"
with open(grid_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["theta1", "theta2", "J", "J_eps"])
    for i, t1 in enumerate(theta1_vals):
        for j, t2 in enumerate(theta2_vals):
            writer.writerow([t1, t2, J_grid[i, j], Jeps_grid[i, j]])

print(f"Saved: {grid_path}")

# θ* の grid インデックス（最近傍）
star_i = int(np.argmin(np.abs(theta1_vals - 1.0)))
star_j = int(np.argmin(np.abs(theta2_vals - 1.0)))

# ============================================================
# 図 1: 2D ヒートマップ (log scale)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, grid, title in zip(
    axes,
    [J_grid, Jeps_grid],
    ["$J(\\theta)$ — without smoothing", "$J_\\varepsilon(\\theta)$ — with smoothing"],
):
    log_grid = np.log1p(grid)   # log(1 + J) で数値的に安定
    im = ax.imshow(
        log_grid,
        origin="lower",
        extent=[-grid_range, grid_range, -grid_range, grid_range],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="log(1 + loss)")

    # θ* をマーク
    ax.scatter([1.0], [1.0], color="red", s=80, zorder=5, label="$\\theta^*=(1,1)$")
    # 最小値点をマーク
    min_idx = np.unravel_index(np.argmin(grid), grid.shape)
    min_t1  = theta1_vals[min_idx[0]]
    min_t2  = theta2_vals[min_idx[1]]
    ax.scatter([min_t2], [min_t1], color="white", marker="x", s=80, zorder=5,
               label=f"min=({min_t1:.2f},{min_t2:.2f})")

    ax.set_xlabel("$\\theta_2$")
    ax.set_ylabel("$\\theta_1$")
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.suptitle("Exp4: Loss Landscape Comparison (d=64, log scale)")
plt.tight_layout()

heatmap_path = RESULTS_DIR / "exp4_heatmap.png"
fig.savefig(heatmap_path, dpi=150)
plt.close(fig)
print(f"Saved: {heatmap_path}")

# ============================================================
# 図 2: 3D サーフェス
# ============================================================

T1, T2 = np.meshgrid(theta1_vals, theta2_vals, indexing="ij")

fig = plt.figure(figsize=(14, 5))

for idx, (grid, title) in enumerate(
    [
        (J_grid,    "$J(\\theta)$ — without smoothing"),
        (Jeps_grid, "$J_\\varepsilon(\\theta)$ — with smoothing"),
    ]
):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    log_grid = np.log1p(grid)
    ax.plot_surface(T1, T2, log_grid, cmap="viridis", alpha=0.85, linewidth=0)
    ax.set_xlabel("$\\theta_1$")
    ax.set_ylabel("$\\theta_2$")
    ax.set_zlabel("log(1 + loss)")
    ax.set_title(title)

fig.suptitle("Exp4: Loss Landscape 3D Surface (d=64, log scale)")
plt.tight_layout()

surface_path = RESULTS_DIR / "exp4_surface.png"
fig.savefig(surface_path, dpi=150)
plt.close(fig)
print(f"Saved: {surface_path}")

# ============================================================
# サマリ統計
# ============================================================

print("\n=== Exp4 Summary ===")
print(f"  J      min={J_grid.min():.6f}  max={J_grid.max():.4f}  "
      f"at theta=({theta1_vals[np.unravel_index(np.argmin(J_grid), J_grid.shape)[0]]:.2f},"
      f"{theta2_vals[np.unravel_index(np.argmin(J_grid), J_grid.shape)[1]]:.2f})")
print(f"  J_eps  min={Jeps_grid.min():.6f}  max={Jeps_grid.max():.4f}  "
      f"at theta=({theta1_vals[np.unravel_index(np.argmin(Jeps_grid), Jeps_grid.shape)[0]]:.2f},"
      f"{theta2_vals[np.unravel_index(np.argmin(Jeps_grid), Jeps_grid.shape)[1]]:.2f})")
print(f"  J at theta*=(1,1)    = {J_grid[star_i, star_j]:.6f}")
print(f"  J_eps at theta*=(1,1)= {Jeps_grid[star_i, star_j]:.6f}")
