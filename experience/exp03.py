"""
exp03.py — Experiment 3: 状態次元への頑健性

目的:
  d ∈ {32, 64, 128} で Exp1 相当の最適化を実行し、
  次元が変わっても raw/smooth の相対的な挙動が維持されることを確認する。

設定:
  - state_dims = [32, 64, 128]
  - 各次元で Dynamics・介入・観測を再生成（dynamics_seed=0 固定）
  - n_init=20 初期値 × 2 条件 (raw / smooth)
  - Adam lr=1e-2, 500 ステップ

出力:
  results/exp03/exp3_runs.csv      : per-run 結果
  results/exp03/exp3_dims.csv      : per-dimension 集計
  results/exp03/exp3_conv_rate.png
  results/exp03/exp3_theta_dist.png
"""

import csv
import statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

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

RESULTS_DIR = Path(__file__).parent / "results" / "exp03"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

state_dims  = cfg["state_dims"]   # [32, 64, 128]
T           = cfg["T"]
n_init      = cfg["n_init"]
lr          = cfg["lr"]
n_steps     = cfg["n_steps"]
theta_range = cfg["theta_init_range"]

CONV_THETA_THR = 0.05
THETA_STAR = torch.tensor(cfg["theta_star"], device=device)

# ============================================================
# メインループ: 3 次元 × 20 inits × 2 conditions
# ============================================================

records_run  = []   # per-run 詳細
records_dims = []   # per-dimension 集計

for d in state_dims:
    # 次元ごとに Dynamics・観測・介入を再生成（seed 固定）
    torch.manual_seed(cfg["dynamics_seed"])
    dyn = Dynamics(d).to(device)
    obs_module = build_observer(d)
    observe    = obs_module

    p0_fixed, zs_target, E1, E2, a, M_base, schedule, kernel = build_target(
        dyn, observe, d=d
    )

    # 初期値サンプリング（次元ごとに独立、seed 固定）
    torch.manual_seed(0)
    theta_inits = (torch.rand(n_init, 2) * 2.0 - 1.0) * theta_range

    for condition in ("raw", "smooth"):
        conv_count   = 0
        final_dists  = []
        final_losses = []

        for init_id in range(n_init):
            theta = (
                theta_inits[init_id].clone().to(device).detach().requires_grad_(True)
            )
            optimizer = optim.Adam([theta], lr=lr)

            converged = False
            conv_step = n_steps
            loss_val  = None

            for step in range(n_steps):
                optimizer.zero_grad()

                if condition == "raw":
                    L = loss_raw(
                        theta, p0_fixed, zs_target, dyn, schedule, T, observe
                    )
                else:
                    L = loss_smooth(
                        theta, p0_fixed, zs_target, dyn, schedule, T, observe, kernel
                    )

                L.backward()
                optimizer.step()

                theta_dist = (theta.detach() - THETA_STAR).norm().item()
                loss_val   = L.item()

                if not converged and theta_dist < CONV_THETA_THR:
                    converged = True
                    conv_step = step

            final_dist = (theta.detach() - THETA_STAR).norm().item()
            if converged:
                conv_count += 1
            final_dists.append(final_dist)
            final_losses.append(loss_val)

            records_run.append(
                {
                    "state_dim":        d,
                    "init_id":          init_id,
                    "condition":        condition,
                    "final_loss":       loss_val,
                    "final_theta_dist": final_dist,
                    "converged":        int(converged),
                    "conv_step":        conv_step,
                }
            )

        mean_dist = statistics.mean(final_dists)
        std_dist  = statistics.stdev(final_dists) if len(final_dists) > 1 else 0.0

        records_dims.append(
            {
                "state_dim":       d,
                "condition":       condition,
                "conv_rate":       conv_count / n_init,
                "mean_final_loss": statistics.mean(final_losses),
                "mean_theta_dist": mean_dist,
                "std_theta_dist":  std_dist,
            }
        )

        print(
            f"[Exp3] d={d:3d} {condition:6s}: "
            f"conv={conv_count}/{n_init}  "
            f"mean_dist={mean_dist:.4f}+/-{std_dist:.4f}"
        )

# ============================================================
# CSV 保存
# ============================================================

runs_path = RESULTS_DIR / "exp3_runs.csv"
dims_path = RESULTS_DIR / "exp3_dims.csv"

with open(runs_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "state_dim", "init_id", "condition",
            "final_loss", "final_theta_dist", "converged", "conv_step",
        ],
    )
    writer.writeheader()
    writer.writerows(records_run)

with open(dims_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "state_dim", "condition", "conv_rate",
            "mean_final_loss", "mean_theta_dist", "std_theta_dist",
        ],
    )
    writer.writeheader()
    writer.writerows(records_dims)

print(f"\nSaved: {runs_path}")
print(f"Saved: {dims_path}")

# ============================================================
# 図 1: 収束率 — 次元別グループ棒グラフ
# ============================================================

raw_conv    = [r["conv_rate"] for r in records_dims if r["condition"] == "raw"]
smooth_conv = [r["conv_rate"] for r in records_dims if r["condition"] == "smooth"]

x     = np.arange(len(state_dims))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width / 2, raw_conv,    width, label="raw J",      color="tomato",    alpha=0.8)
ax.bar(x + width / 2, smooth_conv, width, label="smooth J_ε", color="steelblue", alpha=0.8)

ax.set_xlabel("State dimension d")
ax.set_ylabel("Convergence rate (||theta-theta*|| < 0.05)")
ax.set_title("Exp3: Convergence Rate across State Dimensions")
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in state_dims])
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

conv_fig_path = RESULTS_DIR / "exp3_conv_rate.png"
fig.savefig(conv_fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {conv_fig_path}")

# ============================================================
# 図 2: 最終 theta_dist の分布 — 次元別箱ひげ図
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, condition in zip(axes, ("raw", "smooth")):
    data_by_dim = [
        [
            r["final_theta_dist"]
            for r in records_run
            if r["state_dim"] == d and r["condition"] == condition
        ]
        for d in state_dims
    ]
    color      = "tomato" if condition == "raw" else "steelblue"
    label_title = "Without smoothing (J)" if condition == "raw" else "With smoothing (J_ε)"

    bp = ax.boxplot(
        data_by_dim,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(
        CONV_THETA_THR, color="gray", linestyle="--", linewidth=1.0,
        label=f"threshold={CONV_THETA_THR}"
    )
    ax.set_xlabel("State dimension d")
    ax.set_ylabel("||theta_final - theta*||")
    ax.set_title(label_title)
    ax.set_xticks(range(1, len(state_dims) + 1))
    ax.set_xticklabels([str(d) for d in state_dims])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Exp3: Final theta Distance across State Dimensions (20 inits each)")
plt.tight_layout()

dist_fig_path = RESULTS_DIR / "exp3_theta_dist.png"
fig.savefig(dist_fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {dist_fig_path}")

# ============================================================
# サマリ統計
# ============================================================

print("\n=== Exp3 Summary ===")
for condition in ("raw", "smooth"):
    print(f"  [{condition:6s}]")
    for r in records_dims:
        if r["condition"] == condition:
            print(
                f"    d={r['state_dim']:3d}: conv_rate={r['conv_rate']:.2f}"
                f"  theta_dist={r['mean_theta_dist']:.4f}+/-{r['std_theta_dist']:.4f}"
            )
