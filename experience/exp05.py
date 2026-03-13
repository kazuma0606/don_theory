"""
exp05.py — Experiment 5: ε スケーリング検証（Theorem 1 & Corollary 2）

目的:
  測定 1 (Theorem 1):
    θ* で rollout した軌跡に ε 依存カーネルを適用し、
    ‖smooth_ε(z(θ*)) - z(θ*)‖ が O(ε) でスケールすることを確認。
    → log-log プロットの傾き ≈ 1.0

  測定 2 (Corollary 2):
    各 ε で J_ε を最適化（n_init=20 初期値、最良解を採用）し、
    ‖argmin J_ε - θ*‖ が ε→0 で単調減少することを確認。

設定:
  - eps_list = [1.0, 0.5, 0.2, 0.1, 0.05]
  - d=64, T=20, dynamics_seed=0
  - n_init=20, Adam lr=1e-2, 500 ステップ

出力:
  results/exp05/exp5_theorem1.csv   : ε ごとの軌跡近似誤差
  results/exp05/exp5_corollary2.csv : ε ごとの最小解誤差
  results/exp05/exp5_scaling.png    : 2 パネル図（log-log）
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
    loss_smooth,
    make_kernel,
    rollout,
    smooth_time,
)

RESULTS_DIR = Path(__file__).parent / "results" / "exp05"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

d           = cfg["state_dim"]
T           = cfg["T"]
n_init      = cfg["n_init"]
lr          = cfg["lr"]
n_steps     = cfg["n_steps"]
theta_range = cfg["theta_init_range"]
eps_list    = cfg["eps_list"]   # [1.0, 0.5, 0.2, 0.1, 0.05]

THETA_STAR = torch.tensor(cfg["theta_star"], device=device)

# ============================================================
# モデル・ターゲット軌跡の構築
# ============================================================

torch.manual_seed(cfg["dynamics_seed"])
dyn = Dynamics(d).to(device)
obs_module = build_observer(d)
observe    = obs_module

p0_fixed, zs_target, E1, E2, a, M_base, schedule, kernel_default = build_target(
    dyn, observe
)

# θ* での基準軌跡（非平滑）
with torch.no_grad():
    z_true = rollout(p0_fixed, THETA_STAR, dyn, schedule, T, observe)

# ============================================================
# 測定 1: 軌跡近似誤差（Theorem 1）
# ============================================================

print("=== Measurement 1: Trajectory Approximation Error (Theorem 1) ===")

theorem1_records = []

with torch.no_grad():
    for eps in eps_list:
        kernel   = make_kernel(eps).to(device)
        z_smooth = smooth_time(z_true, kernel)
        err      = (z_smooth - z_true).abs().mean().item()   # MAE (次元不変)
        theorem1_records.append({"eps": eps, "approx_err": err, "kernel_len": kernel.numel()})
        print(f"  eps={eps:.2f}  kernel_len={kernel.numel()}  MAE={err:.6f}")

# log-log 傾きを線形回帰で推定
log_eps = np.log(eps_list)
log_err = np.log([r["approx_err"] for r in theorem1_records])
slope, intercept = np.polyfit(log_eps, log_err, 1)
print(f"\n  log-log slope = {slope:.4f}  (expected ~= 1.0 for O(epsilon))")

# ============================================================
# 測定 2: 最小解の収束（Corollary 2）
# ============================================================

print("\n=== Measurement 2: Minimizer Convergence (Corollary 2) ===")

# 各 ε で n_init 初期値から最適化 → 最良解（最低 J_ε）を採用
torch.manual_seed(0)
theta_inits = (torch.rand(n_init, 2) * 2.0 - 1.0) * theta_range

corollary2_records = []
runs_records       = []   # 全 run の詳細

for eps in eps_list:
    kernel = make_kernel(eps).to(device)

    best_loss  = float("inf")
    best_theta = None

    for init_id in range(n_init):
        theta = (
            theta_inits[init_id].clone().to(device).detach().requires_grad_(True)
        )
        optimizer = optim.Adam([theta], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            L = loss_smooth(
                theta, p0_fixed, zs_target, dyn, schedule, T, observe, kernel
            )
            L.backward()
            optimizer.step()

        final_loss  = L.item()
        final_theta = theta.detach().clone()
        final_dist  = (final_theta - THETA_STAR).norm().item()

        runs_records.append(
            {
                "eps":              eps,
                "init_id":          init_id,
                "final_loss":       final_loss,
                "final_theta_dist": final_dist,
                "theta1":           final_theta[0].item(),
                "theta2":           final_theta[1].item(),
            }
        )

        if final_loss < best_loss:
            best_loss  = final_loss
            best_theta = final_theta

    best_dist = (best_theta - THETA_STAR).norm().item()
    corollary2_records.append(
        {
            "eps":          eps,
            "best_loss":    best_loss,
            "best_theta1":  best_theta[0].item(),
            "best_theta2":  best_theta[1].item(),
            "theta_dist":   best_dist,
        }
    )
    print(
        f"  eps={eps:.2f}  best_loss={best_loss:.6f}"
        f"  theta_eps=({best_theta[0]:.4f},{best_theta[1]:.4f})"
        f"  ||theta_eps-theta*||={best_dist:.4f}"
    )

# ============================================================
# CSV 保存
# ============================================================

t1_path  = RESULTS_DIR / "exp5_theorem1.csv"
cor2_path = RESULTS_DIR / "exp5_corollary2.csv"
runs_path = RESULTS_DIR / "exp5_runs.csv"

with open(t1_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["eps", "kernel_len", "approx_err"])
    writer.writeheader()
    writer.writerows(theorem1_records)

with open(cor2_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["eps", "best_loss", "best_theta1", "best_theta2", "theta_dist"]
    )
    writer.writeheader()
    writer.writerows(corollary2_records)

with open(runs_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["eps", "init_id", "final_loss", "final_theta_dist", "theta1", "theta2"]
    )
    writer.writeheader()
    writer.writerows(runs_records)

print(f"\nSaved: {t1_path}")
print(f"Saved: {cor2_path}")
print(f"Saved: {runs_path}")

# ============================================================
# 図: 2 パネル log-log スケーリング
# ============================================================

eps_arr  = np.array(eps_list)
err_arr  = np.array([r["approx_err"] for r in theorem1_records])
dist_arr = np.array([r["theta_dist"] for r in corollary2_records])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- 左パネル: Theorem 1 ---
ax = axes[0]
ax.loglog(eps_arr, err_arr, "o-", color="steelblue", linewidth=2, markersize=7,
          label="approx error")

# 傾き ≈ 1 の参照線
ref_line = np.exp(intercept) * eps_arr ** slope
ax.loglog(eps_arr, ref_line, "--", color="gray", linewidth=1.2,
          label=f"fit: slope={slope:.2f}")

# 傾き = 1.0 の理想線（O(ε)）
ideal = err_arr[0] / eps_arr[0] * eps_arr
ax.loglog(eps_arr, ideal, ":", color="black", linewidth=1.0, label="slope=1.0 (O(eps))")

ax.set_xlabel("epsilon")
ax.set_ylabel("||smooth_eps(z(theta*)) - z(theta*)||")
ax.set_title("Theorem 1: Trajectory Approximation Error")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# --- 右パネル: Corollary 2 ---
ax = axes[1]

# 各 eps の全 run の分布（散布）
for eps in eps_list:
    run_dists = [r["final_theta_dist"] for r in runs_records if r["eps"] == eps]
    ax.scatter(
        [eps] * len(run_dists), run_dists,
        alpha=0.3, color="tomato", s=20, zorder=2
    )

# 最良解の dist
ax.loglog(eps_arr, dist_arr, "o-", color="darkred", linewidth=2, markersize=8,
          label="best ||theta_eps - theta*||", zorder=5)

# 傾き = 1.0 の参照線
ideal_c2 = dist_arr[0] / eps_arr[0] * eps_arr
ax.loglog(eps_arr, ideal_c2, ":", color="black", linewidth=1.0, label="slope=1.0")

ax.set_xlabel("epsilon")
ax.set_ylabel("||argmin J_eps - theta*||")
ax.set_title("Corollary 2: Minimizer Convergence")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Exp5: epsilon-Scaling Validation (Theorem 1 & Corollary 2)")
plt.tight_layout()

fig_path = RESULTS_DIR / "exp5_scaling.png"
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {fig_path}")

# ============================================================
# サマリ
# ============================================================

print("\n=== Exp5 Summary ===")
print(f"  Theorem 1  log-log slope = {slope:.4f}  (expected ~= 1.0)")
print("  Corollary 2 minimizer distances:")
for r in corollary2_records:
    print(f"    eps={r['eps']:.2f}  ||theta_eps-theta*||={r['theta_dist']:.4f}")
