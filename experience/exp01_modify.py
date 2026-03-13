"""
exp01_modify.py — Experiment 1 (modified): 両側平滑化によるバイアス除去

変更点（exp01.py との差分）:
  - loss_smooth → loss_smooth_sym（ターゲットも同じ kernel で平滑化）
  - バイアスが除去されるため theta* = (1,1) で loss = 0 になる
  - 勾配ノルムの滑らかさを main evidence として強調

目的:
  バイアスを補正した上で「平滑化の本質的な利益 = 勾配場の滑らかさ」を示す。
  raw と smooth_sym の違いは loss 絶対値ではなく勾配ノルムの振る舞いに現れる。

出力:
  results/exp01_modify/exp1m_steps.csv
  results/exp01_modify/exp1m_summary.csv
  results/exp01_modify/exp1m_loss_curves.png
  results/exp01_modify/exp1m_theta_dist.png
  results/exp01_modify/exp1m_grad_norms.png
  results/exp01_modify/exp1m_grad_smoothness.png   ← 新規: 勾配ノルムの変化率
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
    loss_smooth_sym,
)

RESULTS_DIR = Path(__file__).parent / "results" / "exp01_modify"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# パラメータ（exp01.py と共通）
# ============================================================

d           = cfg["state_dim"]
T           = cfg["T"]
n_init      = cfg["n_init"]
lr          = cfg["lr"]
n_steps     = cfg["n_steps"]
theta_range = cfg["theta_init_range"]

CONV_THETA_THR = 0.05
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
# 初期値サンプリング（exp01.py と同一）
# ============================================================

torch.manual_seed(0)
theta_inits = (torch.rand(n_init, 2) * 2.0 - 1.0) * theta_range

# ============================================================
# 最適化ループ
# ============================================================

records_step    = []
records_summary = []

for init_id in range(n_init):
    for condition in ("raw", "smooth_sym"):
        theta = theta_inits[init_id].clone().to(device).detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=lr)

        converged = False
        conv_step = n_steps

        for step in range(n_steps):
            optimizer.zero_grad()

            if condition == "raw":
                L = loss_raw(theta, p0_fixed, zs_target, dyn, schedule, T, observe)
            else:
                L = loss_smooth_sym(
                    theta, p0_fixed, zs_target, dyn, schedule, T, observe, kernel
                )

            L.backward()
            grad_norm  = theta.grad.norm().item()
            optimizer.step()

            loss_val   = L.item()
            theta_dist = (theta.detach() - THETA_STAR).norm().item()

            records_step.append(
                {
                    "init_id":    init_id,
                    "condition":  condition,
                    "step":       step,
                    "loss":       loss_val,
                    "grad_norm":  grad_norm,
                    "theta_dist": theta_dist,
                }
            )

            if not converged and theta_dist < CONV_THETA_THR:
                converged = True
                conv_step = step

        final_loss  = records_step[-1]["loss"]
        final_dist  = records_step[-1]["theta_dist"]
        theta_final = theta.detach().cpu()

        records_summary.append(
            {
                "init_id":          init_id,
                "condition":        condition,
                "final_loss":       final_loss,
                "final_theta_dist": final_dist,
                "converged":        int(converged),
                "conv_step":        conv_step,
                "theta1_final":     theta_final[0].item(),
                "theta2_final":     theta_final[1].item(),
            }
        )

        status = f"conv @step={conv_step}" if converged else "not_converged"
        print(
            f"[Exp1m] init={init_id:2d} {condition:10s}: "
            f"loss={final_loss:.6f}  ||theta-theta*||={final_dist:.4f}  {status}"
        )

# ============================================================
# CSV 保存
# ============================================================

step_path    = RESULTS_DIR / "exp1m_steps.csv"
summary_path = RESULTS_DIR / "exp1m_summary.csv"

with open(step_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["init_id", "condition", "step", "loss", "grad_norm", "theta_dist"]
    )
    writer.writeheader()
    writer.writerows(records_step)

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "init_id", "condition", "final_loss", "final_theta_dist",
            "converged", "conv_step", "theta1_final", "theta2_final",
        ],
    )
    writer.writeheader()
    writer.writerows(records_summary)

print(f"\nSaved: {step_path}")
print(f"Saved: {summary_path}")

# ============================================================
# 共通ユーティリティ
# ============================================================

steps_arr  = np.arange(n_steps)
conditions = ("raw", "smooth_sym")
colors_ind  = {"raw": "tomato",    "smooth_sym": "steelblue"}
colors_mean = {"raw": "darkred",   "smooth_sym": "navy"}
titles      = {"raw": "Without smoothing (J)", "smooth_sym": "With smoothing sym (J_eps_sym)"}

def get_array(field, condition):
    return np.array([
        [r[field] for r in records_step if r["init_id"] == i and r["condition"] == condition]
        for i in range(n_init)
    ])

# ============================================================
# 図 1: 損失曲線
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, cond in zip(axes, conditions):
    arr  = get_array("loss", cond)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    for i in range(n_init):
        ax.plot(steps_arr, arr[i], alpha=0.15, color=colors_ind[cond], linewidth=0.8)
    ax.plot(steps_arr, mean, color=colors_mean[cond], linewidth=2.0, label="mean")
    ax.fill_between(steps_arr, mean - std, mean + std, alpha=0.25, color=colors_ind[cond])
    ax.set_title(titles[cond])
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle("Exp1 (modified): Loss Curves — symmetric smoothing (bias removed)")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "exp1m_loss_curves.png", dpi=150)
plt.close(fig)

# ============================================================
# 図 2: theta_dist 推移
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, cond in zip(axes, conditions):
    arr  = get_array("theta_dist", cond)
    mean = arr.mean(axis=0)
    for i in range(n_init):
        ax.plot(steps_arr, arr[i], alpha=0.15, color=colors_ind[cond], linewidth=0.8)
    ax.plot(steps_arr, mean, color=colors_mean[cond], linewidth=2.0, label="mean")
    ax.axhline(CONV_THETA_THR, color="gray", linestyle="--", linewidth=1.0,
               label=f"threshold={CONV_THETA_THR}")
    ax.set_title(titles[cond])
    ax.set_xlabel("Step"); ax.set_ylabel("||theta - theta*||"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle("Exp1 (modified): Distance to theta*")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "exp1m_theta_dist.png", dpi=150)
plt.close(fig)

# ============================================================
# 図 3: 勾配ノルム推移
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, cond in zip(axes, conditions):
    arr  = get_array("grad_norm", cond)
    mean = arr.mean(axis=0)
    for i in range(n_init):
        ax.plot(steps_arr, arr[i], alpha=0.15, color=colors_ind[cond], linewidth=0.8)
    ax.plot(steps_arr, mean, color=colors_mean[cond], linewidth=2.0, label="mean")
    ax.set_title(titles[cond])
    ax.set_xlabel("Step"); ax.set_ylabel("Gradient Norm"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle("Exp1 (modified): Gradient Norm Trajectories")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "exp1m_grad_norms.png", dpi=150)
plt.close(fig)

# ============================================================
# 図 4: 勾配ノルムの変化率 |Δgrad_norm| — 滑らかさの指標
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, cond in zip(axes, conditions):
    arr   = get_array("grad_norm", cond)
    delta = np.abs(np.diff(arr, axis=1))          # step 間の絶対変化量
    mean  = delta.mean(axis=0)
    std   = delta.std(axis=0)
    steps_d = steps_arr[1:]
    for i in range(n_init):
        ax.plot(steps_d, delta[i], alpha=0.12, color=colors_ind[cond], linewidth=0.8)
    ax.plot(steps_d, mean, color=colors_mean[cond], linewidth=2.0, label="mean |delta g|")
    ax.fill_between(steps_d, mean - std, mean + std, alpha=0.2, color=colors_ind[cond])
    ax.set_title(titles[cond])
    ax.set_xlabel("Step"); ax.set_ylabel("|grad_norm[t] - grad_norm[t-1]|")
    ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle("Exp1 (modified): Gradient Smoothness — |delta grad_norm| per step\n"
             "(smaller = smoother gradient field, key evidence for Frechet differentiability)")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "exp1m_grad_smoothness.png", dpi=150)
plt.close(fig)

print(f"Saved: figures to {RESULTS_DIR}")

# ============================================================
# サマリ統計
# ============================================================

print("\n=== Exp1 (modified) Summary ===")
for condition in conditions:
    cond_recs   = [r for r in records_summary if r["condition"] == condition]
    conv_count  = sum(r["converged"] for r in cond_recs)
    final_losses = [r["final_loss"] for r in cond_recs]
    final_dists  = [r["final_theta_dist"] for r in cond_recs]
    print(
        f"  [{condition:10s}] conv_rate={conv_count}/{n_init} ({conv_count/n_init:.0%})"
        f"  final_loss={statistics.mean(final_losses):.6f}+/-{statistics.stdev(final_losses):.6f}"
        f"  ||theta-theta*||={statistics.mean(final_dists):.4f}+/-{statistics.stdev(final_dists):.4f}"
    )
