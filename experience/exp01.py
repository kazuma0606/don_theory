"""
exp01.py — Experiment 1: 平滑化の有無による最適化の安定性比較

目的:
  - loss_raw  (平滑化なし) : 発散・不安定
  - loss_smooth (平滑化あり): 安定収束

評価指標:
  - 収束率 (loss < CONV_THRESHOLD に達した割合)
  - 最終損失の分散
  - 勾配ノルムの推移

出力:
  results/exp1_steps.csv   : (init_id, condition, step, loss, grad_norm)
  results/exp1_summary.csv : (init_id, condition, final_loss, converged, conv_step, theta1, theta2)
  results/exp1_loss_curves.png
  results/exp1_grad_norms.png
"""

import csv
import statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

matplotlib.use("Agg")

from common import (
    Dynamics,
    build_observer,
    build_target,
    cfg,
    device,
    loss_raw,
    loss_smooth,
)
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "exp01"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

d         = cfg["state_dim"]
T         = cfg["T"]
n_init    = cfg["n_init"]
lr        = cfg["lr"]
n_steps   = cfg["n_steps"]
theta_range = cfg["theta_init_range"]

# 収束判定: ||θ - θ*|| < CONV_THETA_THR
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
# 初期値サンプリング (20 通り固定シード)
# ============================================================

torch.manual_seed(0)
theta_inits = (torch.rand(n_init, 2) * 2.0 - 1.0) * theta_range  # Uniform(-2, 2)

# ============================================================
# 最適化ループ
# ============================================================

records_step    = []   # per-step レコード
records_summary = []   # per-run サマリ

for init_id in range(n_init):
    for condition in ("raw", "smooth"):
        theta = theta_inits[init_id].clone().to(device).detach().requires_grad_(True)
        optimizer = optim.Adam([theta], lr=lr)

        converged = False
        conv_step = n_steps  # 未収束の場合は n_steps

        for step in range(n_steps):
            optimizer.zero_grad()

            if condition == "raw":
                L = loss_raw(theta, p0_fixed, zs_target, dyn, schedule, T, observe)
            else:
                L = loss_smooth(
                    theta, p0_fixed, zs_target, dyn, schedule, T, observe, kernel
                )

            L.backward()
            grad_norm = theta.grad.norm().item()
            optimizer.step()

            loss_val = L.item()
            theta_dist = (theta.detach() - THETA_STAR).norm().item()

            records_step.append(
                {
                    "init_id": init_id,
                    "condition": condition,
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_norm,
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
                "init_id": init_id,
                "condition": condition,
                "final_loss": final_loss,
                "final_theta_dist": final_dist,
                "converged": int(converged),
                "conv_step": conv_step,
                "theta1_final": theta_final[0].item(),
                "theta2_final": theta_final[1].item(),
            }
        )

        status = f"conv @step={conv_step}" if converged else "not_converged"
        print(
            f"[Exp1] init={init_id:2d} {condition:6s}: "
            f"loss={final_loss:.4f}  ||theta-theta*||={final_dist:.4f}  {status}"
        )

# ============================================================
# CSV 保存
# ============================================================

step_path    = RESULTS_DIR / "exp1_steps.csv"
summary_path = RESULTS_DIR / "exp1_summary.csv"

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
# 図 1: 損失曲線
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
steps_arr = np.arange(n_steps)

for ax, condition in zip(axes, ("raw", "smooth")):
    cond_losses = np.array(
        [
            [r["loss"] for r in records_step if r["init_id"] == i and r["condition"] == condition]
            for i in range(n_init)
        ]
    )
    mean_loss = cond_losses.mean(axis=0)
    std_loss  = cond_losses.std(axis=0)
    color_ind = "tomato" if condition == "raw" else "steelblue"
    color_mean = "darkred" if condition == "raw" else "navy"
    label_title = "Without smoothing (J)" if condition == "raw" else "With smoothing (J_eps)"

    for i in range(n_init):
        ax.plot(steps_arr, cond_losses[i], alpha=0.15, color=color_ind, linewidth=0.8)
    ax.plot(steps_arr, mean_loss, color=color_mean, linewidth=2.0, label="mean")
    ax.fill_between(
        steps_arr, mean_loss - std_loss, mean_loss + std_loss, alpha=0.25, color=color_ind
    )
    ax.set_title(label_title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Exp1: Loss Curves (20 random inits, d=64)")
plt.tight_layout()
loss_fig_path = RESULTS_DIR / "exp1_loss_curves.png"
fig.savefig(loss_fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {loss_fig_path}")

# ============================================================
# 図 1b: theta_dist 推移（||theta - theta*||）
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, condition in zip(axes, ("raw", "smooth")):
    cond_dists = np.array(
        [
            [r["theta_dist"] for r in records_step if r["init_id"] == i and r["condition"] == condition]
            for i in range(n_init)
        ]
    )
    mean_dist  = cond_dists.mean(axis=0)
    color_ind  = "tomato" if condition == "raw" else "steelblue"
    color_mean = "darkred" if condition == "raw" else "navy"
    label_title = "Without smoothing" if condition == "raw" else "With smoothing"

    for i in range(n_init):
        ax.plot(steps_arr, cond_dists[i], alpha=0.15, color=color_ind, linewidth=0.8)
    ax.plot(steps_arr, mean_dist, color=color_mean, linewidth=2.0, label="mean")
    ax.axhline(CONV_THETA_THR, color="gray", linestyle="--", linewidth=1.0, label=f"threshold={CONV_THETA_THR}")
    ax.set_title(label_title)
    ax.set_xlabel("Step")
    ax.set_ylabel("||theta - theta*||")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Exp1: Distance to theta* (20 random inits, d=64)")
plt.tight_layout()
dist_fig_path = RESULTS_DIR / "exp1_theta_dist.png"
fig.savefig(dist_fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {dist_fig_path}")

# ============================================================
# 図 2: 勾配ノルム推移
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, condition in zip(axes, ("raw", "smooth")):
    cond_gnorms = np.array(
        [
            [r["grad_norm"] for r in records_step if r["init_id"] == i and r["condition"] == condition]
            for i in range(n_init)
        ]
    )
    mean_gn   = cond_gnorms.mean(axis=0)
    color_ind  = "tomato" if condition == "raw" else "steelblue"
    color_mean = "darkred" if condition == "raw" else "navy"
    label_title = "Without smoothing" if condition == "raw" else "With smoothing"

    for i in range(n_init):
        ax.plot(steps_arr, cond_gnorms[i], alpha=0.15, color=color_ind, linewidth=0.8)
    ax.plot(steps_arr, mean_gn, color=color_mean, linewidth=2.0, label="mean")
    ax.set_title(label_title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Exp1: Gradient Norm Trajectories (20 random inits, d=64)")
plt.tight_layout()
grad_fig_path = RESULTS_DIR / "exp1_grad_norms.png"
fig.savefig(grad_fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {grad_fig_path}")

# ============================================================
# サマリ統計表示
# ============================================================

print("\n=== Exp1 Summary ===")
for condition in ("raw", "smooth"):
    cond_recs    = [r for r in records_summary if r["condition"] == condition]
    conv_count   = sum(r["converged"] for r in cond_recs)
    final_losses = [r["final_loss"] for r in cond_recs]
    final_dists  = [r["final_theta_dist"] for r in cond_recs]
    mean_fl  = statistics.mean(final_losses)
    stdev_fl = statistics.stdev(final_losses) if len(final_losses) > 1 else 0.0
    mean_td  = statistics.mean(final_dists)
    stdev_td = statistics.stdev(final_dists) if len(final_dists) > 1 else 0.0
    print(
        f"  [{condition:6s}] conv_rate={conv_count}/{n_init} ({conv_count/n_init:.0%})"
        f"  final_loss={mean_fl:.4f}+/-{stdev_fl:.4f}"
        f"  ||theta-theta*||={mean_td:.4f}+/-{stdev_td:.4f}"
    )
