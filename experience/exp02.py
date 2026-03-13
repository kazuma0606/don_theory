"""
exp02.py — Experiment 2: ダイナミクスのランダム性への頑健性

目的:
  Dynamics を 10 通りランダム生成し、各 Dynamics で Exp1 相当の最適化を実行。
  平滑化の効果が特定の Dynamics に依存しないことを確認する。

設定:
  - n_dynamics = 10 (seed 0..9)
  - 各 Dynamics で n_init=20 初期値 × 2 条件 (raw / smooth)
  - Adam lr=1e-2, 500 ステップ

出力:
  results/exp02/exp2_runs.csv     : per-run 結果
  results/exp02/exp2_dynamics.csv : per-dynamics 集計
  results/exp02/exp2_conv_rate.png
  results/exp02/exp2_theta_dist.png
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

RESULTS_DIR = Path(__file__).parent / "results" / "exp02"
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
n_dynamics  = cfg["n_dynamics"]

CONV_THETA_THR = 0.05
THETA_STAR = torch.tensor(cfg["theta_star"], device=device)

# 初期値は Exp1 と共通（固定シード）
torch.manual_seed(0)
theta_inits = (torch.rand(n_init, 2) * 2.0 - 1.0) * theta_range

# ============================================================
# メインループ: 10 Dynamics × 20 inits × 2 conditions
# ============================================================

records_run = []   # per-run 詳細
records_dyn = []   # per-dynamics 集計

for dyn_seed in range(n_dynamics):
    torch.manual_seed(dyn_seed)
    dyn = Dynamics(d).to(device)
    obs_module = build_observer(d)
    observe    = obs_module

    p0_fixed, zs_target, E1, E2, a, M_base, schedule, kernel = build_target(
        dyn, observe
    )

    for condition in ("raw", "smooth"):
        conv_count  = 0
        final_dists = []
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
                    "dyn_seed":        dyn_seed,
                    "init_id":         init_id,
                    "condition":       condition,
                    "final_loss":      loss_val,
                    "final_theta_dist": final_dist,
                    "converged":       int(converged),
                    "conv_step":       conv_step,
                }
            )

        mean_dist = statistics.mean(final_dists)
        std_dist  = statistics.stdev(final_dists) if len(final_dists) > 1 else 0.0
        mean_loss = statistics.mean(final_losses)

        records_dyn.append(
            {
                "dyn_seed":       dyn_seed,
                "condition":      condition,
                "conv_rate":      conv_count / n_init,
                "mean_final_loss": mean_loss,
                "mean_theta_dist": mean_dist,
                "std_theta_dist":  std_dist,
            }
        )

        print(
            f"[Exp2] dyn={dyn_seed} {condition:6s}: "
            f"conv={conv_count}/{n_init}  "
            f"mean_dist={mean_dist:.4f}+/-{std_dist:.4f}"
        )

# ============================================================
# CSV 保存
# ============================================================

runs_path = RESULTS_DIR / "exp2_runs.csv"
dyn_path  = RESULTS_DIR / "exp2_dynamics.csv"

with open(runs_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dyn_seed", "init_id", "condition",
            "final_loss", "final_theta_dist", "converged", "conv_step",
        ],
    )
    writer.writeheader()
    writer.writerows(records_run)

with open(dyn_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dyn_seed", "condition", "conv_rate",
            "mean_final_loss", "mean_theta_dist", "std_theta_dist",
        ],
    )
    writer.writeheader()
    writer.writerows(records_dyn)

print(f"\nSaved: {runs_path}")
print(f"Saved: {dyn_path}")

# ============================================================
# 図 1: 収束率 (conv_rate) — 10 Dynamics 横並び棒グラフ
# ============================================================

seeds      = list(range(n_dynamics))
raw_conv   = [r["conv_rate"] for r in records_dyn if r["condition"] == "raw"]
smooth_conv = [r["conv_rate"] for r in records_dyn if r["condition"] == "smooth"]

x     = np.arange(n_dynamics)
width = 0.35

fig, ax = plt.subplots(figsize=(10, 4))
bars_r = ax.bar(x - width / 2, raw_conv,    width, label="raw J",     color="tomato",    alpha=0.8)
bars_s = ax.bar(x + width / 2, smooth_conv, width, label="smooth J_ε", color="steelblue", alpha=0.8)

ax.set_xlabel("Dynamics seed")
ax.set_ylabel("Convergence rate (||theta-theta*|| < 0.05)")
ax.set_title("Exp2: Convergence Rate across 10 Random Dynamics")
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seeds])
ax.set_ylim(0, 1.1)
ax.axhline(
    np.mean(raw_conv), color="darkred",  linestyle="--", linewidth=1.2,
    label=f"raw mean={np.mean(raw_conv):.2f}"
)
ax.axhline(
    np.mean(smooth_conv), color="navy", linestyle="--", linewidth=1.2,
    label=f"smooth mean={np.mean(smooth_conv):.2f}"
)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

conv_fig_path = RESULTS_DIR / "exp2_conv_rate.png"
fig.savefig(conv_fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {conv_fig_path}")

# ============================================================
# 図 2: 最終 θ-distance の分布 — 箱ひげ図
# ============================================================

raw_dists_by_dyn = [
    [r["final_theta_dist"] for r in records_run if r["dyn_seed"] == s and r["condition"] == "raw"]
    for s in seeds
]
smooth_dists_by_dyn = [
    [r["final_theta_dist"] for r in records_run if r["dyn_seed"] == s and r["condition"] == "smooth"]
    for s in seeds
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data_by_dyn, title, color in zip(
    axes,
    [raw_dists_by_dyn, smooth_dists_by_dyn],
    ["Without smoothing (J)", "With smoothing (J_ε)"],
    ["tomato", "steelblue"],
):
    bp = ax.boxplot(
        data_by_dyn,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(CONV_THETA_THR, color="gray", linestyle="--", linewidth=1.0,
               label=f"threshold={CONV_THETA_THR}")
    ax.set_xlabel("Dynamics seed")
    ax.set_ylabel("||theta_final - theta*||")
    ax.set_title(title)
    ax.set_xticks(range(1, n_dynamics + 1))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Exp2: Final theta Distance across 10 Random Dynamics (20 inits each)")
plt.tight_layout()

dist_fig_path = RESULTS_DIR / "exp2_theta_dist.png"
fig.savefig(dist_fig_path, dpi=150)
plt.close(fig)
print(f"Saved: {dist_fig_path}")

# ============================================================
# サマリ統計
# ============================================================

print("\n=== Exp2 Summary ===")
for condition in ("raw", "smooth"):
    cond_recs  = [r for r in records_dyn if r["condition"] == condition]
    conv_rates = [r["conv_rate"] for r in cond_recs]
    dists      = [r["mean_theta_dist"] for r in cond_recs]
    print(
        f"  [{condition:6s}]"
        f"  conv_rate: mean={np.mean(conv_rates):.2f} std={np.std(conv_rates):.2f}"
        f"  theta_dist: mean={np.mean(dists):.4f} std={np.std(dists):.4f}"
    )
