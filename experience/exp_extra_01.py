"""
exp_extra_01.py — Extra Experiment 1: 準ニュートン法（L-BFGS）の導入

目的:
  smooth な landscape（J_ε_sym）では Hessian 近似が安定し、
  準ニュートン法（L-BFGS）が 1 次法（Adam）より効率的に収束することを確認する。

  Fréchet 微分可能な空間での 2 階情報の利用可能性 = 平滑化の追加利益。

条件（4 通りの組み合わせ）:
  raw       + Adam  : 従来ベースライン（exp01）
  smooth_sym + Adam : 1 次法 + 平滑化（exp01_modify）
  raw       + LBFGS : 準 2 次法、非平滑 landscape
  smooth_sym + LBFGS: 準 2 次法 + 平滑化 ← 理論上最も安定するはず

最適化ステップ数:
  Adam  : n_steps=500  （関数評価 500 回）
  L-BFGS: n_steps=100, max_iter=5  （関数評価 最大 500 回、公平な比較）

出力:
  results/exp_extra_01/extra1_steps.csv
  results/exp_extra_01/extra1_summary.csv
  results/exp_extra_01/extra1_loss_curves.png
  results/exp_extra_01/extra1_theta_dist.png
  results/exp_extra_01/extra1_grad_norms.png
  results/exp_extra_01/extra1_convergence_speed.png
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

RESULTS_DIR = Path(__file__).parent / "results" / "exp_extra_01"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

d           = cfg["state_dim"]
T           = cfg["T"]
n_init      = cfg["n_init"]
lr          = cfg["lr"]
theta_range = cfg["theta_init_range"]

N_STEPS_ADAM  = cfg["n_steps"]   # 500
N_STEPS_LBFGS = 100              # L-BFGS ステップ数（max_iter=5 で最大 500 評価）
LBFGS_MAX_ITER = 5               # 1 ステップ内の最大線探索回数

CONV_THETA_THR = 0.05
THETA_STAR = torch.tensor(cfg["theta_star"], device=device)

CONDITIONS = [
    ("raw",        "adam",  "raw + Adam"),
    ("smooth_sym", "adam",  "smooth_sym + Adam"),
    ("raw",        "lbfgs", "raw + L-BFGS"),
    ("smooth_sym", "lbfgs", "smooth_sym + L-BFGS"),
]

# ============================================================
# モデル・ターゲット軌跡の構築
# ============================================================

torch.manual_seed(cfg["dynamics_seed"])
dyn = Dynamics(d).to(device)
obs_module = build_observer(d)
observe    = obs_module

p0_fixed, zs_target, E1, E2, a, M_base, schedule, kernel = build_target(dyn, observe)

# ============================================================
# 初期値サンプリング
# ============================================================

torch.manual_seed(0)
theta_inits = (torch.rand(n_init, 2) * 2.0 - 1.0) * theta_range

# ============================================================
# 最適化ループ
# ============================================================

records_step    = []
records_summary = []

for loss_type, opt_type, label in CONDITIONS:
    n_steps = N_STEPS_ADAM if opt_type == "adam" else N_STEPS_LBFGS

    for init_id in range(n_init):
        theta = theta_inits[init_id].clone().to(device).detach().requires_grad_(True)

        if opt_type == "adam":
            optimizer = optim.Adam([theta], lr=lr)
        else:
            optimizer = optim.LBFGS(
                [theta],
                lr=1.0,
                max_iter=LBFGS_MAX_ITER,
                history_size=10,
                line_search_fn="strong_wolfe",
            )

        converged = False
        conv_step = n_steps

        for step in range(n_steps):

            if opt_type == "adam":
                optimizer.zero_grad()
                if loss_type == "raw":
                    L = loss_raw(theta, p0_fixed, zs_target, dyn, schedule, T, observe)
                else:
                    L = loss_smooth_sym(
                        theta, p0_fixed, zs_target, dyn, schedule, T, observe, kernel
                    )
                L.backward()
                grad_norm = theta.grad.norm().item()
                optimizer.step()
                loss_val = L.item()

            else:  # L-BFGS
                def closure():
                    optimizer.zero_grad()
                    if loss_type == "raw":
                        loss = loss_raw(
                            theta, p0_fixed, zs_target, dyn, schedule, T, observe
                        )
                    else:
                        loss = loss_smooth_sym(
                            theta, p0_fixed, zs_target, dyn, schedule, T, observe, kernel
                        )
                    loss.backward()
                    return loss

                L = optimizer.step(closure)
                loss_val  = L.item()
                grad_norm = theta.grad.norm().item() if theta.grad is not None else 0.0

            theta_dist = (theta.detach() - THETA_STAR).norm().item()

            records_step.append(
                {
                    "label":      label,
                    "init_id":    init_id,
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
                "label":            label,
                "init_id":          init_id,
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
            f"[Extra1] {label:25s} init={init_id:2d}: "
            f"loss={final_loss:.6f}  ||theta-theta*||={final_dist:.4f}  {status}"
        )

# ============================================================
# CSV 保存
# ============================================================

step_path    = RESULTS_DIR / "extra1_steps.csv"
summary_path = RESULTS_DIR / "extra1_summary.csv"

with open(step_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["label", "init_id", "step", "loss", "grad_norm", "theta_dist"]
    )
    writer.writeheader()
    writer.writerows(records_step)

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label", "init_id", "final_loss", "final_theta_dist",
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

LABEL_STYLE = {
    "raw + Adam":           {"color": "tomato",    "line": "-",  "marker": ""},
    "smooth_sym + Adam":    {"color": "steelblue", "line": "-",  "marker": ""},
    "raw + L-BFGS":         {"color": "salmon",    "line": "--", "marker": ""},
    "smooth_sym + L-BFGS":  {"color": "navy",      "line": "--", "marker": ""},
}

def get_steps_array(label):
    steps = sorted(set(r["step"] for r in records_step if r["label"] == label))
    return np.array(steps)

def get_field_matrix(label, field):
    steps = get_steps_array(label)
    return np.array([
        [r[field] for r in records_step if r["init_id"] == i and r["label"] == label]
        for i in range(n_init)
    ])

# ============================================================
# 図 1: 損失曲線（4 条件）
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
for label, style in LABEL_STYLE.items():
    arr   = get_field_matrix(label, "loss")
    steps = get_steps_array(label)
    mean  = arr.mean(axis=0)
    ax.plot(steps, mean, color=style["color"], linestyle=style["line"],
            linewidth=2.0, label=label)
    ax.fill_between(steps, arr.min(axis=0), arr.max(axis=0),
                    alpha=0.12, color=style["color"])

ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_yscale("log")
ax.set_title("Extra Exp1: Loss Curves — 4 conditions")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "extra1_loss_curves.png", dpi=150)
plt.close(fig)

# ============================================================
# 図 2: theta_dist 推移（4 条件）
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
for label, style in LABEL_STYLE.items():
    arr   = get_field_matrix(label, "theta_dist")
    steps = get_steps_array(label)
    mean  = arr.mean(axis=0)
    ax.plot(steps, mean, color=style["color"], linestyle=style["line"],
            linewidth=2.0, label=label)

ax.axhline(CONV_THETA_THR, color="gray", linestyle=":", linewidth=1.0,
           label=f"threshold={CONV_THETA_THR}")
ax.set_xlabel("Step"); ax.set_ylabel("||theta - theta*||"); ax.set_yscale("log")
ax.set_title("Extra Exp1: Distance to theta* — 4 conditions")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "extra1_theta_dist.png", dpi=150)
plt.close(fig)

# ============================================================
# 図 3: 勾配ノルム推移（4 条件）
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
for label, style in LABEL_STYLE.items():
    arr   = get_field_matrix(label, "grad_norm")
    steps = get_steps_array(label)
    mean  = arr.mean(axis=0)
    ax.plot(steps, mean, color=style["color"], linestyle=style["line"],
            linewidth=2.0, label=label)

ax.set_xlabel("Step"); ax.set_ylabel("Gradient Norm"); ax.set_yscale("log")
ax.set_title("Extra Exp1: Gradient Norm — 4 conditions")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "extra1_grad_norms.png", dpi=150)
plt.close(fig)

# ============================================================
# 図 4: 収束速度の比較（収束した run のみ、conv_step の分布）
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))
labels_order = [l for l, _, _ in CONDITIONS]
conv_steps_by_label = []
for label in [l for _, _, l in CONDITIONS]:
    cond_recs = [r for r in records_summary if r["label"] == label and r["converged"]]
    conv_steps_by_label.append([r["conv_step"] for r in cond_recs])

bp = ax.boxplot(
    conv_steps_by_label,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
)
colors_box = ["tomato", "steelblue", "salmon", "navy"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks(range(1, 5))
ax.set_xticklabels([l for _, _, l in CONDITIONS], rotation=12, ha="right")
ax.set_ylabel("Convergence step")
ax.set_title("Extra Exp1: Convergence Speed (converged runs only)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "extra1_convergence_speed.png", dpi=150)
plt.close(fig)

print(f"Saved: figures to {RESULTS_DIR}")

# ============================================================
# サマリ統計
# ============================================================

print("\n=== Extra Exp1 Summary ===")
for _, _, label in CONDITIONS:
    cond_recs   = [r for r in records_summary if r["label"] == label]
    conv_count  = sum(r["converged"] for r in cond_recs)
    conv_recs   = [r for r in cond_recs if r["converged"]]
    final_dists = [r["final_theta_dist"] for r in cond_recs]
    conv_steps  = [r["conv_step"] for r in conv_recs] if conv_recs else [float("nan")]
    print(
        f"  [{label:25s}] conv={conv_count}/{n_init}"
        f"  ||theta-theta*||={statistics.mean(final_dists):.4f}"
        f"  conv_step_median={sorted(conv_steps)[len(conv_steps)//2] if conv_recs else 'N/A'}"
    )
