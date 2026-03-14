#!/usr/bin/env sage
# exp_W7e.sage — W-7e: B₆ (6 strands, 720 permutations) × Intervention Landscape
# スケーリング則: B₃(n=6,r=-0.83)→B₄(n=24,r=-0.72)→B₅(n=120,r=-0.57)→B₆(n=720,r=?)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms

print("=" * 65)
print("  W-7e: B₆ × Intervention Landscape (720 permutations)")
print("  スケーリング則の延長: r はさらに低下するか？")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7e")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Part 1: B₆ の全 720 置換ブレイドと Burau trace
# ══════════════════════════════════════════════════════════════════
print("\n[Part 1] B₆ の全 720 置換ブレイドと Burau trace")
print("-" * 55)

B6   = BraidGroup(6)
gens = list(B6.generators())   # σ₁, σ₂, σ₃, σ₄, σ₅
t    = var('t')

all_perms = list(iter_perms([1, 2, 3, 4, 5, 6]))   # 720 通り

print("  Burau trace を計算中 (720 件)...")
perm_info = []
for k, perm in enumerate(all_perms):
    sp    = SagePerm(list(perm))
    word  = sp.reduced_word()
    b     = B6.one()
    for i in word:
        b = b * gens[i - 1]
    bmat   = b.burau_matrix()(t=QQ(1)/QQ(2))
    btr    = float(bmat.trace())
    writhe = len(word)
    perm_info.append({
        "perm":   list(perm),
        "label":  "".join(str(x) for x in perm),
        "word":   word,
        "burau":  btr,
        "writhe": writhe,
        "order":  [x - 1 for x in perm],
    })
    if (k + 1) % 100 == 0:
        print(f"  ... {k+1}/720 完了")

burau_all  = np.array([p["burau"]  for p in perm_info])
writhe_all = np.array([p["writhe"] for p in perm_info])
print(f"  完了。Burau trace: [{burau_all.min():.4f}, {burau_all.max():.4f}]")
print(f"  Writhe:           [{int(writhe_all.min())}, {int(writhe_all.max())}]")

# ══════════════════════════════════════════════════════════════════
# Part 2: 損失景観 J(θ) の計算（全 720 順序）
# ══════════════════════════════════════════════════════════════════
print("\n[Part 2] 損失景観 J(θ) の計算")
print("-" * 55)

rng = np.random.default_rng(int(42))
d   = 24

A_dyn  = rng.standard_normal((d, d)) * 0.25
A_dyn  = A_dyn - A_dyn.T
b_bias = rng.standard_normal(d) * 0.1

def time_evolution(p):
    return np.tanh(A_dyn @ p + b_bias) * 0.5 + p * 0.4

a_vec  = np.zeros(d);  a_vec[:d//4] = 1.0
M_mat  = rng.standard_normal((d, d)) * 0.3
M_mat  = M_mat / (np.linalg.norm(M_mat, 2) + 1e-8)
bias4  = rng.standard_normal(d) * 0.3
bias5  = rng.standard_normal(d) * 0.2
bias6  = rng.standard_normal(d) * 0.25   # E6 専用

def E1(p, th): return p + th * a_vec
def E2(p, th): return (np.eye(d) + th * M_mat) @ p
def E3(p):     return np.tanh(p * 1.2) * 0.9
def E4(p):     return np.tanh(p * 0.8 + bias4) * 0.7
def E5(p):     return p * 0.6 + np.tanh(bias5) * 0.4
def E6(p):     return np.tanh(p + bias6) * 0.85  # 新規

N_ROUNDS = 2          # 各ラウンド = 6 介入
T_total  = N_ROUNDS * 6   # 12 ステップ

def apply_order(order, theta, p0):
    p    = p0.copy()
    traj = [p.copy()]
    for _ in range(N_ROUNDS):
        for idx in order:
            p = time_evolution(p)
            if   idx == 0: p = E1(p, theta[0])
            elif idx == 1: p = E2(p, theta[1])
            elif idx == 2: p = E3(p)
            elif idx == 3: p = E4(p)
            elif idx == 4: p = E5(p)
            else:          p = E6(p)
            traj.append(p.copy())
    return np.array(traj)

def compute_J(order, theta, p0, z_target):
    traj = apply_order(order, theta, p0)
    return float(np.mean(np.sum((traj - z_target)**2, axis=1)))

theta_star = np.array([1.0, 1.0])
p0         = rng.standard_normal(d) * 0.5
ref_order  = [0, 1, 2, 3, 4, 5]
z_target   = apply_order(ref_order, theta_star, p0)

N_GRID  = 25
th1_arr = np.linspace(-1.0, 3.0, N_GRID)
th2_arr = np.linspace(-1.0, 3.0, N_GRID)
TH1, TH2 = np.meshgrid(th1_arr, th2_arr)

# J をフラット配列で持つ（メモリ節約）
all_J_flat = np.zeros((720, N_GRID * N_GRID))
ref_land   = None

print(f"  グリッド: {N_GRID}×{N_GRID},  d={d},  T={T_total},  n_perms=720")
print(f"  計算中 (720 × {N_GRID}×{N_GRID} = {720*N_GRID*N_GRID} 点)...")

for k, info in enumerate(perm_info):
    order  = info["order"]
    J_grid = np.array([
        [compute_J(order, [th1_arr[j], th2_arr[i]], p0, z_target)
         for j in range(N_GRID)]
        for i in range(N_GRID)
    ])
    all_J_flat[k] = J_grid.ravel()
    j_min  = float(J_grid.min())
    j_star = compute_J(order, theta_star, p0, z_target)
    if ref_land is None:
        ref_land = J_grid.ravel();  dist = 0.0
    else:
        dist = float(np.sqrt(np.mean((J_grid.ravel() - ref_land)**2)))
    info["j_min"]  = j_min
    info["j_star"] = j_star
    info["dist"]   = dist
    if (k + 1) % 100 == 0:
        print(f"  ... {k+1}/720 完了")

dist_all  = np.array([info["dist"]  for info in perm_info])
print(f"  完了。景観距離: [{dist_all.min():.4f}, {dist_all.max():.4f}]")

# ══════════════════════════════════════════════════════════════════
# Part 3: 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[Part 3] 可視化")
print("-" * 55)

labels      = [info["label"]  for info in perm_info]
burau_vals  = np.array([info["burau"]  for info in perm_info])
dist_vals   = np.array([info["dist"]   for info in perm_info])
writhe_vals = np.array([info["writhe"] for info in perm_info])

r_b, p_b    = stats.pearsonr(burau_vals, dist_vals)
r_w, p_w    = stats.pearsonr(writhe_vals, dist_vals)
slope_b, intercept_b, *_ = stats.linregress(burau_vals, dist_vals)

# 図1: 代表景観 — Burau 上位 6 + 下位 6
sorted_asc = sorted(perm_info, key=lambda x: x["burau"])
rep12      = sorted_asc[-6:] + sorted_asc[:6]

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle("W-7e: B6 Loss Landscapes\n"
             "Top 6 Burau (simple) + Bottom 6 Burau (complex)", fontsize=12)
for ax, info in zip(axes.flat, rep12):
    idx = labels.index(info["label"])
    J   = all_J_flat[idx].reshape(N_GRID, N_GRID)
    im  = ax.contourf(TH1, TH2, J, levels=18, cmap="viridis")
    ax.contour(TH1, TH2, J, levels=6, colors="white", alpha=0.3, linewidths=0.5)
    ax.plot(*theta_star, "r*", ms=9)
    ax.set_title(f"{info['label']}  Burau={info['burau']:.3f}\n"
                 f"dist={info['dist']:.3f}", fontsize=8)
    ax.set_xlabel("theta1", fontsize=7);  ax.set_ylabel("theta2", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p = OUT / "W7e_fig1_landscapes.png"
fig.savefig(str(p), dpi=120, bbox_inches="tight");  plt.close()
print(f"  fig1 saved: {p}")

# 図2: メイン相関図（n=720）
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"W-7e: B6 Braid Invariant vs Loss Landscape  (n=720)", fontsize=12)

ax = axes[0]
ax.scatter(burau_vals, dist_vals, c=writhe_vals, cmap="plasma",
           s=8, alpha=0.5, zorder=5)
x_line = np.linspace(burau_vals.min(), burau_vals.max(), 100)
ax.plot(x_line, slope_b * x_line + intercept_b, "r-", alpha=0.9, lw=2,
        label=f"r={r_b:.4f}")
plt.colorbar(
    plt.cm.ScalarMappable(cmap="plasma",
        norm=plt.Normalize(writhe_vals.min(), writhe_vals.max())),
    ax=ax, label="Writhe")
ax.set_xlabel("Burau trace (t=1/2)")
ax.set_ylabel("Landscape distance from identity (RMSE)")
ax.set_title(f"Burau trace:  r = {r_b:.4f},  p = {p_b:.2e}\n"
             f"B3:-0.83  B4:-0.72  B5:-0.57  B6:{r_b:.4f}", fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

# 距離分布を Burau trace 四分位で色分けヒストグラム
ax2 = axes[1]
q25, q75 = np.percentile(burau_vals, [25, 75])
mask_hi = burau_vals >= q75      # 上位 25%（単純な braid）
mask_lo = burau_vals <= q25      # 下位 25%（複雑な braid）
mask_mid = ~mask_hi & ~mask_lo

bins = np.linspace(dist_vals.min(), dist_vals.max(), 30)
ax2.hist(dist_vals[mask_hi],  bins=bins, alpha=0.6, color="steelblue",
         label=f"Burau top 25% (simple, n={mask_hi.sum()})")
ax2.hist(dist_vals[mask_mid], bins=bins, alpha=0.4, color="gray",
         label=f"Burau mid 50% (n={mask_mid.sum()})")
ax2.hist(dist_vals[mask_lo],  bins=bins, alpha=0.6, color="tomato",
         label=f"Burau bot 25% (complex, n={mask_lo.sum()})")
ax2.set_xlabel("Landscape distance from identity")
ax2.set_ylabel("Count")
ax2.set_title("Distance distribution by Burau quartile", fontsize=10)
ax2.legend(fontsize=8);  ax2.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7e_fig2_correlation.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig2 saved: {p}")

# 図3: スケーリング則（B₃→B₄→B₅→B₆）
SCALING = [
    ("B3", 3,   6, -0.8287, -0.8287),
    ("B4", 4,  24, -0.7226, -0.7226),
    ("B5", 5, 120, -0.5651, -0.5651),
    ("B6", 6, 720,  r_b,     r_b),
]
ns  = [s[2] for s in SCALING]
rs  = [s[3] for s in SCALING]
Ks  = [s[1] for s in SCALING]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("W-7: Scaling Law  B3 → B4 → B5 → B6", fontsize=12)

ax = axes[0]
ax.plot(ns, [abs(r) for r in rs], "o-", ms=10, lw=2, color="steelblue")
for n_val, r_val, (name, K, *_) in zip(ns, rs, SCALING):
    ax.annotate(f"{name}\nr={r_val:.4f}",
                (n_val, abs(r_val)), textcoords="offset points",
                xytext=(8, -15), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("n (permutations = K!)", fontsize=11)
ax.set_ylabel("|Pearson r|", fontsize=11)
ax.set_title("Correlation magnitude vs n\n(log scale)", fontsize=10)
ax.grid(True, alpha=0.3);  ax.set_ylim(0, 1)

ax2 = axes[1]
ax2.plot(Ks, [abs(r) for r in rs], "s-", ms=10, lw=2, color="tomato")
for K_val, r_val, (name, *_) in zip(Ks, rs, SCALING):
    ax2.annotate(f"{name}\nr={r_val:.4f}",
                 (K_val, abs(r_val)), textcoords="offset points",
                 xytext=(5, 5), fontsize=9)
ax2.set_xlabel("K (number of interventions = strands)", fontsize=11)
ax2.set_ylabel("|Pearson r|", fontsize=11)
ax2.set_title("|r| vs K (number of strands)", fontsize=10)
ax2.grid(True, alpha=0.3);  ax2.set_ylim(0, 1)

plt.tight_layout()
p = OUT / "W7e_fig3_scaling.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig3 saved: {p}")

# 図4: 先頭介入ごとの距離分布（E1 が先頭か？）
fig, ax = plt.subplots(figsize=(10, 5))
first_elem = [info["order"][0] for info in perm_info]  # 0-indexed
colors6 = plt.cm.tab10(np.linspace(0, 0.6, 6))
for e in range(6):
    mask  = np.array(first_elem) == e
    dvals = dist_vals[mask]
    ax.scatter(np.full(mask.sum(), e), dvals, alpha=0.3, s=15,
               color=colors6[e], zorder=3)
    ax.plot([e - 0.3, e + 0.3], [dvals.mean(), dvals.mean()],
            lw=3, color=colors6[e], zorder=5,
            label=f"E{e+1} first: mean={dvals.mean():.2f}")

ax.set_xticks(range(6))
ax.set_xticklabels([f"E{i+1} first" for i in range(6)])
ax.set_ylabel("Landscape distance from identity")
ax.set_title("B6: Distance distribution by first intervention\n"
             "(horizontal lines = means)")
ax.legend(fontsize=8, loc="upper right");  ax.grid(True, alpha=0.3)
plt.tight_layout()
p = OUT / "W7e_fig4_first_elem.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig4 saved: {p}")

# ══════════════════════════════════════════════════════════════════
# Part 4: 統計サマリー
# ══════════════════════════════════════════════════════════════════
print("\n[Part 4] 統計サマリー")
print("-" * 65)

max_dist = dist_all.max()
max_idx  = int(dist_all.argmax())
nearest  = sorted(perm_info[1:], key=lambda x: x["dist"])[0]

# 先頭介入ごとの平均距離
first_means = []
for e in range(6):
    mask = np.array([info["order"][0] for info in perm_info]) == e
    first_means.append((e + 1, dist_vals[mask].mean(), mask.sum()))

print(f"""
  ━━ Burau trace vs 景観距離 (n=720) ━━
  Pearson r  = {r_b:.4f}
  p-value    = {p_b:.2e}

  ━━ Writhe vs 景観距離 ━━
  Pearson r  = {r_w:.4f}
  p-value    = {p_w:.2e}

  ━━ スケーリング則まとめ ━━
  B₃ (K=3, n=  6):  r = -0.8287
  B₄ (K=4, n= 24):  r = -0.7226
  B₅ (K=5, n=120):  r = -0.5651
  B₆ (K=6, n=720):  r = {r_b:.4f}

  ━━ 景観統計 ━━
  距離の平均: {dist_all.mean():.4f}
  距離の最大: {max_dist:.4f}  ({labels[max_idx]})
  identity に最近傍: {nearest['label']}  (dist={nearest['dist']:.4f})

  ━━ 先頭介入ごとの平均景観距離 ━━""")

for e, mean_d, cnt in sorted(first_means, key=lambda x: x[1]):
    bar = "█" * int(mean_d * 3)
    print(f"  E{e} first: {mean_d:.4f}  {bar}")

print(f"""
  ━━ Burau 四分位ごとの距離統計 ━━
  上位 25% (simple):  mean={dist_vals[mask_hi].mean():.4f}  std={dist_vals[mask_hi].std():.4f}
  中位 50%:           mean={dist_vals[mask_mid].mean():.4f}  std={dist_vals[mask_mid].std():.4f}
  下位 25% (complex): mean={dist_vals[mask_lo].mean():.4f}  std={dist_vals[mask_lo].std():.4f}
""")

print("=" * 65)
print(f"  W-7e 実験完了  出力先: {OUT}")
print("=" * 65)
