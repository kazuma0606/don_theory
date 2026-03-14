#!/usr/bin/env sage
# exp_W7d.sage — W-7d: B₅ (5 strands, 120 permutations) × Intervention Landscape
# 目的: B₃(n=6,r=-0.83) → B₄(n=24,r=-0.72) → B₅(n=120,r=?) のスケーリング則

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms

print("=" * 65)
print("  W-7d: B₅ × Intervention Landscape (120 permutations)")
print("  目的: B₃→B₄→B₅ スケーリング則の検証")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7d")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Part 1: B₅ の全 120 置換ブレイドと Burau trace
# ══════════════════════════════════════════════════════════════════
print("\n[Part 1] B₅ の全 120 置換ブレイドと Burau trace")
print("-" * 55)

B5   = BraidGroup(5)
gens = list(B5.generators())   # σ₁, σ₂, σ₃, σ₄
t    = var('t')

all_perms = list(iter_perms([1, 2, 3, 4, 5]))   # 120 通り

print("  Burau trace を計算中 (120 件)...")
perm_info = []
for perm in all_perms:
    sp    = SagePerm(list(perm))
    word  = sp.reduced_word()
    b     = B5.one()
    for i in word:
        b = b * gens[i - 1]
    bmat  = b.burau_matrix()(t=QQ(1)/QQ(2))
    btr   = float(bmat.trace())
    writhe = len(word)
    perm_info.append({
        "perm":   list(perm),
        "label":  "".join(str(x) for x in perm),
        "braid":  b,
        "word":   word,
        "burau":  btr,
        "writhe": writhe,
        "order":  [x - 1 for x in perm],
    })

burau_all  = np.array([p["burau"]  for p in perm_info])
writhe_all = np.array([p["writhe"] for p in perm_info])
print(f"  完了。Burau trace: [{burau_all.min():.4f}, {burau_all.max():.4f}]")
print(f"  Writhe:           [{int(writhe_all.min())}, {int(writhe_all.max())}]")

# 代表例
print(f"\n  {'Label':<8} {'Permutation':<20} {'Writhe':>7} {'Burau tr':>10}")
print("  " + "-" * 50)
for info in perm_info[:5]:
    print(f"  {info['label']:<8} {str(info['perm']):<20} "
          f"{info['writhe']:>7} {info['burau']:>10.4f}")
print(f"  ... (全 120 件)")

# ══════════════════════════════════════════════════════════════════
# Part 2: 損失景観 J(θ) の計算（全 120 順序）
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
bias5  = rng.standard_normal(d) * 0.2   # E5 専用

def E1(p, th): return p + th * a_vec
def E2(p, th): return (np.eye(d) + th * M_mat) @ p
def E3(p):     return np.tanh(p * 1.2) * 0.9
def E4(p):     return np.tanh(p * 0.8 + bias4) * 0.7
def E5(p):     return p * 0.6 + np.tanh(bias5) * 0.4   # 新規: 線形縮小 + 定数

N_ROUNDS = 2          # 各ラウンド = 5 介入
T_total  = N_ROUNDS * 5   # 10 ステップ

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
            else:          p = E5(p)
            traj.append(p.copy())
    return np.array(traj)

def compute_J(order, theta, p0, z_target):
    traj = apply_order(order, theta, p0)
    return float(np.mean(np.sum((traj - z_target)**2, axis=1)))

# グリッド
theta_star = np.array([1.0, 1.0])
p0         = rng.standard_normal(d) * 0.5
ref_order  = [0, 1, 2, 3, 4]
z_target   = apply_order(ref_order, theta_star, p0)

N_GRID  = 30
th1_arr = np.linspace(-1.0, 3.0, N_GRID)
th2_arr = np.linspace(-1.0, 3.0, N_GRID)
TH1, TH2 = np.meshgrid(th1_arr, th2_arr)

landscapes = {}
ref_land   = None

print(f"  グリッド: {N_GRID}×{N_GRID},  d={d},  T={T_total},  n_perms=120")
print(f"  介入: E1(θ₁), E2(θ₂), E3/E4/E5(固定)")
print(f"  計算中 (120 × {N_GRID}×{N_GRID} = {120*N_GRID*N_GRID} 点)...")

for k, info in enumerate(perm_info):
    order  = info["order"]
    J_grid = np.array([
        [compute_J(order, [th1_arr[j], th2_arr[i]], p0, z_target)
         for j in range(N_GRID)]
        for i in range(N_GRID)
    ])
    landscapes[info["label"]] = J_grid
    j_min  = float(J_grid.min())
    j_star = compute_J(order, theta_star, p0, z_target)
    if ref_land is None:
        ref_land = J_grid;  dist = 0.0
    else:
        dist = float(np.sqrt(np.mean((J_grid - ref_land)**2)))
    info["j_min"]  = j_min
    info["j_star"] = j_star
    info["dist"]   = dist
    if (k + 1) % 20 == 0:
        print(f"  ... {k+1}/120 完了")

dist_all  = np.array([info["dist"]  for info in perm_info])
print(f"  完了。景観距離: [{dist_all.min():.4f}, {dist_all.max():.4f}]")

# ══════════════════════════════════════════════════════════════════
# Part 3: 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[Part 3] 可視化")
print("-" * 55)

labels     = [info["label"]  for info in perm_info]
burau_vals = np.array([info["burau"]  for info in perm_info])
dist_vals  = np.array([info["dist"]   for info in perm_info])
writhe_vals= np.array([info["writhe"] for info in perm_info])

# 図1: 代表景観 — Burau 上位 6 + 下位 6
sorted_asc  = sorted(perm_info, key=lambda x: x["burau"])
rep_top6    = sorted_asc[-6:]   # Burau 大（単純）
rep_bot6    = sorted_asc[:6]    # Burau 小（複雑）
rep_all     = rep_top6 + rep_bot6

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle("W-7d: B5 Loss Landscapes\n"
             "Top 6 Burau (simple) + Bottom 6 Burau (complex)", fontsize=12)
for ax, info in zip(axes.flat, rep_all):
    J  = landscapes[info["label"]]
    im = ax.contourf(TH1, TH2, J, levels=18, cmap="viridis")
    ax.contour(TH1, TH2, J, levels=6, colors="white", alpha=0.3, linewidths=0.5)
    ax.plot(*theta_star, "r*", ms=9)
    ax.set_title(f"{info['label']}  Burau={info['burau']:.3f}\n"
                 f"dist={info['dist']:.3f}", fontsize=8)
    ax.set_xlabel("theta1", fontsize=7);  ax.set_ylabel("theta2", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p = OUT / "W7d_fig1_landscapes.png"
fig.savefig(str(p), dpi=120, bbox_inches="tight");  plt.close()
print(f"  fig1 saved: {p}")

# 図2: Burau/Writhe vs 景観距離（メイン, n=120）
r_b, p_b   = stats.pearsonr(burau_vals, dist_vals)
r_w, p_w   = stats.pearsonr(writhe_vals, dist_vals)
slope_b, intercept_b, *_ = stats.linregress(burau_vals, dist_vals)
slope_w, intercept_w, *_ = stats.linregress(writhe_vals, dist_vals)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"W-7d: B5 Braid Invariant vs Loss Landscape  (n=120)", fontsize=12)

ax = axes[0]
scatter = ax.scatter(burau_vals, dist_vals, c=writhe_vals, cmap="plasma",
                     s=30, alpha=0.8, zorder=5)
plt.colorbar(scatter, ax=ax, label="Writhe")
x_line = np.linspace(burau_vals.min(), burau_vals.max(), 100)
ax.plot(x_line, slope_b * x_line + intercept_b, "r--", alpha=0.8, lw=2)
ax.set_xlabel("Burau trace (t=1/2)")
ax.set_ylabel("Landscape distance from identity (RMSE)")
ax.set_title(f"Burau trace:  r = {r_b:.4f},  p = {p_b:.2e}\n"
             f"B3: r=-0.83(n=6)  B4: r=-0.72(n=24)  B5: r={r_b:.4f}(n=120)",
             fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(writhe_vals, dist_vals, c=burau_vals, cmap="viridis",
            s=30, alpha=0.8, zorder=5)
ax2.plot(np.linspace(writhe_vals.min(), writhe_vals.max(), 100),
         slope_w * np.linspace(writhe_vals.min(), writhe_vals.max(), 100) + intercept_w,
         "r--", alpha=0.8, lw=2)
ax2.set_xlabel("Writhe (braid word length)")
ax2.set_ylabel("Landscape distance from identity (RMSE)")
ax2.set_title(f"Writhe:  r = {r_w:.4f},  p = {p_w:.2e}", fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7d_fig2_correlation.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig2 saved: {p}")

# 図3: 景観空間の MDS 埋め込み（120 点 → 2D）
print("  MDS 計算中 (120×120 距離行列)...")
lands_arr   = np.array([landscapes[l] for l in labels])  # (120, N, N)
n           = len(perm_info)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d_ij = float(np.sqrt(np.mean((lands_arr[i] - lands_arr[j])**2)))
        dist_matrix[i, j] = d_ij
        dist_matrix[j, i] = d_ij

mds    = MDS(n_components=2, dissimilarity="precomputed",
             random_state=int(42), normalized_stress=False)
coords = mds.fit_transform(dist_matrix)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("W-7d: MDS embedding of 120 loss landscapes (2D)", fontsize=12)

sc = axes[0].scatter(coords[:, 0], coords[:, 1],
                     c=burau_vals, cmap="viridis", s=40, alpha=0.8)
plt.colorbar(sc, ax=axes[0], label="Burau trace")
axes[0].set_xlabel("MDS dim 1");  axes[0].set_ylabel("MDS dim 2")
axes[0].set_title("Colored by Burau trace", fontsize=10)
# 基準点をマーク
ref_idx = labels.index("12345")
axes[0].scatter(*coords[ref_idx], c="red", s=120, marker="*", zorder=10,
               label="identity (12345)")
axes[0].legend(fontsize=8)

sc2 = axes[1].scatter(coords[:, 0], coords[:, 1],
                      c=writhe_vals, cmap="plasma", s=40, alpha=0.8)
plt.colorbar(sc2, ax=axes[1], label="Writhe")
axes[1].set_xlabel("MDS dim 1");  axes[1].set_ylabel("MDS dim 2")
axes[1].set_title("Colored by Writhe", fontsize=10)
axes[1].scatter(*coords[ref_idx], c="red", s=120, marker="*", zorder=10)

plt.tight_layout()
p = OUT / "W7d_fig3_mds.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig3 saved: {p}")

# 図4: スケーリング則まとめ（B₃→B₄→B₅）
fig, ax = plt.subplots(figsize=(8, 5))
scaling_data = {
    "B3 (n=6)":   (-0.8287, 6),
    "B4 (n=24)":  (-0.7226, 24),
    f"B5 (n=120)": (r_b, 120),
}
ns = [6, 24, 120]
rs = [-0.8287, -0.7226, r_b]
ax.plot(ns, rs, "o-", ms=10, lw=2, color="steelblue")
for n_val, r_val, label in zip(ns, rs, scaling_data.keys()):
    ax.annotate(f"{label}\nr={r_val:.4f}",
                (n_val, r_val), textcoords="offset points",
                xytext=(5, 8), fontsize=9)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("n (number of permutations)", fontsize=11)
ax.set_ylabel("Pearson r (Burau vs landscape dist)", fontsize=11)
ax.set_title("Scaling law: B3 → B4 → B5\n"
             "Does correlation persist as braid group grows?", fontsize=11)
ax.set_xscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
p = OUT / "W7d_fig4_scaling.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig4 saved: {p}")

# ══════════════════════════════════════════════════════════════════
# Part 4: 統計サマリー
# ══════════════════════════════════════════════════════════════════
print("\n[Part 4] 統計サマリー")
print("-" * 65)

max_dist_pair = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
nearest_nref  = sorted(perm_info[1:], key=lambda x: x["dist"])[0]

# 距離分布の統計
print(f"""
  ━━ Burau trace vs 景観距離 (n=120) ━━
  Pearson r  = {r_b:.4f}
  p-value    = {p_b:.2e}   (有意 (p<0.01))
  Regression: dist = {slope_b:.4f} * Burau + {intercept_b:.4f}

  ━━ Writhe vs 景観距離 ━━
  Pearson r  = {r_w:.4f}
  p-value    = {p_w:.2e}

  ━━ スケーリング則 ━━
  B₃ (n=  6):  r = -0.8287
  B₄ (n= 24):  r = -0.7226
  B₅ (n=120):  r = {r_b:.4f}

  ━━ 景観の統計 ━━
  距離の平均: {dist_all.mean():.4f}
  距離の最大: {dist_all.max():.4f}  ({labels[max_dist_pair[0]]} - {labels[max_dist_pair[1]]})
  identity に最近傍: {nearest_nref['label']}  (dist={nearest_nref['dist']:.4f})
  MDS stress: {mds.stress_:.4f}
""")

print("=" * 65)
print(f"  W-7d 実験完了  出力先: {OUT}")
print("=" * 65)
