#!/usr/bin/env sage
# exp_W7c.sage — W-7c: B₄ (4 strands, 24 permutations) × Intervention Landscape
# 目的: W-7 の相関 r = -0.83 が 24 点に増やしても再現・強化されるか検証

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms

print("=" * 65)
print("  W-7c: B₄ × Intervention Landscape (24 permutations)")
print("  目的: 相関 r = -0.83 の統計的検証 (6点 → 24点)")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7c")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Part 1: B₄ の全 24 置換要素と Burau trace
# ══════════════════════════════════════════════════════════════════
print("\n[Part 1] B₄ の全 24 置換ブレイドと Burau trace")
print("-" * 55)

B4   = BraidGroup(4)
gens = list(B4.generators())   # σ₁, σ₂, σ₃
t    = var('t')

all_perms = list(iter_perms([1, 2, 3, 4]))   # 24 通り

perm_info = []
for perm in all_perms:
    sp   = SagePerm(list(perm))
    word = sp.reduced_word()          # 1-indexed simple transpositions
    b    = B4.one()
    for i in word:
        b = b * gens[i - 1]
    bmat = b.burau_matrix()(t=QQ(1)/QQ(2))
    btr  = float(bmat.trace())
    # Writhe = sum of exponents in the word
    writhe = len(word)               # positive word → writhe = length
    perm_info.append({
        "perm":   list(perm),
        "label":  "".join(str(x) for x in perm),
        "braid":  b,
        "word":   word,
        "burau":  btr,
        "writhe": writhe,
        "order":  [x - 1 for x in perm],   # 0-indexed
    })

# 代表例を印刷
print(f"\n  {'Label':<8} {'Permutation':<18} {'Word':<20} {'Writhe':>7} {'Burau tr':>10}")
print("  " + "-" * 68)
for info in perm_info[:8]:
    print(f"  {info['label']:<8} {str(info['perm']):<18} "
          f"{str(info['word']):<20} {info['writhe']:>7} {info['burau']:>10.4f}")
print(f"  ... (全 {len(perm_info)} 件)")

burau_all = np.array([p["burau"] for p in perm_info])
print(f"\n  Burau trace range: [{burau_all.min():.4f}, {burau_all.max():.4f}]")

# ══════════════════════════════════════════════════════════════════
# Part 2: 損失景観 J(θ) の計算（全 24 順序）
# ══════════════════════════════════════════════════════════════════
print("\n[Part 2] 損失景観 J(θ) の計算")
print("-" * 55)

rng = np.random.default_rng(int(42))
d   = 24

# ダイナミクス
A_dyn  = rng.standard_normal((d, d)) * 0.25
A_dyn  = A_dyn - A_dyn.T
b_bias = rng.standard_normal(d) * 0.1

def time_evolution(p):
    return np.tanh(A_dyn @ p + b_bias) * 0.5 + p * 0.4

# 4 つの介入演算子 (E1, E2 がパラメータ付き, E3・E4 は固定非線形)
a_vec = np.zeros(d);  a_vec[:d//4] = 1.0
M_mat = rng.standard_normal((d, d)) * 0.3
M_mat = M_mat / (np.linalg.norm(M_mat, 2) + 1e-8)
bias4 = rng.standard_normal(d) * 0.3   # E4 専用バイアス

def E1(p, th): return p + th * a_vec
def E2(p, th): return (np.eye(d) + th * M_mat) @ p
def E3(p):     return np.tanh(p * 1.2) * 0.9
def E4(p):     return np.tanh(p * 0.8 + bias4) * 0.7   # 新規追加

FUNCS = [E1, E2, E3, E4]

N_ROUNDS = 3        # 各ラウンド = 4 介入
T_total  = N_ROUNDS * 4   # 12 ステップ

def apply_order(order, theta, p0):
    """order: length-4 list of indices into FUNCS"""
    p    = p0.copy()
    traj = [p.copy()]
    for _ in range(N_ROUNDS):
        for idx in order:
            p = time_evolution(p)
            if   idx == 0: p = E1(p, theta[0])
            elif idx == 1: p = E2(p, theta[1])
            elif idx == 2: p = E3(p)
            else:          p = E4(p)
            traj.append(p.copy())
    return np.array(traj)

def compute_J(order, theta, p0, z_target):
    traj = apply_order(order, theta, p0)
    return float(np.mean(np.sum((traj - z_target)**2, axis=1)))

# 基準: 自然順 [E1,E2,E3,E4], θ* = (1,1)
theta_star = np.array([1.0, 1.0])
p0         = rng.standard_normal(d) * 0.5
ref_order  = [0, 1, 2, 3]
z_target   = apply_order(ref_order, theta_star, p0)

# グリッド
N_GRID  = 35
th1_arr = np.linspace(-1.0, 3.0, N_GRID)
th2_arr = np.linspace(-1.0, 3.0, N_GRID)
TH1, TH2 = np.meshgrid(th1_arr, th2_arr)

landscapes  = {}
ref_land    = None
print(f"  グリッド: {N_GRID}×{N_GRID},  d={d},  T={T_total}")
print(f"  介入: E1(θ₁), E2(θ₂), E3(固定), E4(固定)")
print(f"  θ* = {theta_star}")
print(f"\n  {'Label':<8} {'J_min':>8} {'J@θ*':>8} {'Dist_e':>10} {'Burau':>8}")
print("  " + "-" * 48)

for info in perm_info:
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
    print(f"  {info['label']:<8} {j_min:>8.4f} {j_star:>8.4f} {dist:>10.4f} {info['burau']:>8.4f}")

# ══════════════════════════════════════════════════════════════════
# Part 3: 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[Part 3] 可視化")
print("-" * 55)

# 図1: 代表 12 景観（Burau trace 順に並べる）
sorted_info = sorted(perm_info, key=lambda x: x["burau"], reverse=True)
rep12       = sorted_info[:12]   # Burau trace が高い上位 12

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle("W-7c: B4 Loss Landscapes (top-12 by Burau trace)\n"
             "higher Burau = more similar to identity", fontsize=12)
for ax, info in zip(axes.flat, rep12):
    J  = landscapes[info["label"]]
    im = ax.contourf(TH1, TH2, J, levels=20, cmap="viridis")
    ax.contour(TH1, TH2, J, levels=8, colors="white", alpha=0.3, linewidths=0.5)
    ax.plot(*theta_star, "r*", ms=10)
    ax.set_title(f"{info['label']}  Burau={info['burau']:.3f}\n"
                 f"J_min={info['j_min']:.3f}  dist={info['dist']:.3f}", fontsize=8)
    ax.set_xlabel("theta1", fontsize=7);  ax.set_ylabel("theta2", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p = OUT / "W7c_fig1_landscapes.png"
fig.savefig(str(p), dpi=130, bbox_inches="tight");  plt.close()
print(f"  fig1 saved: {p}")

# 図2: Burau trace vs 景観距離 — メイン結果
burau_vals = np.array([info["burau"] for info in perm_info])
dist_vals  = np.array([info["dist"]  for info in perm_info])

r_val, p_val = stats.pearsonr(burau_vals, dist_vals)
slope, intercept, *_ = stats.linregress(burau_vals, dist_vals)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"W-7c: B4 Braid Invariant vs Loss Landscape  (n=24)", fontsize=12)

ax = axes[0]
ax.scatter(burau_vals, dist_vals, s=60, alpha=0.7, zorder=5)
for info in perm_info:
    ax.annotate(info["label"], (info["burau"], info["dist"]),
                fontsize=6, textcoords="offset points", xytext=(2, 2))
x_line = np.linspace(burau_vals.min(), burau_vals.max(), 100)
ax.plot(x_line, slope * x_line + intercept, "r--", alpha=0.7, label=f"regression")
ax.set_xlabel("Burau trace (t=1/2)")
ax.set_ylabel("Landscape distance from identity (RMSE)")
ax.set_title(f"Pearson r = {r_val:.4f},  p = {p_val:.2e}\n"
             f"(W-7 reference: r = -0.8287, n=6)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Writhe との比較
writhe_vals = np.array([info["writhe"] for info in perm_info])
r_w, p_w    = stats.pearsonr(writhe_vals, dist_vals)

ax2 = axes[1]
ax2.scatter(writhe_vals, dist_vals, s=60, alpha=0.7, color="green", zorder=5)
sw, iw, *_ = stats.linregress(writhe_vals, dist_vals)
x2 = np.linspace(writhe_vals.min(), writhe_vals.max(), 100)
ax2.plot(x2, sw * x2 + iw, "r--", alpha=0.7)
ax2.set_xlabel("Writhe (braid word length)")
ax2.set_ylabel("Landscape distance from identity (RMSE)")
ax2.set_title(f"Writhe vs Dist:  r = {r_w:.4f},  p = {p_w:.2e}", fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7c_fig2_correlation.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig2 saved: {p}")

# 図3: 距離ヒートマップ（24×24 の景観間距離行列）
dist_matrix = np.zeros((24, 24))
labels24    = [info["label"] for info in perm_info]
lands_arr   = np.array([landscapes[l] for l in labels24])  # (24, N, N)
for i in range(24):
    for j in range(i+1, 24):
        d_ij = float(np.sqrt(np.mean((lands_arr[i] - lands_arr[j])**2)))
        dist_matrix[i, j] = d_ij
        dist_matrix[j, i] = d_ij

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(24));  ax.set_xticklabels(labels24, rotation=90, fontsize=7)
ax.set_yticks(range(24));  ax.set_yticklabels(labels24, fontsize=7)
ax.set_title("W-7c: Pairwise Landscape Distance Matrix (24x24)", fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p = OUT / "W7c_fig3_distance_matrix.png"
fig.savefig(str(p), dpi=130, bbox_inches="tight");  plt.close()
print(f"  fig3 saved: {p}")

# ══════════════════════════════════════════════════════════════════
# Part 4: 統計的検証
# ══════════════════════════════════════════════════════════════════
print("\n[Part 4] 統計的検証")
print("-" * 65)

print(f"""
  ━━ Burau trace vs 景観距離 ━━
  n = 24  (W-7: n = 6)
  Pearson r  = {r_val:.4f}   (W-7: r = -0.8287)
  p-value    = {p_val:.2e}   ({'有意 (p<0.01)' if p_val < 0.01 else '非有意'})
  Regression: dist = {slope:.4f} * Burau + {intercept:.4f}

  ━━ Writhe vs 景観距離 ━━
  Pearson r  = {r_w:.4f}
  p-value    = {p_w:.2e}   ({'有意 (p<0.01)' if p_w < 0.01 else '非有意'})

  ━━ 仮説検証 ━━
""")

# H1: r が B₃ と同程度維持されるか
print(f"  H1 (B₄ でも r ≈ -0.83 が成立): r = {r_val:.4f}")
print("    →", "B₄(n=24) でも強い負の相関を確認。統計的に有意。"
              if abs(r_val) > 0.6 and p_val < 0.01
              else "B₄ では相関が弱まった")

# H2: Writhe より Burau trace の方が予測力が高いか
print(f"\n  H2 (Burau vs Writhe の予測力): |r_Burau|={abs(r_val):.4f} vs |r_Writhe|={abs(r_w):.4f}")
if abs(r_val) > abs(r_w):
    print("    → Burau trace は Writhe より景観距離の良い予測子")
else:
    print("    → Writhe の方が景観距離との相関が強い")

# H3: 最遠の景観ペアは何か
max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
print(f"\n  H3 (最も遠い景観ペア): "
      f"{labels24[max_idx[0]]} - {labels24[max_idx[1]]}  "
      f"(dist = {dist_matrix[max_idx]:.4f})")

# 最近傍（identity 除く）
dists_from_id = [info["dist"] for info in perm_info[1:]]
nearest = perm_info[1 + np.argmin(dists_from_id)]
print(f"  H4 (identity に最も近い順序): {nearest['label']}  (dist = {nearest['dist']:.4f})")

print(f"""
=================================================================
  W-7c 実験完了
  出力先: {OUT}
=================================================================
""")
