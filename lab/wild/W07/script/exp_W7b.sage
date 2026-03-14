#!/usr/bin/env sage
# exp_W7b.sage — W-7b: Pure Braid × Intervention Landscape
# 問い: 置換が恒等でも、braid word の「位相」が異なると損失景観は変わるか？

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 65)
print("  W-7b: Pure Braid × Intervention Landscape")
print("  問い: 位相だけが異なるとき、損失景観は変わるか？")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7b")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Part 1: Pure Braid の代数構造
# ══════════════════════════════════════════════════════════════════
print("\n[Part 1] Pure Braid の代数構造")
print("-" * 50)

B  = BraidGroup(3)
s1, s2 = B.generators()
t = var('t')

# 全て pure braid（置換 = 恒等）— 正の生成元のみで構成
pure_braids = {
    "e":           B.one(),            # 交差なし（基準）
    "σ₁²":        s1**2,              # E1-E2 が 1 回巻き付く
    "σ₂²":        s2**2,              # E2-E3 が 1 回巻き付く
    "σ₁²σ₂²":    s1**2 * s2**2,      # 順番に巻き付く
    "σ₂²σ₁²":    s2**2 * s1**2,      # 逆順に巻き付く（H2 の主役）
    "σ₁²σ₂²σ₁²": s1**2 * s2**2 * s1**2,   # より複雑
    "σ₂²σ₁²σ₂²": s2**2 * s1**2 * s2**2,   # 鏡像
    "Δ²=(σ₁σ₂)³": (s1*s2)**3,        # full twist = B₃ の中心元
}

print(f"\n  {'Name':<16} {'Permutation':<14} {'Pure?':<7} {'Word':<20} {'Burau tr(½)'}")
print("  " + "-" * 65)
for name, b in pure_braids.items():
    perm  = list(b.permutation())
    word  = list(b.Tietze())
    is_pure = (perm == [1, 2, 3])
    bmat  = b.burau_matrix()(t=QQ(1)/QQ(2))
    btr   = float(bmat.trace())
    print(f"  {name:<16} {str(perm):<14} {'✓' if is_pure else '✗':<7} {str(word):<20} {btr:.4f}")

# ── 介入順序への変換 ──────────────────────────────────────────────
# σ₁ (gen 1) → E1 と E2 を交換 → 順序 [E2, E1, E3] = index [1,0,2]
# σ₂ (gen 2) → E2 と E3 を交換 → 順序 [E1, E3, E2] = index [0,2,1]
# e  (no gen) → 変更なし          → 順序 [E1, E2, E3] = index [0,1,2]

GEN_ORDER = {1: [1,0,2],  -1: [1,0,2],   # σ₁ / σ₁⁻¹ → 同じ置換
             2: [0,2,1],  -2: [0,2,1]}    # σ₂ / σ₂⁻¹

N_ROUNDS = 6   # 全 braid を 6 ラウンドで正規化（最長 word 長 = 6）

def braid_to_rounds(b, n=N_ROUNDS):
    """braid word → n ラウンドの介入順序リスト（不足は identity 補填）"""
    word   = list(b.Tietze())
    rounds = [GEN_ORDER.get(g, [0,1,2]) for g in word]
    while len(rounds) < n:
        rounds.append([0, 1, 2])  # identity 補填
    return rounds[:n]

T_total = N_ROUNDS * 3   # 18 ステップ
print(f"\n  N_ROUNDS={N_ROUNDS},  T_total={T_total}")
print(f"\n  {'Name':<16} ラウンド列（σ₁=[213] σ₂=[132] e=[123]）")
print("  " + "-" * 70)

LABELS = {(1,0,2): "σ₁[213]", (0,2,1): "σ₂[132]", (0,1,2): "e[123]"}
for name, b in pure_braids.items():
    rnds = braid_to_rounds(b)
    lbl  = " ".join(LABELS[tuple(r)] for r in rnds)
    print(f"  {name:<16} {lbl}")

# ══════════════════════════════════════════════════════════════════
# Part 2: 損失景観 J(θ) の計算
# ══════════════════════════════════════════════════════════════════
print("\n[Part 2] 損失景観 J(θ) の計算")
print("-" * 50)

rng = np.random.default_rng(int(42))
d   = 24

A_dyn  = rng.standard_normal((d, d)) * 0.25
A_dyn  = A_dyn - A_dyn.T
b_bias = rng.standard_normal(d) * 0.1

def time_evolution(p):
    return np.tanh(A_dyn @ p + b_bias) * 0.5 + p * 0.4

a_vec = np.zeros(d);  a_vec[:d//4] = 1.0
M_mat = rng.standard_normal((d, d)) * 0.3
M_mat = M_mat / (np.linalg.norm(M_mat, 2) + 1e-8)

def E1(p, th): return p + th * a_vec
def E2(p, th): return (np.eye(d) + th * M_mat) @ p
def E3(p):     return np.tanh(p * 1.2) * 0.9

def apply_rounds(rounds, theta, p0):
    p    = p0.copy()
    traj = [p.copy()]
    for rnd in rounds:
        for idx in rnd:
            p = time_evolution(p)
            if   idx == 0: p = E1(p, theta[0])
            elif idx == 1: p = E2(p, theta[1])
            else:          p = E3(p)
            traj.append(p.copy())
    return np.array(traj)

def compute_J(rounds, theta, p0, z_target):
    traj = apply_rounds(rounds, theta, p0)
    return float(np.mean(np.sum((traj - z_target)**2, axis=1)))

# 基準: e（全 identity ラウンド）, θ* = (1,1)
theta_star  = np.array([1.0, 1.0])
p0          = rng.standard_normal(d) * 0.5
ref_rounds  = braid_to_rounds(B.one())
z_target    = apply_rounds(ref_rounds, theta_star, p0)

# グリッド
N_GRID  = 40
th1_arr = np.linspace(-1.0, 3.0, N_GRID)
th2_arr = np.linspace(-1.0, 3.0, N_GRID)
TH1, TH2 = np.meshgrid(th1_arr, th2_arr)

landscapes = {}
summaries  = {}
ref_land   = None

print(f"  グリッド: {N_GRID}×{N_GRID},  d={d},  T={T_total}")
print(f"  θ* = {theta_star}")
print(f"\n  {'Name':<16} {'J_min':>8} {'J@θ*':>8} {'Dist_e':>10} {'Burau':>8}")
print("  " + "-" * 55)

for name, b in pure_braids.items():
    rnds   = braid_to_rounds(b)
    J_grid = np.array([
        [compute_J(rnds, [th1_arr[j], th2_arr[i]], p0, z_target)
         for j in range(N_GRID)]
        for i in range(N_GRID)
    ])
    landscapes[name] = J_grid
    j_min  = float(J_grid.min())
    j_star = compute_J(rnds, theta_star, p0, z_target)
    btr    = float(b.burau_matrix()(t=QQ(1)/QQ(2)).trace())

    if ref_land is None:
        ref_land = J_grid;  dist = 0.0
    else:
        dist = float(np.sqrt(np.mean((J_grid - ref_land)**2)))

    summaries[name] = {"j_min": j_min, "j_star": j_star, "dist": dist, "burau": btr}
    print(f"  {name:<16} {j_min:>8.4f} {j_star:>8.4f} {dist:>10.4f} {btr:>8.4f}")

# ══════════════════════════════════════════════════════════════════
# Part 3: 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[Part 3] 可視化")
print("-" * 50)

names = list(pure_braids.keys())

# 図1: 全景観（2×4）
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("W-7b: Pure Braid × Loss Landscape\n"
             "（全 braid の置換 = 恒等。位相だけが異なる）", fontsize=13)
for ax, name in zip(axes.flat, names):
    J  = landscapes[name]
    im = ax.contourf(TH1, TH2, J, levels=25, cmap="viridis")
    ax.contour(TH1, TH2, J, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    ax.plot(*theta_star, "r*", ms=12)
    ax.set_title(f"{name}\nJ_min={summaries[name]['j_min']:.3f}", fontsize=9)
    ax.set_xlabel("θ₁", fontsize=8);  ax.set_ylabel("θ₂", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p = OUT / "W7b_fig1_landscapes.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  図1 保存: {p}")

# 図2: e との差分
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("W-7b: 差分  J(braid) − J(e)  \n"
             "赤=braid の方が loss 大 / 青=小", fontsize=13)
for ax, name in zip(axes.flat, names):
    diff = landscapes[name] - landscapes["e"]
    vmax = max(abs(diff).max(), 1e-3)
    im   = ax.contourf(TH1, TH2, diff, levels=25, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
    ax.contour(TH1, TH2, diff, levels=[0], colors="black", linewidths=1)
    ax.plot(*theta_star, "r*", ms=12)
    ax.set_title(f"{name}\nRMSE={summaries[name]['dist']:.4f}", fontsize=9)
    ax.set_xlabel("θ₁", fontsize=8);  ax.set_ylabel("θ₂", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
p = OUT / "W7b_fig2_diff.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  図2 保存: {p}")

# 図3: 主要比較 ＋ Burau 相関
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("W-7b: 重要ペア比較", fontsize=12)

# σ₁²σ₂² vs σ₂²σ₁²（同じ生成元、順序違い）
d12  = landscapes["σ₁²σ₂²"] - landscapes["σ₂²σ₁²"]
vmax = max(abs(d12).max(), 1e-3)
im   = axes[0].contourf(TH1, TH2, d12, levels=25, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
axes[0].contour(TH1, TH2, d12, levels=[0], colors="black", linewidths=1)
axes[0].plot(*theta_star, "r*", ms=12)
rmse12 = float(np.sqrt(np.mean(d12**2)))
axes[0].set_title(f"σ₁²σ₂² − σ₂²σ₁²\nRMSE={rmse12:.4f}\n(同生成元・順序だけ違う)", fontsize=10)
axes[0].set_xlabel("θ₁");  axes[0].set_ylabel("θ₂")
plt.colorbar(im, ax=axes[0], fraction=0.046)

# σ₁²σ₂²σ₁² vs σ₂²σ₁²σ₂²（鏡像比較）
d_mir = landscapes["σ₁²σ₂²σ₁²"] - landscapes["σ₂²σ₁²σ₂²"]
vmax  = max(abs(d_mir).max(), 1e-3)
im    = axes[1].contourf(TH1, TH2, d_mir, levels=25, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax)
axes[1].contour(TH1, TH2, d_mir, levels=[0], colors="black", linewidths=1)
axes[1].plot(*theta_star, "r*", ms=12)
rmse_mir = float(np.sqrt(np.mean(d_mir**2)))
axes[1].set_title(f"σ₁²σ₂²σ₁² − σ₂²σ₁²σ₂²\nRMSE={rmse_mir:.4f}\n(鏡像 braid)", fontsize=10)
axes[1].set_xlabel("θ₁");  axes[1].set_ylabel("θ₂")
plt.colorbar(im, ax=axes[1], fraction=0.046)

# Burau trace vs 景観距離
burau_vals = [summaries[n]["burau"] for n in names]
dist_vals  = [summaries[n]["dist"]  for n in names]
axes[2].scatter(burau_vals, dist_vals, s=80, zorder=5)
for i, name in enumerate(names):
    axes[2].annotate(name, (burau_vals[i], dist_vals[i]),
                     textcoords="offset points", xytext=(5,3), fontsize=7)
if len(set(dist_vals)) > 1:
    r = float(np.corrcoef(burau_vals, dist_vals)[0, 1])
    axes[2].set_title(f"Burau trace vs Landscape dist\nr = {r:.4f}", fontsize=10)
else:
    axes[2].set_title("Burau trace vs Landscape dist", fontsize=10)
axes[2].set_xlabel("Burau trace (t=½)")
axes[2].set_ylabel("景観距離 from e (RMSE)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7b_fig3_compare.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  図3 保存: {p}")

# ══════════════════════════════════════════════════════════════════
# Part 4: 仮説検証
# ══════════════════════════════════════════════════════════════════
print("\n[Part 4] 数値サマリーと仮説検証")
print("-" * 65)
print(f"  {'Name':<16} {'J_min':>8} {'J@θ*':>8} {'Dist_e':>10} {'Burau':>8}")
print("  " + "-" * 55)
for name in names:
    s = summaries[name]
    print(f"  {name:<16} {s['j_min']:>8.4f} {s['j_star']:>8.4f} {s['dist']:>10.4f} {s['burau']:>8.4f}")

print("\n[仮説検証]")
print("-" * 65)

max_dist = max(summaries[n]["dist"] for n in names)
print(f"\n  H1（pure braid でも景観変化するか）")
print(f"    最大景観距離 (e からの RMSE) = {max_dist:.4f}")
print("    →", "置換が恒等でも、braid word の位相が景観に影響する" if max_dist > 0.1
               else "景観変化は軽微（pure braid の効果は小さい）")

d12 = float(np.sqrt(np.mean((landscapes["σ₁²σ₂²"] - landscapes["σ₂²σ₁²"])**2)))
print(f"\n  H2（生成元の順序非可換性）σ₁²σ₂² vs σ₂²σ₁²")
print(f"    景観距離 = {d12:.4f}")
print("    →", "純粋な braid レベルでも生成元の順序が景観に影響する（非可換性の残響）"
              if d12 > 0.01 else "生成元の順序は景観に影響しない")

btr_ft = summaries["Δ²=(σ₁σ₂)³"]["dist"]
print(f"\n  H3（Full twist Δ² の特殊性）")
print(f"    Δ² の景観距離 = {btr_ft:.4f}")
print("    →", f"Δ² は最{'も大きい' if btr_ft >= max_dist * 0.8 else 'も大きいわけではない'}効果を持つ"
              f" (max_dist={max_dist:.4f} の {100*btr_ft/max(max_dist,1e-9):.0f}%)")

burau_vals = [summaries[n]["burau"] for n in names]
dist_vals  = [summaries[n]["dist"]  for n in names]
if len(set(dist_vals)) > 1:
    r_pb = float(np.corrcoef(burau_vals, dist_vals)[0, 1])
    print(f"\n  H4（Burau trace との相関）r = {r_pb:.4f}")
    print("    →", "W-7 と同様に Braid 不変量が景観距離と相関する" if abs(r_pb) > 0.6
                  else "Pure braid では Burau trace との相関が弱い")

print(f"""
=================================================================
  W-7b 実験完了
  出力先: {OUT}
=================================================================
""")
