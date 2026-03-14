#!/usr/bin/env sage
# -*- coding: utf-8 -*-
"""
exp_W7.sage — Braid Group B_3 × Intervention Landscape

目的:
  K=3 の介入列を編み紐（braid）として表現し、
  「Braid の代数構造（不変量・置換・関係式）は
   損失景観の幾何と対応するか？」を数値的に探索する。

構成:
  Part 1: SageMath で B_3 の元を列挙し置換・Burau 行列・Alexander 多項式を計算
  Part 2: 各介入順序（6通り）に対して J(θ) を 2D グリッドで計算
  Part 3: 景観の可視化（6景観の並置・差分・距離行列）
  Part 4: Braid 不変量と景観距離の相関分析

仮説:
  H1: 介入順序で損失景観が変わる（非可換性の定量的確認）
  H2: Braid 不変量（Burau trace）が景観の「幾何的距離」と相関する
  H3: Braid 関係式 σ₁σ₂σ₁ = σ₂σ₁σ₂ に対応する順序は同じ景観を持つ

実行:
  conda activate sage
  cd lab/wild
  sage exp_W7.sage
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path(__file__).parent / "results" / "W7"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  W-7: Braid Group B₃ × Intervention Landscape")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════
# Part 1: B_3 の代数構造
# ═══════════════════════════════════════════════════════════════
print("\n[Part 1] BraidGroup(3) の代数構造")
print("-" * 50)

B3 = BraidGroup(3)
s1, s2 = B3.gens()

# S_3 に対応する B_3 の 6 元（置換群の全元）
braids = {
    'e':      B3.one(),
    's1':     s1,
    's2':     s2,
    's1s2':   s1 * s2,
    's2s1':   s2 * s1,
    's1s2s1': s1 * s2 * s1,
}

# Braid 関係式の確認: σ₁σ₂σ₁ = σ₂σ₁σ₂
assert s1*s2*s1 == s2*s1*s2, "Braid relation failed!"
print(f"  Braid relation σ₁σ₂σ₁ = σ₂σ₁σ₂ : CONFIRMED")
print(f"  (= full twist / longest element of S_3)\n")

# 各 braid の置換・Writhe・Burau trace
print(f"  {'Name':<10} {'Permutation':<18} {'Writhe':>7}  {'Burau tr (t=½)':>15}")
print("  " + "-" * 55)

perm_to_order = {}   # braid 名 → 介入適用順序（0-indexed）
burau_traces  = {}   # braid 名 → Burau matrix trace at t=0.5
t_val = QQ(1) / QQ(2)

for name, b in braids.items():
    perm   = b.permutation()          # Sage の Permutation オブジェクト
    writhe = sum(b.Tietze())          # writhe = signed crossing 数
    bm     = b.burau_matrix()         # Burau 表現行列（t のシンボリック行列）
    trace  = sum(bm[i, i].subs(t=t_val) for i in range(3))
    burau_traces[name] = float(trace)

    # 置換 π から介入順序を導出
    # 「底から読んだ順」= π^{-1} が適用される順序
    pi_inv = {int(perm(i)): i for i in range(1, 4)}
    order  = [pi_inv[j] - 1 for j in range(1, 4)]  # 0-indexed
    perm_to_order[name] = order

    print(f"  {name:<10} {str([int(perm(i)) for i in range(1,4)]):<18}"
          f" {writhe:>7}  {float(trace):>15.4f}")

# Alexander 多項式
print("\n  Alexander 多項式:")
alex_polys = {}
for name, b in braids.items():
    p = b.alexander_polynomial()
    alex_polys[name] = p
    print(f"    {name:<10}: {p}")

print("\n  介入順序マッピング（1=E1, 2=E2, 3=E3）:")
label_map = {
    'e':      'E1→E2→E3  (identity)',
    's1':     'E2→E1→E3  (swap 1,2)',
    's2':     'E1→E3→E2  (swap 2,3)',
    's1s2':   'E3→E1→E2  (3-cycle)',
    's2s1':   'E2→E3→E1  (3-cycle⁻¹)',
    's1s2s1': 'E3→E2→E1  (full reverse)',
}
for name, order in perm_to_order.items():
    print(f"    {name:<10}: {[o+1 for o in order]}  {label_map[name]}")

# ═══════════════════════════════════════════════════════════════
# Part 2: 介入演算子と損失景観の計算
# ═══════════════════════════════════════════════════════════════
print("\n[Part 2] 損失景観 J(θ) の計算")
print("-" * 50)

rng = np.random.default_rng(int(42))

# 状態空間設定
d = 24     # 状態次元
T = 10     # 時間ステップ数

# 時間発展ダイナミクス（安定な非線形系）
A_dyn   = rng.standard_normal((d, d)) * 0.25
A_dyn   = A_dyn - A_dyn.T          # 歪対称 → 固有値が純虚数 → 安定
b_bias  = rng.standard_normal(d) * 0.1

def time_evolution(p):
    return np.tanh(A_dyn @ p + b_bias) * 0.5 + p * 0.4

# ── 3 つの介入演算子 ──────────────────────────────────────────
# E1: 加法的（sparse）, E2: 線形, E3: 非線形（固定）
a_vec = np.zeros(d)
a_vec[:d // 4] = 1.0                        # sparse additive vector

M_mat = rng.standard_normal((d, d)) * 0.3  # 線形介入行列
M_mat = M_mat / (np.linalg.norm(M_mat, 2) + 1e-8)  # スペクトルノルム正規化

def E1(p, th1):  return p + th1 * a_vec
def E2(p, th2):  return (np.eye(d) + th2 * M_mat) @ p
def E3(p):       return np.tanh(p * 1.2) * 0.9

def apply_seq(order, theta, p0):
    """指定した介入順序で T ステップの軌道を返す"""
    p = p0.copy()
    traj = [p.copy()]
    for t in range(T):
        p = time_evolution(p)
        idx = order[t % 3]
        if   idx == 0: p = E1(p, theta[0])
        elif idx == 1: p = E2(p, theta[1])
        else:          p = E3(p)
        traj.append(p.copy())
    return np.array(traj)   # (T+1, d)

def compute_J(order, theta, p0, z_target):
    traj = apply_seq(order, theta, p0)
    return float(np.mean(np.sum((traj - z_target) ** 2, axis=1)))

# 目標軌道（基準順序 E1→E2→E3, θ*=(1,1)）
p0        = rng.standard_normal(d) * 0.5
theta_star = np.array([1.0, 1.0])
z_target   = apply_seq([0, 1, 2], theta_star, p0)

# 2D パラメータグリッド
N = 45
th_grid = np.linspace(-2.5, 2.5, N)
TH1, TH2 = np.meshgrid(th_grid, th_grid)

print(f"  グリッド: {N}×{N},  d={d},  T={T}")
print(f"  θ* = {theta_star}")

landscapes = {}
for name, order in perm_to_order.items():
    print(f"  {name:<10} [{', '.join(f'E{o+1}' for o in order)}] ...", end='', flush=True)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            J[i, j] = compute_J(order, np.array([TH1[i,j], TH2[i,j]]), p0, z_target)
    landscapes[name] = J
    print(f"  min={J.min():.4f}  max={J.max():.4f}  θ*-val={compute_J(order, theta_star, p0, z_target):.4f}")

# ═══════════════════════════════════════════════════════════════
# Part 3: 可視化
# ═══════════════════════════════════════════════════════════════
print("\n[Part 3] 可視化")
print("-" * 50)

names     = list(braids.keys())
vmin_all  = min(L.min() for L in landscapes.values())
vmax_all  = float(np.percentile(np.concatenate([L.ravel() for L in landscapes.values()]), 92))

# ── 図 1: 6景観の並置 ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
for ax, name in zip(axes.flat, names):
    J  = landscapes[name]
    im = ax.contourf(TH1, TH2, J, levels=25, cmap='plasma',
                     vmin=vmin_all, vmax=vmax_all)
    ax.set_title(
        f"braid: {name}\n{label_map[name]}\nBurau tr = {burau_traces[name]:.3f}",
        fontsize=9)
    ax.set_xlabel('θ₁', fontsize=9)
    ax.set_ylabel('θ₂', fontsize=9)
    ax.plot(*theta_star, 'w*', markersize=12, markeredgecolor='k', label='θ*=(1,1)')
    ax.axhline(0, color='white', lw=0.4, alpha=0.4)
    ax.axvline(0, color='white', lw=0.4, alpha=0.4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    'W-7: Loss Landscapes J(θ) for All 6 Orderings in B₃\n'
    'Do different braid orderings produce different landscapes?',
    fontsize=12, fontweight='bold')
plt.tight_layout()
p1 = OUT / "W7_fig1_landscapes.png"
fig.savefig(str(p1), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  図1 保存: {p1}")

# ── 図 2: 差分景観（identity braid との差） ──────────────────
J_ref = landscapes['e']
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
for ax, name in zip(axes.flat, names):
    diff = landscapes[name] - J_ref
    vd   = float(np.abs(diff).max())
    im   = ax.contourf(TH1, TH2, diff, levels=25, cmap='RdBu_r',
                       vmin=-vd, vmax=vd)
    ax.set_title(f"J({name}) − J(e)\nmax|Δ| = {vd:.4f}", fontsize=9)
    ax.set_xlabel('θ₁', fontsize=9)
    ax.set_ylabel('θ₂', fontsize=9)
    ax.plot(*theta_star, 'k*', markersize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    'W-7: Landscape Differences from Identity Ordering\n'
    'Red = ordering gives higher loss, Blue = lower loss than E1→E2→E3',
    fontsize=12, fontweight='bold')
plt.tight_layout()
p2 = OUT / "W7_fig2_diff.png"
fig.savefig(str(p2), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  図2 保存: {p2}")

# ── 図 3: Braid 不変量 vs 景観距離 + 距離行列 ─────────────────
landscape_dists = {
    name: float(np.sqrt(np.mean((J - J_ref) ** 2)))
    for name, J in landscapes.items()
}

fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# 散布図: Burau trace vs landscape distance
ax1 = fig.add_subplot(gs[0])
colors = plt.cm.tab10(np.linspace(0, 1, 6))
for i, name in enumerate(names):
    ax1.scatter(burau_traces[name], landscape_dists[name],
                s=140, color=colors[i], zorder=3,
                label=f"{name} ({label_map[name].split()[0]})")
    ax1.annotate(name, (burau_traces[name], landscape_dists[name]),
                 textcoords='offset points', xytext=(6, 4), fontsize=8)
ax1.set_xlabel('Burau Matrix Trace  (t = ½)', fontsize=11)
ax1.set_ylabel('L2 Distance from Identity Landscape', fontsize=11)
ax1.set_title('Braid Invariant vs Landscape Distance\n'
              'H2: Do similar braids give similar landscapes?', fontsize=10)
ax1.legend(fontsize=7, loc='upper left')
ax1.grid(alpha=0.3)

# 距離行列
ax2 = fig.add_subplot(gs[1])
n   = len(names)
D   = np.zeros((n, n))
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        D[i, j] = float(np.sqrt(np.mean((landscapes[n1] - landscapes[n2]) ** 2)))
im = ax2.imshow(D, cmap='YlOrRd', interpolation='nearest')
ax2.set_xticks(range(n)); ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(n)); ax2.set_yticklabels(names, fontsize=9)
ax2.set_title('Pairwise Landscape Distance Matrix\n(Brighter = More Different)', fontsize=10)
for i in range(n):
    for j in range(n):
        ax2.text(j, i, f'{D[i,j]:.3f}', ha='center', va='center',
                 fontsize=7, color='black' if D[i,j] < D.max()*0.6 else 'white')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

fig.suptitle('W-7: Braid Structure vs Loss Landscape Geometry', fontsize=13, fontweight='bold')
p3 = OUT / "W7_fig3_invariant_distance.png"
fig.savefig(str(p3), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  図3 保存: {p3}")

# ── 図 4: 3D 景観サーフェス（全 6 braid） ─────────────────────
fig = plt.figure(figsize=(18, 12))
for k, name in enumerate(names):
    ax  = fig.add_subplot(2, 3, k+1, projection='3d')
    J   = landscapes[name]
    J_c = np.clip(J, vmin_all, vmax_all)
    ax.plot_surface(TH1, TH2, J_c, cmap='viridis', alpha=0.85,
                    rstride=2, cstride=2, linewidth=0)
    ax.set_title(f'{name}: {label_map[name]}', fontsize=8)
    ax.set_xlabel('θ₁', fontsize=8); ax.set_ylabel('θ₂', fontsize=8)
    ax.set_zlabel('J', fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle('W-7: 3D Loss Surface for Each Braid Ordering\n'
             'Topology of the landscape — how does ordering reshape the bowl?',
             fontsize=12, fontweight='bold')
plt.tight_layout()
p4 = OUT / "W7_fig4_3d_surfaces.png"
fig.savefig(str(p4), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  図4 保存: {p4}")

# ═══════════════════════════════════════════════════════════════
# Part 4: 数値サマリーと考察
# ═══════════════════════════════════════════════════════════════
print("\n[Part 4] 数値サマリー")
print("-" * 70)
print(f"  {'Braid':<10} {'Order':<16} {'J_min':>8} {'J@θ*':>8} "
      f"{'Dist_e':>10} {'Burau_tr':>10}")
print("  " + "-" * 65)
for name in names:
    J     = landscapes[name]
    order = perm_to_order[name]
    j_at  = compute_J(order, theta_star, p0, z_target)
    print(f"  {name:<10} {str([o+1 for o in order]):<16} "
          f"{J.min():>8.4f} {j_at:>8.4f} "
          f"{landscape_dists[name]:>10.4f} {burau_traces[name]:>10.4f}")

print("\n[仮説の検証結果]")
print("-" * 70)

# H1: 景観は順序で変わるか？
max_dist = max(landscape_dists.values())
print(f"\n  H1 (非可換性): 最大景観距離 = {max_dist:.4f}")
print(f"    → {'景観は順序で有意に変化する' if max_dist > 0.01 else '景観はほぼ同一（非可換性が小さい）'}")

# H2: Burau trace と景観距離の相関
bt_vals = [burau_traces[n] for n in names]
ld_vals = [landscape_dists[n] for n in names]
bt_arr  = np.array(bt_vals)
ld_arr  = np.array(ld_vals)
corr    = float(np.corrcoef(bt_arr, ld_arr)[0, 1])
print(f"\n  H2 (Braid 不変量との相関): Pearson r = {corr:.4f}")
print(f"    Burau trace vs landscape distance の相関係数")
print(f"    → {'強い相関あり' if abs(corr) > 0.7 else '中程度の相関' if abs(corr) > 0.4 else '相関弱い・なし'}")

# H3: Braid 関係式 s1s2s1 = s2s1s2
#     （今回は置換が同じなので自動的に同一景観）
print(f"\n  H3 (Braid 関係式): σ₁σ₂σ₁ と σ₂σ₁σ₂ は B₃ で同じ元")
print(f"    → 同じ置換 → 同じ介入順序 → 景観距離 = 0 (自動的に成立)")
print(f"\n  ★ 次の問い (W-7b): 純 Braid（置換は恒等、位相だけ違う）は")
print(f"     同じ介入順序でも異なる景観を生むか？")
print(f"     例: (σ₁σ₂)³ = full twist（B₃ の中心元）")
print(f"         σ₁²  = 同じ strand を 2 回交差")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  W-7 実験完了")
print(f"  出力先: {OUT}")
print("=" * 65)
