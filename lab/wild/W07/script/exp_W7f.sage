#!/usr/bin/env sage
# exp_W7f.sage — Jones 多項式 vs Burau trace の比較 (K=3,4,5,6)
# 問い: より豊かな位相不変量(Jones)を使うと Burau より高い相関が得られるか？
#       また、K が増えるにつれて p 値はどう推移するか？

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms
import time

print("=" * 65)
print("  W-7f: Jones多項式 vs Burau trace  (K=3,4,5,6)")
print("  統一セットアップで相関係数・p値のスケーリングを比較")
print("=" * 65)

# ── Jones 多項式の初期化ウォームアップ（初回は遅い）──────────────
print("  Jones 多項式の初期化中 (初回呼び出し)...")
_B3_warmup = BraidGroup(3)
_s1w, _s2w = _B3_warmup.generators()
_ = (_s1w * _s2w).jones_polynomial()   # warm-up: 以降は高速
print("  初期化完了")

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7f")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 設定: 全 K で統一
# ══════════════════════════════════════════════════════════════════
N_ROUNDS = 2      # 全 K 共通のラウンド数
N_GRID   = 25     # グリッドサイズ
SEED     = int(42)

# Jones 計算タイムアウト（1 braid あたり、秒）
JONES_TIME_LIMIT = 10.0

# ══════════════════════════════════════════════════════════════════
# 数値セットアップ（全 K 共有）
# ══════════════════════════════════════════════════════════════════
rng = np.random.default_rng(SEED)
d   = 24

A_dyn  = rng.standard_normal((d, d)) * 0.25
A_dyn  = A_dyn - A_dyn.T
b_bias = rng.standard_normal(d) * 0.1

def time_evolution(p):
    return np.tanh(A_dyn @ p + b_bias) * 0.5 + p * 0.4

a_vec = np.zeros(d);  a_vec[:d//4] = 1.0
M_mat = rng.standard_normal((d, d)) * 0.3
M_mat = M_mat / (np.linalg.norm(M_mat, 2) + 1e-8)
bias_list = [rng.standard_normal(d) * (0.2 + 0.05 * i) for i in range(4)]

def make_operators(K):
    """K 個の介入演算子を返す (E1,E2 はθ付き)"""
    ops = [
        lambda p, th, _=None: p + th[0] * a_vec,           # E1(θ₁)
        lambda p, th, _=None: (np.eye(d) + th[1]*M_mat)@p, # E2(θ₂)
        lambda p, th, _=None: np.tanh(p * 1.2) * 0.9,      # E3 固定
        lambda p, th, _=None: np.tanh(p * 0.8 + bias_list[0]) * 0.7,  # E4
        lambda p, th, _=None: p * 0.6 + np.tanh(bias_list[1]) * 0.4,  # E5
        lambda p, th, _=None: np.tanh(p + bias_list[2]) * 0.85,        # E6
    ]
    return ops[:K]

def apply_order(order, theta, p0, K):
    ops = make_operators(K)
    p   = p0.copy()
    traj = [p.copy()]
    for _ in range(N_ROUNDS):
        for idx in order:
            p = time_evolution(p)
            p = ops[idx](p, theta)
            traj.append(p.copy())
    return np.array(traj)

def compute_J(order, theta, p0, z_target, K):
    traj = apply_order(order, theta, p0, K)
    return float(np.mean(np.sum((traj - z_target)**2, axis=1)))

theta_star = np.array([1.0, 1.0])
p0_master  = rng.standard_normal(d) * 0.5

th1_arr = np.linspace(-1.0, 3.0, N_GRID)
th2_arr = np.linspace(-1.0, 3.0, N_GRID)
TH1, TH2 = np.meshgrid(th1_arr, th2_arr)

# ══════════════════════════════════════════════════════════════════
# Jones 多項式の特徴量抽出
# ══════════════════════════════════════════════════════════════════
def jones_features(b):
    """
    braid b の closure の Jones 多項式から特徴量を抽出。
    jp は Symbolic Expression in t (may contain sqrt(t) = t^(1/2))。
    評価は正の実数点のみ使用（複素数回避）。
    戻り値: dict, 失敗時: None
    """
    try:
        t0  = time.time()
        jp  = b.jones_polynomial()   # Symbolic Expression in variable t
        elapsed = time.time() - t0
        if jp == 0:
            return {"j_t025": 0.0, "j_t1": 0.0, "j_t4": 0.0,
                    "j_nterms": 0, "elapsed": elapsed}
        # ── 評価点: 正の実数のみ（sqrt(t) が実数になる） ──
        t_sym = SR.var('t')
        j_t025 = float(jp.subs(t=RR(0.25)))   # sqrt(t)=0.5
        j_t1   = float(jp.subs(t=RR(1.0)))    # sqrt(t)=1 → V(1) = link invariant
        j_t4   = float(jp.subs(t=RR(4.0)))    # sqrt(t)=2
        # ── 多項式の複雑さ: 項数 ──
        jp_exp = jp.expand()
        ops    = jp_exp.operands()
        n_terms = len(ops) if len(ops) >= 2 else 1
        return {"j_t025":   j_t025,   "j_t1":   j_t1,
                "j_t4":     j_t4,     "j_nterms": float(n_terms),
                "elapsed":  elapsed}
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════════════════
# メインループ: K=3,4,5,6
# ══════════════════════════════════════════════════════════════════
results = {}   # K → dict of arrays

for K in [3, 4, 5, 6]:
    print(f"\n{'='*55}")
    print(f"  K={K}  ({K}! = {factorial(K)} 順列)")
    print(f"{'='*55}")

    BK   = BraidGroup(K)
    gK   = list(BK.generators())
    t_var = var('t')

    all_perms = list(iter_perms(list(range(1, K+1))))
    n_perms   = len(all_perms)

    # ── 代数的特徴量の計算 ──────────────────────────────────────
    print(f"  [1] Burau trace + Jones多項式 を計算中 ({n_perms} 件)...")
    burau_arr   = np.zeros(n_perms)
    writhe_arr  = np.zeros(n_perms, dtype=int)
    j_half_arr  = np.full(n_perms, np.nan)
    j_neg1_arr  = np.full(n_perms, np.nan)
    j_span_arr  = np.full(n_perms, np.nan)
    j_l1_arr    = np.full(n_perms, np.nan)
    jones_ok    = True   # このK でJonesが使えるか

    t_jones_total = 0.0
    for i, perm in enumerate(all_perms):
        sp   = SagePerm(list(perm))
        word = sp.reduced_word()
        b    = BK.one()
        for idx in word:
            b = b * gK[idx - 1]
        # Burau
        bmat = b.burau_matrix()(t=QQ(1)/QQ(2))
        burau_arr[i]  = float(bmat.trace())
        writhe_arr[i] = len(word)
        # Jones（全件計算。ウォームアップ済みなので高速）
        if jones_ok:
            jf = jones_features(b)
            if jf is None:
                jones_ok = False
                print(f"    Jones 計算エラー → K={K} は Burau のみ")
            else:
                j_half_arr[i] = jf["j_t025"]
                j_neg1_arr[i] = jf["j_t1"]
                j_span_arr[i] = jf["j_t4"]
                j_l1_arr[i]   = jf["j_nterms"]
                t_jones_total += jf["elapsed"]
                # 最初の 5 件でペースを確認
                if i == 4 and t_jones_total > 0:
                    avg = t_jones_total / 5
                    est = avg * n_perms
                    if est > 180:
                        jones_ok = False
                        j_half_arr[:] = np.nan
                        print(f"    Jones 推定時間 {est:.0f}s > 3分 → K={K} はスキップ")
                    else:
                        print(f"    Jones 速度確認: 平均 {avg*1000:.1f}ms/件, 推定 {est:.0f}s")
        if (i+1) % max(1, n_perms//4) == 0:
            print(f"    {i+1}/{n_perms} 完了  (Jones: {'有効' if jones_ok else '無効'})")

    print(f"  Burau trace: [{burau_arr.min():.3f}, {burau_arr.max():.3f}]")
    if jones_ok:
        valid = ~np.isnan(j_half_arr)
        print(f"  Jones(q=1/2) valid: {valid.sum()}/{n_perms},  "
              f"range=[{np.nanmin(j_half_arr):.3f}, {np.nanmax(j_half_arr):.3f}]")

    # ── 景観距離の計算 ───────────────────────────────────────────
    print(f"  [2] 景観距離を計算中 ({n_perms} × {N_GRID}×{N_GRID} = {n_perms*N_GRID*N_GRID} 点)...")
    p0       = p0_master.copy()
    ref_ord  = list(range(K))
    z_target = apply_order(ref_ord, theta_star, p0, K)

    dist_arr = np.zeros(n_perms)
    ref_flat = None
    for i, perm in enumerate(all_perms):
        order  = [x - 1 for x in perm]
        J_grid = np.array([
            [compute_J(order, [th1_arr[j], th2_arr[ii]], p0, z_target, K)
             for j in range(N_GRID)]
            for ii in range(N_GRID)
        ])
        flat = J_grid.ravel()
        if ref_flat is None:
            ref_flat = flat;  dist_arr[i] = 0.0
        else:
            dist_arr[i] = float(np.sqrt(np.mean((flat - ref_flat)**2)))
    print(f"  景観距離: [{dist_arr.min():.4f}, {dist_arr.max():.4f}]")

    # ── 相関計算 ─────────────────────────────────────────────────
    print(f"  [3] 相関計算")
    corrs = {}
    features = {
        "Burau(t=½)":   burau_arr,
        "Writhe":        writhe_arr.astype(float),
    }
    if jones_ok:
        features["Jones(t=¼)"]   = j_half_arr   # j_t025
        features["Jones(t=1)"]   = j_neg1_arr   # j_t1
        features["Jones(t=4)"]   = j_span_arr   # j_t4
        features["Jones #terms"] = j_l1_arr     # n_terms

    for fname, fvals in features.items():
        mask = ~np.isnan(fvals) & ~np.isnan(dist_arr)
        if mask.sum() < 3:
            corrs[fname] = (np.nan, np.nan)
            continue
        r, p = stats.pearsonr(fvals[mask], dist_arr[mask])
        corrs[fname] = (float(r), float(p))
        print(f"    {fname:<20}: r={r:+.4f}  p={p:.2e}  (n={mask.sum()})")

    results[K] = {
        "n":          n_perms,
        "burau":      burau_arr,
        "writhe":     writhe_arr,
        "dist":       dist_arr,
        "j_half":     j_half_arr,
        "j_neg1":     j_neg1_arr,
        "j_span":     j_span_arr,
        "j_l1":       j_l1_arr,
        "jones_ok":   jones_ok,
        "corrs":      corrs,
    }

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

FEAT_COLORS = {
    "Burau(t=½)":    "steelblue",
    "Writhe":         "gray",
    "Jones(t=¼)":    "tomato",
    "Jones(t=1)":    "darkorange",
    "Jones(t=4)":    "purple",
    "Jones #terms":  "green",
}

# 図1: r vs K  と  |r| vs K
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("W-7f: Scaling of correlation with landscape distance\n"
             "Burau trace vs Jones polynomial features", fontsize=12)

# 全フィーチャーの r を収集
all_features = ["Burau(t=½)", "Writhe", "Jones(t=¼)", "Jones(t=1)", "Jones(t=4)", "Jones #terms"]
K_vals = sorted(results.keys())

for fname in all_features:
    r_vals = []
    k_list = []
    for K in K_vals:
        if fname in results[K]["corrs"]:
            r, p = results[K]["corrs"][fname]
            if not np.isnan(r):
                r_vals.append(r)
                k_list.append(K)
    if len(k_list) < 2:
        continue
    color = FEAT_COLORS.get(fname, "black")
    lw    = 2.5 if "Burau" in fname or "Jones(q=½)" in fname else 1.2
    ls    = "-"  if "Burau" in fname or "Jones(q=½)" in fname else "--"
    axes[0].plot(k_list, r_vals, "o" + ls, ms=8, lw=lw, color=color, label=fname)
    axes[1].plot(k_list, [abs(r) for r in r_vals], "o" + ls, ms=8, lw=lw,
                 color=color, label=fname)

axes[0].axhline(0, color="black", lw=0.8, alpha=0.5)
axes[0].set_xlabel("K (number of strands / interventions)", fontsize=11)
axes[0].set_ylabel("Pearson r", fontsize=11)
axes[0].set_title("r vs K  (negative = complex braid → large distance)", fontsize=10)
axes[0].legend(fontsize=8);  axes[0].grid(True, alpha=0.3)
axes[0].set_xticks([3,4,5,6])

axes[1].set_xlabel("K (number of strands / interventions)", fontsize=11)
axes[1].set_ylabel("|Pearson r|", fontsize=11)
axes[1].set_title("|r| vs K  (higher = better predictor)", fontsize=10)
axes[1].legend(fontsize=8);  axes[1].grid(True, alpha=0.3)
axes[1].set_xticks([3,4,5,6]);  axes[1].set_ylim(0, 1)

plt.tight_layout()
p = OUT / "W7f_fig1_r_scaling.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig1 saved: {p}")

# 図2: p 値 vs K（log scale）
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("W-7f: p-value scaling vs K\n(lower = more statistically significant)", fontsize=12)

for fname in all_features:
    p_vals = []
    k_list = []
    for K in K_vals:
        if fname in results[K]["corrs"]:
            r, p_v = results[K]["corrs"][fname]
            if not np.isnan(p_v):
                p_vals.append(p_v)
                k_list.append(K)
    if len(k_list) < 2:
        continue
    color = FEAT_COLORS.get(fname, "black")
    lw    = 2.5 if "Burau" in fname or "Jones(q=½)" in fname else 1.2
    ls    = "-"  if "Burau" in fname or "Jones(q=½)" in fname else "--"
    ax.semilogy(k_list, p_vals, "o" + ls, ms=8, lw=lw, color=color, label=fname)

ax.axhline(0.01, color="red", lw=1, alpha=0.6, linestyle=":", label="p=0.01")
ax.axhline(0.05, color="orange", lw=1, alpha=0.6, linestyle=":", label="p=0.05")
ax.set_xlabel("K (number of strands)", fontsize=11)
ax.set_ylabel("p-value (log scale)", fontsize=11)
ax.legend(fontsize=8, loc="upper right");  ax.grid(True, alpha=0.3)
ax.set_xticks([3,4,5,6])
plt.tight_layout()
p = OUT / "W7f_fig2_pvalue.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig2 saved: {p}")

# 図3: K=4 での Burau vs Jones の散布図比較（Jones が有効な最大 K）
best_K_jones = max((K for K in K_vals if results[K]["jones_ok"]), default=None)
if best_K_jones is not None:
    R = results[best_K_jones]
    dist_v = R["dist"]
    valid  = ~np.isnan(R["j_half"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"W-7f: Burau vs Jones features  (K={best_K_jones}, n={R['n']})", fontsize=12)

    panels = [
        ("Burau(t=½)",  R["burau"],    "steelblue"),
        ("Jones(t=¼)",  R["j_half"],   "tomato"),
        ("Jones(t=1)",  R["j_neg1"],   "darkorange"),
    ]
    for ax, (fname, fvals, color) in zip(axes, panels):
        m = ~np.isnan(fvals)
        if m.sum() < 3:
            ax.set_visible(False); continue
        r, pv = stats.pearsonr(fvals[m], dist_v[m])
        sl, ic, *_ = stats.linregress(fvals[m], dist_v[m])
        ax.scatter(fvals[m], dist_v[m], s=25, alpha=0.6, color=color)
        x_line = np.linspace(np.nanmin(fvals), np.nanmax(fvals), 100)
        ax.plot(x_line, sl * x_line + ic, "k--", lw=1.5)
        ax.set_xlabel(fname, fontsize=10)
        ax.set_ylabel("Landscape distance", fontsize=10)
        ax.set_title(f"r = {r:.4f},  p = {pv:.2e}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = OUT / "W7f_fig3_scatter.png"
    fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
    print(f"  fig3 saved: {p}")

# 図4: Jones L1 / span も含めた全特徴量の r まとめ（バーチャート）
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("W-7f: All features — |r| for each K", fontsize=12)

for ax, K in zip(axes.flat, K_vals):
    R = results[K]
    fnames = [f for f in all_features if f in R["corrs"]]
    r_abs  = [abs(R["corrs"][f][0]) for f in fnames if not np.isnan(R["corrs"][f][0])]
    fnames = [f for f in fnames if not np.isnan(R["corrs"][f][0])]
    colors = [FEAT_COLORS.get(f, "gray") for f in fnames]
    bars   = ax.bar(range(len(fnames)), r_abs, color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(fnames)))
    ax.set_xticklabels(fnames, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Pearson r|")
    ax.set_title(f"K={K}  (n={R['n']})", fontsize=10)
    ax.set_ylim(0, 1);  ax.grid(True, alpha=0.3, axis="y")
    for bar, r_v in zip(bars, r_abs):
        ax.text(bar.get_x() + bar.get_width()/2, r_v + 0.01, f"{r_v:.3f}",
                ha="center", va="bottom", fontsize=7)

plt.tight_layout()
p = OUT / "W7f_fig4_bar.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig4 saved: {p}")

# ══════════════════════════════════════════════════════════════════
# サマリーテーブル
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  総合サマリー: 相関係数 r と p 値")
print("=" * 70)
header = f"  {'Feature':<20}" + "".join(f"  K={K}(n={results[K]['n']})" for K in K_vals)
print(header)
print("  " + "-" * (len(header) - 2))

for fname in all_features:
    row = f"  {fname:<20}"
    for K in K_vals:
        R = results[K]
        if fname in R["corrs"]:
            r, pv = R["corrs"][fname]
            if np.isnan(r):
                row += "       N/A    "
            else:
                stars = "***" if pv < 0.001 else "** " if pv < 0.01 else "*  " if pv < 0.05 else "   "
                row += f"  {r:+.3f}{stars}"
        else:
            row += "       N/A    "
    print(row)

print("\n  *** p<0.001  ** p<0.01  * p<0.05")

print(f"""
  ━━ Jones 多項式が利用できた K ━━""")
for K in K_vals:
    status = "有効" if results[K]["jones_ok"] else "タイムアウト/スキップ"
    print(f"  K={K}: Jones = {status}")

print(f"""
=================================================================
  W-7f 実験完了  出力先: {OUT}
=================================================================
""")
