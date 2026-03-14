#!/usr/bin/env sage
# exp_W7h.sage — 2段階スクリーニング: cycle type × Burau trace
# 戦略比較: ランダム / グローバルBurau / 2段階 / オラクル

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from sklearn.cluster import KMeans
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms

print("=" * 65)
print("  W-7h: 2段階スクリーニング設計の検証")
print("  Stage1: cycle type 分類  Stage2: 型内 Burau ランキング")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7h")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 数値セットアップ（W-7g と同一 seed）
# ══════════════════════════════════════════════════════════════════
rng = np.random.default_rng(int(42))
d   = 24
A_dyn  = rng.standard_normal((d,d))*0.25;  A_dyn = A_dyn - A_dyn.T
b_bias = rng.standard_normal(d)*0.1
a_vec  = np.zeros(d);  a_vec[:d//4] = 1.0
M_mat  = rng.standard_normal((d,d))*0.3
M_mat  = M_mat / (np.linalg.norm(M_mat,2)+1e-8)
bias_list = [rng.standard_normal(d)*(0.2+0.05*i) for i in range(4)]
theta_star = np.array([1.0, 1.0])
p0_master  = rng.standard_normal(d)*0.5
N_ROUNDS = 2;  N_GRID = 25
th1 = np.linspace(-1.0, 3.0, N_GRID)
th2 = np.linspace(-1.0, 3.0, N_GRID)

def time_ev(p): return np.tanh(A_dyn@p+b_bias)*0.5+p*0.4
def make_ops(K):
    return [
        lambda p,th,_=None: p+th[0]*a_vec,
        lambda p,th,_=None: (np.eye(d)+th[1]*M_mat)@p,
        lambda p,th,_=None: np.tanh(p*1.2)*0.9,
        lambda p,th,_=None: np.tanh(p*0.8+bias_list[0])*0.7,
        lambda p,th,_=None: p*0.6+np.tanh(bias_list[1])*0.4,
        lambda p,th,_=None: np.tanh(p+bias_list[2])*0.85,
    ][:K]

def compute_dist(order, K):
    ops = make_ops(K);  p0 = p0_master.copy()
    ref = list(range(K))
    def traj(ord_, th):
        p = p0.copy();  t = [p.copy()]
        for _ in range(N_ROUNDS):
            for idx in ord_:
                p = time_ev(p);  p = ops[idx](p,th);  t.append(p.copy())
        return np.array(t)
    zt = traj(ref, theta_star)
    def J(ord_, th): return float(np.mean(np.sum((traj(ord_,th)-zt)**2,axis=1)))
    grid = np.array([[J(order,[th1[j],th2[i]]) for j in range(N_GRID)]
                     for i in range(N_GRID)])
    ref_g= np.array([[J(ref,  [th1[j],th2[i]]) for j in range(N_GRID)]
                     for i in range(N_GRID)])
    return float(np.sqrt(np.mean((grid.ravel()-ref_g.ravel())**2)))

# ══════════════════════════════════════════════════════════════════
# データ構築: K=4, K=6
# ══════════════════════════════════════════════════════════════════
ALL = {}
for K in [4, 6]:
    print(f"\n[K={K}] 全 {factorial(K)} 件のデータ構築中...")
    BK = BraidGroup(K);  gK = list(BK.generators())
    rows = []
    for perm in iter_perms(range(1, K+1)):
        sp = SagePerm(list(perm))
        b  = BK.one()
        for i in sp.reduced_word(): b = b * gK[i-1]
        btr = float(b.burau_matrix()(t=QQ(1)/QQ(2)).trace())
        ct  = str(list(sp.cycle_type()))
        dist= compute_dist([x-1 for x in perm], K)
        rows.append({"perm": list(perm), "label": "".join(str(x) for x in perm),
                     "burau": btr, "writhe": len(sp.reduced_word()),
                     "dist": dist, "ct": ct})
    ALL[K] = rows
    dists = [r["dist"] for r in rows]
    print(f"  完了。dist: [{min(dists):.3f}, {max(dists):.3f}]  "
          f"mean={np.mean(dists):.3f}")

# ══════════════════════════════════════════════════════════════════
# 2段階モデルの構築
# ══════════════════════════════════════════════════════════════════
def build_twostage_model(rows):
    """
    各 cycle type について:
      - within-type Pearson r を計算
      - r の符号から「高い Burau が良いか低い Burau が良いか」を決定
    戻り値: dict{ct: {"r": r, "direction": +1 or -1, "n": n, "mean_dist": mean}}
    """
    ct_groups = defaultdict(list)
    for r in rows: ct_groups[r["ct"]].append(r)
    model = {}
    for ct, grp in ct_groups.items():
        bv = np.array([r["burau"] for r in grp])
        dv = np.array([r["dist"]  for r in grp])
        if len(grp) >= 3:
            rr, pp = stats.pearsonr(bv, dv)
        else:
            rr, pp = 0.0, 1.0
        # r < 0: 高 Burau → 低 dist → 高い順に選ぶ (direction=+1)
        # r > 0: 高 Burau → 高 dist → 低い順に選ぶ (direction=-1)
        direction = -1 if rr > 0 else +1
        model[ct] = {"r": rr, "p": pp, "direction": direction,
                     "n": len(grp), "mean_dist": float(dv.mean())}
    return model, ct_groups

# ══════════════════════════════════════════════════════════════════
# スクリーニング戦略の定義
# ══════════════════════════════════════════════════════════════════
def strategy_random(rows, N, seed=0):
    idx = np.random.default_rng(int(seed)).choice(len(rows), N, replace=False)
    return [rows[i] for i in idx]

def strategy_global_burau(rows, N):
    """グローバル Burau 降順（高い = 単純 = 良いはず）"""
    return sorted(rows, key=lambda r: -r["burau"])[:N]

def strategy_twostage(rows, N, model, ct_groups):
    """
    Stage1: cycle type 別に候補を Burau でランキング（方向を type ごとに調整）
    Stage2: 各 type から cycle_type_size / total * N 件を比例配分で選択
    """
    total = len(rows)
    selected = []
    for ct, grp in ct_groups.items():
        n_select = max(1, round(N * len(grp) / total))
        direction = model[ct]["direction"]
        # direction=+1: 高 Burau 優先, direction=-1: 低 Burau 優先
        ranked = sorted(grp, key=lambda r: direction * r["burau"], reverse=True)
        selected.extend(ranked[:n_select])
    # 超過分を除去（mean_dist が小さい型から優先して保持）
    if len(selected) > N:
        selected = sorted(selected, key=lambda r: r["dist"])[:N]
    # 不足分を補填（残りをランダムに）
    elif len(selected) < N:
        selected_labels = {r["label"] for r in selected}
        remaining = [r for r in rows if r["label"] not in selected_labels]
        np.random.default_rng(int(99)).shuffle(remaining)
        selected.extend(remaining[:N - len(selected)])
    return selected[:N]

def strategy_oracle(rows, N):
    """真の上位 N 件（実際には使えないオラクル）"""
    return sorted(rows, key=lambda r: r["dist"])[:N]

def evaluate(selected):
    dists = np.array([r["dist"] for r in selected])
    return {"mean": float(dists.mean()), "std": float(dists.std()),
            "max": float(dists.max()), "min": float(dists.min())}

# ══════════════════════════════════════════════════════════════════
# 主要実験: 各 N で 4 戦略を比較
# ══════════════════════════════════════════════════════════════════
print("\n[メイン比較] 各 N でのスクリーニング性能")
print("=" * 65)

results_by_K = {}
for K in [4, 6]:
    rows = ALL[K]
    model, ct_groups = build_twostage_model(rows)
    total = len(rows)

    print(f"\n  ── K={K} ({total} 通り) ──")
    print(f"\n  [2段階モデル: cycle type ごとの方向]")
    for ct, info in sorted(model.items(), key=lambda x: x[1]["mean_dist"]):
        sign = "↑高Burau優先" if info["direction"]==+1 else "↓低Burau優先"
        print(f"    {ct:<25} n={info['n']:3d}  "
              f"mean_dist={info['mean_dist']:.3f}  "
              f"within_r={info['r']:+.3f}  {sign}")

    N_values = [3, 5, 10, 20, 30] if K == 4 else [10, 20, 50, 100, 150]
    N_values = [n for n in N_values if n < total]

    perf = defaultdict(list)
    print(f"\n  N_sel  | Random       | GlobalBurau  | 2-Stage      | Oracle")
    print(f"  " + "-"*62)
    for N in N_values:
        # 30回ランダムの平均
        rand_means = [evaluate(strategy_random(rows, N, seed=s))["mean"]
                      for s in range(30)]
        r_m = np.mean(rand_means);  r_s = np.std(rand_means)
        gb = evaluate(strategy_global_burau(rows, N))
        ts = evaluate(strategy_twostage(rows, N, model, ct_groups))
        oc = evaluate(strategy_oracle(rows, N))
        print(f"  N={N:<4} | {r_m:.3f}±{r_s:.3f} | "
              f"{gb['mean']:.3f}±{gb['std']:.3f} | "
              f"{ts['mean']:.3f}±{ts['std']:.3f} | "
              f"{oc['mean']:.3f}±{oc['std']:.3f}")
        perf["random_mean"].append(r_m);  perf["random_std"].append(r_s)
        perf["global_mean"].append(gb["mean"])
        perf["twostage_mean"].append(ts["mean"])
        perf["oracle_mean"].append(oc["mean"])
        perf["N"].append(N)

    results_by_K[K] = {"perf": perf, "model": model,
                       "ct_groups": ct_groups, "rows": rows}

    # Precision@N (何件本当の上位 N を取れたか)
    true_top = {r["label"] for r in strategy_oracle(rows, max(N_values))}
    print(f"\n  [Precision@N — 真の上位N件のうち何件を取れたか]")
    print(f"  N_sel  | GlobalBurau  | 2-Stage")
    print(f"  " + "-"*35)
    for N in N_values:
        true_topN = {r["label"] for r in strategy_oracle(rows, N)}
        gb_sel  = {r["label"] for r in strategy_global_burau(rows, N)}
        ts_sel  = {r["label"] for r in
                   strategy_twostage(rows, N, model, ct_groups)}
        p_gb = float(len(true_topN & gb_sel)) / float(N)
        p_ts = float(len(true_topN & ts_sel)) / float(N)
        print(f"  N={N:<4} | {p_gb:.3f}         | {p_ts:.3f}")

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

# ── 図1: K=6 の 4 戦略パフォーマンス比較 ───────────────────────
K = 6
rows   = ALL[K]
model, ct_groups = build_twostage_model(rows)
perf   = results_by_K[K]["perf"]
N_vals = perf["N"]
all_dists = np.array([r["dist"] for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"W-7h: 2-Stage Screening vs Alternatives  (K={K}, n=720)", fontsize=12)

ax = axes[0]
ax.plot(N_vals, perf["random_mean"],   "o--", color="gray",      lw=2, ms=8, label="Random")
ax.plot(N_vals, perf["global_mean"],   "s-",  color="steelblue", lw=2, ms=8, label="Global Burau")
ax.plot(N_vals, perf["twostage_mean"], "^-",  color="tomato",    lw=2.5, ms=10, label="2-Stage (ct+Burau)")
ax.plot(N_vals, perf["oracle_mean"],   "D--", color="green",     lw=1.5, ms=8, label="Oracle (upper bound)")
ax.axhline(all_dists.mean(), color="black", lw=1, linestyle=":", alpha=0.5,
           label=f"pop mean={all_dists.mean():.2f}")
ax.set_xlabel("N selected", fontsize=11);  ax.set_ylabel("Mean landscape distance", fontsize=11)
ax.set_title("Mean distance of selected N orderings\n(lower = better screening)", fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

# 改善率
ax2 = axes[1]
rand_arr = np.array(perf["random_mean"])
gb_arr   = np.array(perf["global_mean"])
ts_arr   = np.array(perf["twostage_mean"])
oc_arr   = np.array(perf["oracle_mean"])
gb_imp   = (rand_arr - gb_arr) / rand_arr * 100
ts_imp   = (rand_arr - ts_arr) / rand_arr * 100
oc_imp   = (rand_arr - oc_arr) / rand_arr * 100

ax2.plot(N_vals, gb_imp, "s-",  color="steelblue", lw=2, ms=8, label="Global Burau")
ax2.plot(N_vals, ts_imp, "^-",  color="tomato",    lw=2.5, ms=10, label="2-Stage")
ax2.plot(N_vals, oc_imp, "D--", color="green",     lw=1.5, ms=8, label="Oracle")
ax2.axhline(0, color="black", lw=1, alpha=0.5)
ax2.set_xlabel("N selected", fontsize=11)
ax2.set_ylabel("Improvement over random (%)", fontsize=11)
ax2.set_title("% improvement in mean dist vs random\n(higher = better)", fontsize=10)
ax2.legend(fontsize=9);  ax2.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7h_fig1_strategy_comparison.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig1 saved: {p}")

# ── 図2: K=6 — cycle type 別の選択可視化 ──────────────────────
N_demo = 50
rows6    = ALL[6]
model6, ct_groups6 = build_twostage_model(rows6)
gb_sel50  = set(r["label"] for r in strategy_global_burau(rows6, N_demo))
ts_sel50  = set(r["label"] for r in strategy_twostage(rows6, N_demo, model6, ct_groups6))
oc_sel50  = set(r["label"] for r in strategy_oracle(rows6, N_demo))

burau6 = np.array([r["burau"] for r in rows6])
dist6  = np.array([r["dist"]  for r in rows6])
types6 = [r["ct"] for r in rows6]
unique_cts6 = sorted(set(types6))
ct_cmap = {ct: plt.cm.tab20(i/len(unique_cts6)) for i,ct in enumerate(unique_cts6)}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"W-7h: N={N_demo} selected orderings (K=6, n=720)", fontsize=12)

panels = [
    ("GlobalBurau", gb_sel50, "steelblue"),
    ("2-Stage",     ts_sel50, "tomato"),
    ("Oracle",      oc_sel50, "green"),
]
for ax, (name, sel, hi_color) in zip(axes, panels):
    for ct in unique_cts6:
        mask = np.array(types6) == ct
        c = ct_cmap[ct]
        ax.scatter(burau6[mask], dist6[mask], s=12, alpha=0.25, color=c, zorder=2)
    # 選択された点を強調
    sel_mask = np.array([r["label"] in sel for r in rows6])
    ax.scatter(burau6[sel_mask], dist6[sel_mask], s=60, color=hi_color,
               edgecolors="black", lw=0.8, zorder=5, alpha=0.9)
    mean_sel = dist6[sel_mask].mean() if sel_mask.sum() > 0 else 0
    ax.set_xlabel("Burau trace");  ax.set_ylabel("Landscape distance")
    ax.set_title(f"{name}\nmean dist of selected = {mean_sel:.3f}", fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7h_fig2_selection_viz.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig2 saved: {p}")

# ── 図3: cycle type 別の「方向付き Burau」の有効性 ─────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle("W-7h: Within-cycle-type Burau screening (K=6)", fontsize=12)

ct_sorted_by_n = sorted(ct_groups6.items(), key=lambda x: -len(x[1]))[:6]
for ax, (ct, grp) in zip(axes.flat, ct_sorted_by_n):
    bv = np.array([r["burau"] for r in grp])
    dv = np.array([r["dist"]  for r in grp])
    rr = model6[ct]["r"]
    direction = model6[ct]["direction"]

    ax.scatter(bv, dv, s=40, alpha=0.7,
               color=ct_cmap.get(ct, "gray"), edgecolors="black", lw=0.3)
    # 回帰直線
    if len(grp) >= 3:
        sl, ic, *_ = stats.linregress(bv, dv)
        xs = np.linspace(bv.min(), bv.max(), 50)
        ax.plot(xs, sl*xs+ic, "r--", lw=1.5)
    # 選択される側を矢印で示す
    dir_text = "Select HIGH Burau" if direction==+1 else "Select LOW Burau"
    dir_color= "steelblue"         if direction==+1 else "tomato"
    ax.set_xlabel("Burau trace", fontsize=9);  ax.set_ylabel("Landscape dist", fontsize=9)
    ax.set_title(f"cycle type = {ct}\nn={len(grp)},  r={rr:+.3f}\n{dir_text}",
                 fontsize=9, color=dir_color)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7h_fig3_within_ct.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig3 saved: {p}")

# ══════════════════════════════════════════════════════════════════
# サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  総合サマリー")
print("="*65)

for K in [4, 6]:
    rows  = ALL[K]
    perf  = results_by_K[K]["perf"]
    model = results_by_K[K]["model"]
    all_d = np.array([r["dist"] for r in rows])

    print(f"\n  ── K={K} ({len(rows)} 通り) ──")
    print(f"  全体平均距離: {all_d.mean():.4f}")

    # N=20 の比較
    N_target = 20 if K == 6 else 5
    if N_target in perf["N"]:
        idx = perf["N"].index(N_target)
        print(f"\n  N={N_target} のスクリーニング比較:")
        print(f"    Random:       mean={perf['random_mean'][idx]:.4f}")
        print(f"    GlobalBurau:  mean={perf['global_mean'][idx]:.4f}  "
              f"(+{(perf['random_mean'][idx]-perf['global_mean'][idx])/perf['random_mean'][idx]*100:.1f}%改善)")
        print(f"    2-Stage:      mean={perf['twostage_mean'][idx]:.4f}  "
              f"(+{(perf['random_mean'][idx]-perf['twostage_mean'][idx])/perf['random_mean'][idx]*100:.1f}%改善)")
        print(f"    Oracle:       mean={perf['oracle_mean'][idx]:.4f}  "
              f"(+{(perf['random_mean'][idx]-perf['oracle_mean'][idx])/perf['random_mean'][idx]*100:.1f}%改善)")

    # 2段階の肝: 方向が逆転する type
    reversed_cts = [(ct, info) for ct, info in model.items() if info["r"] > 0.05]
    if reversed_cts:
        print(f"\n  ⚠ GlobalBurauと逆方向の cycle type:")
        for ct, info in sorted(reversed_cts, key=lambda x: -x[1]["r"]):
            print(f"    {ct}: r={info['r']:+.3f}  n={info['n']}  "
                  f"→ LOW Burau が良い（グローバルと逆）")

print(f"""
=================================================================
  W-7h 実験完了  出力先: {OUT}
=================================================================
""")
