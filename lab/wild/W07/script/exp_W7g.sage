#!/usr/bin/env sage
# exp_W7g.sage — 非線形・クラスター構造の検証
# 問い: 相関の低下は「偽相関」「環状構造」「クラスター構造」のどれか？

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms

print("=" * 65)
print("  W-7g: 非線形・環状・クラスター構造の検証")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7g")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 数値セットアップ（W-7f と同一）
# ══════════════════════════════════════════════════════════════════
rng = np.random.default_rng(int(42))
d   = 24
A_dyn  = rng.standard_normal((d, d)) * 0.25;  A_dyn = A_dyn - A_dyn.T
b_bias = rng.standard_normal(d) * 0.1
a_vec  = np.zeros(d);  a_vec[:d//4] = 1.0
M_mat  = rng.standard_normal((d, d)) * 0.3
M_mat  = M_mat / (np.linalg.norm(M_mat, 2) + 1e-8)
bias_list = [rng.standard_normal(d) * (0.2 + 0.05*i) for i in range(4)]
theta_star = np.array([1.0, 1.0])
p0_master  = rng.standard_normal(d) * 0.5

def time_evolution(p):
    return np.tanh(A_dyn @ p + b_bias) * 0.5 + p * 0.4

def make_ops(K):
    return [
        lambda p,th,_=None: p + th[0]*a_vec,
        lambda p,th,_=None: (np.eye(d)+th[1]*M_mat)@p,
        lambda p,th,_=None: np.tanh(p*1.2)*0.9,
        lambda p,th,_=None: np.tanh(p*0.8+bias_list[0])*0.7,
        lambda p,th,_=None: p*0.6+np.tanh(bias_list[1])*0.4,
        lambda p,th,_=None: np.tanh(p+bias_list[2])*0.85,
    ][:K]

N_ROUNDS = 2;  N_GRID = 25
th1 = np.linspace(-1.0, 3.0, N_GRID)
th2 = np.linspace(-1.0, 3.0, N_GRID)
TH1, TH2 = np.meshgrid(th1, th2)

def compute_J_grid(order, K):
    ops  = make_ops(K)
    p0   = p0_master.copy()
    ref  = list(range(K))
    # 基準軌道
    def traj(ord_, th):
        p = p0.copy(); t = [p.copy()]
        for _ in range(N_ROUNDS):
            for idx in ord_:
                p = time_evolution(p); p = ops[idx](p,th); t.append(p.copy())
        return np.array(t)
    zt = traj(ref, theta_star)
    def J(ord_, th):
        tr = traj(ord_, th)
        return float(np.mean(np.sum((tr-zt)**2, axis=1)))
    grid = np.array([[J(order,[th1[j],th2[i]]) for j in range(N_GRID)]
                     for i in range(N_GRID)])
    ref_g = np.array([[J(ref,[th1[j],th2[i]]) for j in range(N_GRID)]
                      for i in range(N_GRID)])
    return float(np.sqrt(np.mean((grid.ravel()-ref_g.ravel())**2)))

def perm_to_braid_burau(perm, BK, gK):
    sp = SagePerm(list(perm))
    b  = BK.one()
    for i in sp.reduced_word(): b = b * gK[i-1]
    btr = float(b.burau_matrix()(t=QQ(1)/QQ(2)).trace())
    return b, btr, len(sp.reduced_word())

def cycle_type_str(perm):
    """1-indexed permutation リストの cycle type を文字列で返す"""
    ct = SagePerm(list(perm)).cycle_type()
    return str(list(ct))  # e.g. "[4]", "[2, 1, 1]"

# ══════════════════════════════════════════════════════════════════
# K=4 と K=6 のデータを構築
# ══════════════════════════════════════════════════════════════════
data = {}
for K in [4, 6]:
    print(f"\n[K={K}] データ構築中 ({factorial(K)} 件)...")
    BK = BraidGroup(K);  gK = list(BK.generators())
    rows = []
    for perm in iter_perms(range(1, K+1)):
        b, btr, writhe = perm_to_braid_burau(perm, BK, gK)
        ct  = cycle_type_str(perm)
        dist = compute_J_grid([x-1 for x in perm], K)
        rows.append({"perm": list(perm), "burau": btr,
                     "writhe": writhe, "dist": dist, "cycle_type": ct})
    data[K] = rows
    print(f"  完了。dist: [{min(r['dist'] for r in rows):.3f}, "
          f"{max(r['dist'] for r in rows):.3f}]")

# ══════════════════════════════════════════════════════════════════
# 分析 1: Pearson vs Spearman — 非線形性の確認
# ══════════════════════════════════════════════════════════════════
print("\n[分析1] Pearson vs Spearman 相関")
print("-" * 55)
for K in [4, 6]:
    rows = data[K]
    burau = np.array([r["burau"]  for r in rows])
    dist  = np.array([r["dist"]   for r in rows])
    rp, pp = stats.pearsonr(burau, dist)
    rs, ps = spearmanr(burau, dist)
    print(f"  K={K}: Pearson r={rp:+.4f}(p={pp:.2e})  "
          f"Spearman ρ={rs:+.4f}(p={ps:.2e})  "
          f"差={abs(rp)-abs(rs):+.4f}")

# ══════════════════════════════════════════════════════════════════
# 分析 2: Cycle type（置換の型）別の分布
# ══════════════════════════════════════════════════════════════════
print("\n[分析2] Cycle type 別の相関")
print("-" * 55)
for K in [4, 6]:
    rows = data[K]
    from collections import defaultdict
    ct_groups = defaultdict(list)
    for r in rows:
        ct_groups[r["cycle_type"]].append(r)
    print(f"  K={K}:")
    ct_corrs = {}
    for ct, grp in sorted(ct_groups.items(), key=lambda x: -len(x[1])):
        bv = np.array([r["burau"] for r in grp])
        dv = np.array([r["dist"]  for r in grp])
        if len(grp) >= 3:
            rr, pp = stats.pearsonr(bv, dv)
            print(f"    cycle_type={ct:<18} n={len(grp):3d}  "
                  f"mean_dist={dv.mean():.3f}  r={rr:+.3f}(p={pp:.2e})")
            ct_corrs[ct] = (rr, pp, dv.mean(), len(grp))
        else:
            print(f"    cycle_type={ct:<18} n={len(grp):3d}  "
                  f"mean_dist={dv.mean():.3f}  (n<3, skip)")
    data[K + 100] = ct_corrs  # save for plotting

# ══════════════════════════════════════════════════════════════════
# 分析 3: 環状構造 — K-cycle の冪乗列をトレース
# ══════════════════════════════════════════════════════════════════
print("\n[分析3] K-cycle の冪乗列（環状構造の検証）")
print("-" * 55)
cyclic_data = {}
for K in [4, 6]:
    BK = BraidGroup(K);  gK = list(BK.generators())
    # K-cycle: (1→2→3→...→K→1)
    gen_cycle = list(range(2, K+1)) + [1]   # [2,3,...,K,1]
    powers = []
    perm = list(range(1, K+1))  # identity
    for power in range(K+1):
        b, btr, writhe = perm_to_braid_burau(perm, BK, gK)
        dist = compute_J_grid([x-1 for x in perm], K)
        ct = cycle_type_str(perm)
        label = "".join(str(x) for x in perm)
        powers.append({"power": power, "perm": perm[:], "burau": btr,
                       "writhe": writhe, "dist": dist,
                       "cycle_type": ct, "label": label})
        # 次の冪乗: perm = gen_cycle ∘ perm
        perm = [gen_cycle[perm[i]-1] for i in range(K)]
    cyclic_data[K] = powers
    print(f"  K={K} ({K}-cycle の冪乗):")
    for p in powers:
        print(f"    g^{p['power']}={p['label']:<10} "
              f"type={p['cycle_type']:<18} "
              f"Burau={p['burau']:.3f}  dist={p['dist']:.3f}")

# ══════════════════════════════════════════════════════════════════
# 分析 4: K-means クラスタリング
# ══════════════════════════════════════════════════════════════════
print("\n[分析4] K-means クラスタリング")
print("-" * 55)
kmeans_data = {}
for K in [4, 6]:
    rows  = data[K]
    burau = np.array([r["burau"]  for r in rows])
    dist  = np.array([r["dist"]   for r in rows])
    X = StandardScaler().fit_transform(np.column_stack([burau, dist]))
    # クラスタ数 2〜5 で within-cluster r を比較
    best_k, best_r = 1, abs(stats.pearsonr(burau, dist)[0])
    for nk in range(2, 6):
        km   = KMeans(n_clusters=nk, random_state=int(42), n_init=10)
        labs = km.fit_predict(X)
        within_rs = []
        for c in range(nk):
            mask = labs == c
            if mask.sum() >= 3:
                rr, _ = stats.pearsonr(burau[mask], dist[mask])
                within_rs.append(abs(rr))
        avg_r = np.mean(within_rs) if within_rs else 0
        print(f"  K={K}, clusters={nk}: mean within-cluster |r| = {avg_r:.4f}")
        if avg_r > best_r:
            best_r = avg_r;  best_k = nk
    print(f"  → K={K} の最良クラスタ数: {best_k}  (within |r|={best_r:.4f})")
    # 最良クラスタでラベルを保存
    km_best = KMeans(n_clusters=best_k, random_state=int(42), n_init=10)
    kmeans_data[K] = km_best.fit_predict(X)

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

# ── 図1: K=4 の全構造を 4 パネルで ─────────────────────────────
rows4   = data[4]
burau4  = np.array([r["burau"] for r in rows4])
dist4   = np.array([r["dist"]  for r in rows4])
types4  = [r["cycle_type"] for r in rows4]
unique_types4 = sorted(set(types4))
color_map4 = {ct: plt.cm.tab10(i/len(unique_types4))
              for i, ct in enumerate(unique_types4)}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("W-7g: Non-linear / Cluster Structure Analysis (K=4)", fontsize=13)

# Panel A: cycle type 色分け散布図
ax = axes[0, 0]
for ct in unique_types4:
    mask = np.array(types4) == ct
    ax.scatter(burau4[mask], dist4[mask], s=60, alpha=0.8,
               color=color_map4[ct], label=f"{ct}(n={mask.sum()})", zorder=5)
ax.set_xlabel("Burau trace");  ax.set_ylabel("Landscape distance")
ax.set_title("A: Colored by cycle type", fontsize=10)
ax.legend(fontsize=7, loc="upper right");  ax.grid(True, alpha=0.3)

# Panel B: 残差プロット（非線形性確認）
ax = axes[0, 1]
sl, ic, *_ = stats.linregress(burau4, dist4)
residuals  = dist4 - (sl * burau4 + ic)
ax.scatter(burau4, residuals, c=[list(unique_types4).index(t) for t in types4],
           cmap="tab10", s=60, alpha=0.8)
ax.axhline(0, color="red", lw=1.5, linestyle="--")
ax.set_xlabel("Burau trace");  ax.set_ylabel("Residual from linear fit")
ax.set_title("B: Residuals from linear fit\n(pattern = non-linearity)", fontsize=10)
ax.grid(True, alpha=0.3)

# Panel C: K-means クラスタリング
ax = axes[1, 0]
km_labels4 = kmeans_data[4]
scatter = ax.scatter(burau4, dist4, c=km_labels4, cmap="Set1", s=60, alpha=0.8)
ax.set_xlabel("Burau trace");  ax.set_ylabel("Landscape distance")
ax.set_title("C: K-means clustering", fontsize=10)
plt.colorbar(scatter, ax=ax, label="cluster")
ax.grid(True, alpha=0.3)

# Panel D: 環状構造 — K-cycle の冪乗列
ax = axes[1, 1]
cp4 = cyclic_data[4]
bv  = [p["burau"] for p in cp4]
dv  = [p["dist"]  for p in cp4]
lv  = [p["label"] for p in cp4]
ax.plot(bv, dv, "o-", ms=12, lw=2, color="steelblue", zorder=5)
ax.plot(bv[0], dv[0], "*", ms=20, color="red", zorder=10, label="e (identity)")
for i, (b_, d_, l_) in enumerate(zip(bv, dv, lv)):
    ax.annotate(f"g^{i}\n{l_}", (b_, d_),
                textcoords="offset points", xytext=(8, 4), fontsize=8)
ax.set_xlabel("Burau trace");  ax.set_ylabel("Landscape distance")
ax.set_title(f"D: 4-cycle powers g^0...g^4\n(環状軌跡の確認)", fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7g_fig1_K4_structure.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig1 saved: {p}")

# ── 図2: K=6 の大規模版（720点）──────────────────────────────────
rows6   = data[6]
burau6  = np.array([r["burau"] for r in rows6])
dist6   = np.array([r["dist"]  for r in rows6])
types6  = [r["cycle_type"] for r in rows6]
unique_types6 = sorted(set(types6))
ct_to_idx6 = {ct: i for i, ct in enumerate(unique_types6)}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("W-7g: Non-linear / Cluster Structure Analysis (K=6, n=720)", fontsize=13)

# Panel A: cycle type 色分け
ax = axes[0, 0]
scatter = ax.scatter(burau6, dist6,
                     c=[ct_to_idx6[t] for t in types6],
                     cmap="tab20", s=8, alpha=0.5)
ax.set_xlabel("Burau trace");  ax.set_ylabel("Landscape distance")
ax.set_title("A: Colored by cycle type (K=6)", fontsize=10)
ax.grid(True, alpha=0.3)
# 凡例（cycle type ごとの平均距離）
ct_corrs6 = data[6 + 100]
legend_items = [(ct, ct_corrs6[ct][2]) for ct in unique_types6 if ct in ct_corrs6]
legend_items.sort(key=lambda x: x[1])
for ct, md in legend_items[:5]:
    ax.plot([], [], "o", color=plt.cm.tab20(ct_to_idx6[ct]/len(unique_types6)),
            label=f"{ct} mean={md:.2f}", ms=5)
ax.legend(fontsize=6, loc="upper right")

# Panel B: cycle type ごとの平均距離（バーチャート）
ax = axes[0, 1]
from collections import defaultdict
ct_mean = defaultdict(list)
for r in rows6:
    ct_mean[r["cycle_type"]].append(r["dist"])
ct_sorted = sorted(ct_mean.items(), key=lambda x: np.mean(x[1]))
cts6  = [x[0] for x in ct_sorted]
means = [np.mean(x[1]) for x in ct_sorted]
stds  = [np.std(x[1]) for x in ct_sorted]
ns    = [len(x[1]) for x in ct_sorted]
colors_bar = [plt.cm.tab20(ct_to_idx6.get(ct, 0)/len(unique_types6)) for ct in cts6]
ax.barh(range(len(cts6)), means, xerr=stds, color=colors_bar, alpha=0.8,
        edgecolor="black", lw=0.5)
ax.set_yticks(range(len(cts6)))
ax.set_yticklabels([f"{ct}(n={n})" for ct, n in zip(cts6, ns)], fontsize=7)
ax.set_xlabel("Mean landscape distance ± std")
ax.set_title("B: Distance by cycle type\n(sorted by mean)", fontsize=10)
ax.axvline(np.mean(dist6), color="red", linestyle="--", lw=1.5,
           label=f"overall mean={np.mean(dist6):.2f}")
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis="x")

# Panel C: K-means クラスタリング
ax = axes[1, 0]
km_labels6 = kmeans_data[6]
scatter = ax.scatter(burau6, dist6, c=km_labels6, cmap="Set1",
                     s=8, alpha=0.5)
ax.set_xlabel("Burau trace");  ax.set_ylabel("Landscape distance")
ax.set_title("C: K-means clustering (K=6)", fontsize=10)
plt.colorbar(scatter, ax=ax, label="cluster")
ax.grid(True, alpha=0.3)

# Panel D: 環状構造 — 6-cycle の冪乗列
ax = axes[1, 1]
cp6 = cyclic_data[6]
bv6 = [p["burau"] for p in cp6]
dv6 = [p["dist"]  for p in cp6]
lv6 = [f"g^{p['power']}" for p in cp6]
ax.plot(bv6, dv6, "o-", ms=10, lw=2, color="tomato", zorder=5)
ax.plot(bv6[0], dv6[0], "*", ms=20, color="navy", zorder=10, label="e (identity)")
for i, (b_, d_, l_) in enumerate(zip(bv6, dv6, lv6)):
    ax.annotate(f"{l_}\n{cp6[i]['cycle_type']}", (b_, d_),
                textcoords="offset points", xytext=(6, 3), fontsize=7)
ax.set_xlabel("Burau trace");  ax.set_ylabel("Landscape distance")
ax.set_title("D: 6-cycle powers g^0...g^6\n(環状軌跡の確認)", fontsize=10)
ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

plt.tight_layout()
p = OUT / "W7g_fig2_K6_structure.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig2 saved: {p}")

# ── 図3: cycle type 内の相関まとめ ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("W-7g: Within-cycle-type correlation (Burau vs distance)", fontsize=12)

for ax, K in zip(axes, [4, 6]):
    rows  = data[K]
    ct_groups = defaultdict(list)
    for r in rows:
        ct_groups[r["cycle_type"]].append(r)
    cts  = sorted(ct_groups.keys(), key=lambda x: -len(ct_groups[x]))
    within_rs, ns_ct, mds = [], [], []
    for ct in cts:
        grp = ct_groups[ct]
        bv_ = np.array([r["burau"] for r in grp])
        dv_ = np.array([r["dist"]  for r in grp])
        if len(grp) >= 3:
            rr, _ = stats.pearsonr(bv_, dv_)
            within_rs.append(rr)
        else:
            within_rs.append(np.nan)
        ns_ct.append(len(grp))
        mds.append(np.mean(dv_))
    global_r, _ = stats.pearsonr(
        np.array([r["burau"] for r in rows]),
        np.array([r["dist"]  for r in rows])
    )
    colors_ = [plt.cm.RdBu_r(0.5 + r_/2) if not np.isnan(r_) else "gray"
               for r_ in within_rs]
    bars = ax.bar(range(len(cts)), within_rs, color=colors_, alpha=0.8,
                  edgecolor="black", lw=0.5)
    ax.axhline(global_r, color="navy", lw=2, linestyle="--",
               label=f"global r={global_r:.3f}")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels([f"{ct}\n(n={n})" for ct, n in zip(cts, ns_ct)],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Within-cluster Pearson r")
    ax.set_title(f"K={K}: within cycle-type r\nvs global r={global_r:.3f}", fontsize=10)
    ax.set_ylim(-1, 1);  ax.legend(fontsize=9);  ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
p = OUT / "W7g_fig3_within_cycle_r.png"
fig.savefig(str(p), dpi=150, bbox_inches="tight");  plt.close()
print(f"  fig3 saved: {p}")

# ══════════════════════════════════════════════════════════════════
# サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  総合サマリー")
print("="*65)

for K in [4, 6]:
    rows  = data[K]
    burau = np.array([r["burau"] for r in rows])
    dist  = np.array([r["dist"]  for r in rows])
    rp, _ = stats.pearsonr(burau, dist)
    rs, _ = spearmanr(burau, dist)
    print(f"\n  K={K}:")
    print(f"    Pearson r  = {rp:+.4f}")
    print(f"    Spearman ρ = {rs:+.4f}")
    print(f"    Pearson - Spearman = {rp-rs:+.4f}  "
          f"({'非線形成分あり' if abs(rp-rs) > 0.05 else '線形で十分'})")

    # 環状構造: K-cycle の冪乗が (Burau, dist) 空間で経路を描くか
    cp = cyclic_data[K]
    bv = [p["burau"] for p in cp]
    dv = [p["dist"]  for p in cp]
    print(f"    {K}-cycle 冪乗の Burau 軌跡: {[f'{v:.2f}' for v in bv]}")
    print(f"    {K}-cycle 冪乗の dist 軌跡:  {[f'{v:.2f}' for v in dv]}")
    turns = sum(1 for i in range(1,len(bv)-1)
                if (bv[i]-bv[i-1])*(bv[i+1]-bv[i]) < 0)
    print(f"    Burau 軌跡の向き転換回数: {turns}  "
          f"({'環状/非単調' if turns > 0 else '単調'})")

    # Cycle type 別の within-r 平均
    ct_groups = defaultdict(list)
    for r in rows: ct_groups[r["cycle_type"]].append(r)
    within_rs_all = []
    for ct, grp in ct_groups.items():
        if len(grp) >= 3:
            bv_ = [r["burau"] for r in grp]
            dv_ = [r["dist"]  for r in grp]
            rr, _ = stats.pearsonr(bv_, dv_)
            within_rs_all.append(abs(rr))
    if within_rs_all:
        print(f"    cycle type 内の平均|r| = {np.mean(within_rs_all):.4f}  "
              f"(global |r| = {abs(rp):.4f})")
        if np.mean(within_rs_all) > abs(rp):
            print("    → cycle type 内では相関が強い ✓ クラスター構造あり")
        else:
            print("    → cycle type で層別しても相関は改善しない")

print(f"""
=================================================================
  W-7g 実験完了  出力先: {OUT}
=================================================================
""")
