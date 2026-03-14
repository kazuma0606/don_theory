#!/usr/bin/env sage
# exp_W7-2b.sage — W-7-2 Phase 2
# Burau(1/2) 固有値スペクトル（K-1 個）による多変量解析
# Phase 1 との比較: trace（1次元）→ 固有値ベクトル（5次元）
# モデル: OLS (statsmodels) + LassoCV + RidgeCV
# 評価:  5-fold CV R²,  Precision@N,  VIF 診断,  PCA 直交化
#
# 仮説:
#   trace は固有値の和 = 1次の対称関数
#   固有値ベクトルは高次の対称関数を含み，より多くの情報を持つ
#   → R² が 0.29（Phase 1 上限）を有意に超えれば Burau 非忠実性が天井の原因でない
#   → 超えなければ Burau 表現自体の情報量限界 → Phase 3 (LKB) へ

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import eigvals as scipy_eigvals
from collections import defaultdict
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except ImportError:
    print("  [警告] statsmodels 未インストール。pip install statsmodels")
    HAS_SM = False

print("=" * 65)
print("  W-7-2b: Burau 固有値スペクトル × 多変量回帰 (Phase 2)")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/W07/results/W7-2b")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 数値セットアップ（共通 seed）
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
# データ構築
# ══════════════════════════════════════════════════════════════════
K = 6
print(f"\n[K={K}] 全 {factorial(K)} 件のデータ構築中...")
print(f"  特徴量: Burau(1/2) 固有値 {K-1} 個（実部・虚部）+ 対称関数")
BK = BraidGroup(K);  gK = list(BK.generators())

rows = []
for perm_iter in iter_perms(range(1, K+1)):
    sp    = SagePerm(list(perm_iter))
    b     = BK.one()
    for i in sp.reduced_word(): b = b * gK[i-1]

    # Burau 行列を t=1/2 で数値行列として取得
    bm_sage = b.burau_matrix()(t=QQ(1)/QQ(2))
    # numpy 行列に変換
    bm_np = np.array([[float(bm_sage[r][c])
                       for c in range(bm_sage.ncols())]
                      for r in range(bm_sage.nrows())], dtype=np.float64)

    # 固有値（複素数）
    eigs = np.linalg.eigvals(bm_np)
    eigs_sorted = sorted(eigs, key=lambda z: (abs(z), z.real))

    # 実部・虚部（K×K 行列なので K 個の固有値）
    eig_re = np.array([z.real for z in eigs_sorted])
    eig_im = np.array([z.imag for z in eigs_sorted])
    eig_abs = np.array([abs(z) for z in eigs_sorted])

    # 対称関数（Elementary Symmetric Polynomials）
    # e1=Σλ_i=trace, e2=Σ_{i<j}λ_iλ_j, ..., eK=Πλ_i=det
    trace_val   = float(bm_np.trace())               # e1（= Phase1 の Burau trace）
    det_val     = float(np.linalg.det(bm_np))        # eK
    frob_norm   = float(np.linalg.norm(bm_np, 'fro'))# フロベニウスノルム
    spec_radius = float(max(abs(z) for z in eigs))   # スペクトル半径

    # 固有値の分散・歪度（分布の形状）
    eig_abs_std  = float(np.std(eig_abs))
    eig_re_std   = float(np.std(eig_re))

    order = [x-1 for x in perm_iter]
    ct    = list(sp.cycle_type())

    rows.append({
        "label"      : "".join(str(x+1) for x in order),
        "order"      : order,
        "trace"      : trace_val,
        "det"        : det_val,
        "frob"       : frob_norm,
        "spec_rad"   : spec_radius,
        "eig_re"     : eig_re,       # (K,)
        "eig_im"     : eig_im,       # (K,)
        "eig_abs"    : eig_abs,      # (K,)
        "eig_abs_std": eig_abs_std,
        "eig_re_std" : eig_re_std,
        "cayley"     : K - len(ct),
        "ct_str"     : str(ct),
        "dist"       : compute_dist(order, K),
    })

n_rows = len(rows)
dists  = np.array([r["dist"] for r in rows])
print(f"  完了。dist: [{dists.min():.3f}, {dists.max():.3f}]  mean={dists.mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# 特徴行列の組み立て
# ══════════════════════════════════════════════════════════════════
# A: Burau trace のみ（Phase1 ベースライン）
X_A = np.array([[r["trace"]] for r in rows])

# B: 固有値絶対値 K個（実対称関数的情報）
X_B = np.array([r["eig_abs"] for r in rows])                     # (720, K)

# C: 固有値実部 + 虚部（複素構造を含む）
X_C = np.hstack([np.array([r["eig_re"] for r in rows]),
                 np.array([r["eig_im"] for r in rows])])          # (720, 2K)

# D: スカラー特徴（trace, det, frob, spec_rad, eig_abs_std）
X_D = np.array([[r["trace"], r["det"], r["frob"],
                 r["spec_rad"], r["eig_abs_std"], r["eig_re_std"]]
                for r in rows])                                    # (720, 6)

# E: 固有値絶対値 + スカラー特徴 + Cayley（総合）
X_E = np.hstack([X_B,
                 np.array([[r["det"], r["frob"],
                            r["eig_abs_std"], r["cayley"]]
                           for r in rows])])                      # (720, K+4)

y = dists

feature_sets = {
    "A: trace のみ（Phase1 base）"  : (X_A, ["trace"]),
    "B: 固有値絶対値 K個"            : (X_B, [f"|λ{i}|" for i in range(K)]),
    "C: 固有値実部+虚部 2K個"        : (X_C, [f"Re(λ{i})" for i in range(K)] +
                                             [f"Im(λ{i})" for i in range(K)]),
    "D: スカラー特徴量 6個"          : (X_D, ["trace","det","frob",
                                              "spec_rad","eig_std","re_std"]),
    "E: 固有値+スカラー+Cayley（総合）": (X_E, [f"|λ{i}|" for i in range(K)] +
                                              ["det","frob","eig_std","cayley"]),
}

# PCA 直交化版（多重共線性対策）
# B の固有値に PCA を適用して直交成分に変換
pca_b = PCA(n_components=min(K, 5))
X_B_pca = pca_b.fit_transform(StandardScaler().fit_transform(X_B))
feature_sets["F: 固有値PCA（直交化）"] = (
    X_B_pca, [f"PC{i+1}({pca_b.explained_variance_ratio_[i]*100:.1f}%)"
              for i in range(X_B_pca.shape[1])])

# ══════════════════════════════════════════════════════════════════
# 5-fold CV 設定
# ══════════════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=int(42))

# ══════════════════════════════════════════════════════════════════
# ① 単変量相関：各スカラー特徴量
# ══════════════════════════════════════════════════════════════════
print("\n[① スカラー特徴量の単変量 Pearson r（Phase1 との比較）]")
scalar_feats = [
    ("trace（Phase1）",  [r["trace"]       for r in rows]),
    ("det（行列式）",    [r["det"]         for r in rows]),
    ("frob（Frobノルム）",[r["frob"]       for r in rows]),
    ("spec_radius",      [r["spec_rad"]    for r in rows]),
    ("|λ|の標準偏差",    [r["eig_abs_std"] for r in rows]),
    ("Cayley距離",       [r["cayley"]      for r in rows]),
]
print(f"  {'特徴量':<22}  {'r':>8}  {'p値':>12}  {'R²':>8}")
print("  " + "-"*55)
for name, vals in scalar_feats:
    r_val, p_val = stats.pearsonr(vals, dists)
    print(f"  {name:<22}  {r_val:+8.4f}  {p_val:12.3e}  {r_val**2:8.4f}")

# 固有値絶対値の各成分の相関
print(f"\n  [固有値絶対値 |λ_i| の単変量相関（i=0..{K-1}）]")
for i in range(K):
    vals_i = X_B[:, i]
    r_val, p_val = stats.pearsonr(vals_i, dists)
    print(f"  |λ{i}|  r={r_val:+.4f}  p={p_val:.3e}  R²={r_val**2:.4f}")

# ══════════════════════════════════════════════════════════════════
# ② OLS 重回帰（主要特徴セット）
# ══════════════════════════════════════════════════════════════════
if HAS_SM:
    print("\n[② OLS 重回帰分析]")
    for name in ["D: スカラー特徴量 6個",
                 "F: 固有値PCA（直交化）",
                 "E: 固有値+スカラー+Cayley（総合）"]:
        X, feat_names = feature_sets[name]
        print(f"\n  ── {name} ──")
        X_sc  = StandardScaler().fit_transform(X)
        X_ols = sm.add_constant(X_sc)

        # 完全多重共線性の場合は pinv で対処
        try:
            model = sm.OLS(y, X_ols).fit()
        except Exception as e:
            print(f"  [OLS error: {e}] → Ridge のみ表示")
            continue

        print(f"  {'変数':<18}  {'係数':>9}  {'t値':>8}  {'p値':>10}  有意")
        print("  " + "-"*58)
        for j, cname in enumerate(["const"] + list(feat_names)):
            coef = model.params[j]
            tval = model.tvalues[j]
            pval = model.pvalues[j]
            sig  = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else ""))
            print(f"  {cname:<18}  {coef:9.4f}  {tval:8.3f}  {pval:10.4e}  {sig}")
        print(f"  R²={model.rsquared:.4f}  Adj.R²={model.rsquared_adj:.4f}"
              f"  F={model.fvalue:.2f}  F_p={model.f_pvalue:.3e}")

        # VIF（PCA 版は直交なのでスキップ）
        if "PCA" not in name and X.shape[1] >= 2:
            vifs = []
            try:
                vifs = [variance_inflation_factor(X_sc, j)
                        for j in range(X_sc.shape[1])]
            except Exception:
                pass
            if vifs:
                print(f"  [VIF] ", end="")
                for fname, vif_val in zip(feat_names, vifs):
                    flag = "⚠" if vif_val > 10 else ""
                    print(f"{fname}={vif_val:.1f}{flag}  ", end="")
                print()

# ══════════════════════════════════════════════════════════════════
# ③ 5-fold CV R²
# ══════════════════════════════════════════════════════════════════
print("\n[③ 5-fold CV R²]")
print(f"  {'特徴量セット':<36}  {'OLS':>7}  {'Ridge':>7}  {'Lasso':>7}")
print("  " + "-"*62)

cv_r2_results = {}
baseline_r2   = None
for name, (X, feat_names) in feature_sets.items():
    pipe_ols   = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    pipe_ridge = Pipeline([("sc", StandardScaler()),
                           ("m", RidgeCV(alphas=[0.01,0.1,1,10,100], cv=kf))])
    pipe_lasso = Pipeline([("sc", StandardScaler()),
                           ("m", LassoCV(cv=kf, random_state=int(42),
                                         max_iter=5000))])

    r2_ols   = float(np.mean(cross_val_score(pipe_ols,   X, y, cv=kf, scoring="r2")))
    r2_ridge = float(np.mean(cross_val_score(pipe_ridge, X, y, cv=kf, scoring="r2")))
    r2_lasso = float(np.mean(cross_val_score(pipe_lasso, X, y, cv=kf, scoring="r2")))

    cv_r2_results[name] = {"ols": r2_ols, "ridge": r2_ridge, "lasso": r2_lasso}

    if baseline_r2 is None:
        baseline_r2 = r2_ridge
        tag = "  ← Phase1 基準"
    else:
        delta = r2_ridge - baseline_r2
        tag = f"  (Δridge={delta:+.4f})"
        if delta > 0.02:
            tag += " ★"
    print(f"  {name:<36}  {r2_ols:7.4f}  {r2_ridge:7.4f}  {r2_lasso:7.4f}{tag}")

# ══════════════════════════════════════════════════════════════════
# ④ 5-fold CV Precision@N
# ══════════════════════════════════════════════════════════════════
N_vals = [10, 20, 50, 100]

def oracle_labels(rows_sub, N):
    N = int(min(N, len(rows_sub)))
    return {r["label"] for r in sorted(rows_sub, key=lambda r: r["dist"])[:N]}

def precision_at(sel, true):
    return float(len(sel & true)) / float(len(true)) if true else 0.0

print("\n[④ 5-fold CV Precision@N (RidgeCV)]")
hdr = f"  {'特徴量セット':<36}|" + "".join(f"  @{N:<4}|" for N in N_vals)
print(hdr)
print("  " + "-"*(36 + 9*len(N_vals) + 3))

prec_results = {}
for name, (X, feat_names) in feature_sets.items():
    prec_by_N = {N: [] for N in N_vals}
    for tr_idx, te_idx in kf.split(X):
        rows_te = [rows[i] for i in te_idx]
        sc      = StandardScaler().fit(X[tr_idx])
        X_tr_sc = sc.transform(X[tr_idx])
        X_te_sc = sc.transform(X[te_idx])
        reg     = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)
        reg.fit(X_tr_sc, y[tr_idx])
        y_pred  = reg.predict(X_te_sc)
        for N in N_vals:
            n        = int(min(N, len(rows_te)))
            true_top = oracle_labels(rows_te, n)
            ranked   = np.argsort(y_pred)[:n]
            sel      = {rows_te[i]["label"] for i in ranked}
            prec_by_N[N].append(precision_at(sel, true_top))
    prec_results[name] = {N: float(np.mean(v)) for N, v in prec_by_N.items()}
    line = f"  {name:<36}|" + "".join(f"  {prec_results[name][N]:.3f}|" for N in N_vals)
    print(line)

print(f"\n  参考（W-7 既知値）:")
print(f"  {'GlobalBurau(1/2)':<36}|  0.600|  0.600|  0.608|  —   |")
print(f"  {'3-Stage-25%':<36}|  —    |  —    |  0.868|  —   |")
print(f"  {'SRM+GF_M30':<36}|  —    |  0.900|  0.852|  —   |")

# ══════════════════════════════════════════════════════════════════
# ⑤ PCA 寄与率と各 PC の dist との相関
# ══════════════════════════════════════════════════════════════════
print("\n[⑤ PCA 分析（固有値絶対値の直交分解）]")
print(f"  累積寄与率: {np.cumsum(pca_b.explained_variance_ratio_).round(4)}")
print(f"\n  各主成分と dist の相関:")
for i in range(X_B_pca.shape[1]):
    r_val, p_val = stats.pearsonr(X_B_pca[:, i], dists)
    print(f"  PC{i+1} ({pca_b.explained_variance_ratio_[i]*100:.1f}%): "
          f"r={r_val:+.4f}  p={p_val:.3e}  R²={r_val**2:.4f}")

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("W-7-2b: Burau Eigenvalue Spectrum Analysis (K=6)", fontsize=13)

# ── 図1: 固有値絶対値 |λ_i| vs dist（全6成分）──────────────
for i in range(min(K, 6)):
    ax = axes[i // 3][i % 3]
    x_i = X_B[:, i]
    r_i, p_i = stats.pearsonr(x_i, dists)
    ax.scatter(x_i, dists, alpha=0.12, s=7, color="steelblue")
    m, b_ = np.polyfit(x_i, dists, 1)
    xs = np.linspace(x_i.min(), x_i.max(), 100)
    ax.plot(xs, m*xs+b_, "r-", lw=1.5, alpha=0.8)
    ax.set_xlabel(f"|λ{i}|")
    ax.set_ylabel("dist")
    ax.set_title(f"|λ{i}|: r={r_i:+.4f}, R²={r_i**2:.4f}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "W7-2b_fig1_eigenval_scatter.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig1 saved")

# ── 図2: CV R² 比較 + Precision@50 ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("W-7-2b: CV R² and Precision@N by Feature Set", fontsize=12)

ax = axes[0]
names_short = ["A:trace", "B:|λ|×6", "C:Re+Im", "D:scalar", "E:総合", "F:PCA"]
ridge_r2s   = [cv_r2_results[n]["ridge"] for n in feature_sets]
colors_bar  = ["#2196F3" if i==0 else "#4CAF50" if r2>0.30 else "#FF9800"
               for i, r2 in enumerate(ridge_r2s)]
bars = ax.bar(range(len(ridge_r2s)), ridge_r2s, color=colors_bar, alpha=0.85)
ax.axhline(0.2719, ls="--", color="blue",  alpha=0.6, lw=1.5, label="Phase1 base (0.272)")
ax.axhline(0.2870, ls="--", color="orange",alpha=0.6, lw=1.5, label="Phase1 best (0.287)")
for i, v in enumerate(ridge_r2s):
    ax.text(i, v+0.002, f"{v:.4f}", ha="center", fontsize=8)
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("5-fold CV R² (RidgeCV)")
ax.set_title("CV R² comparison")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(ridge_r2s)*1.2 + 0.02)

ax = axes[1]
colors_p50  = ["#2196F3" if i==0 else "#4CAF50" for i in range(len(feature_sets))]
p50_vals    = [prec_results[n][50] for n in feature_sets]
ax.bar(range(len(p50_vals)), p50_vals, color=colors_p50, alpha=0.85)
ax.axhline(0.608, ls="--", color="blue",  alpha=0.5, lw=1.5, label="GlobalBurau P@50=0.608")
ax.axhline(0.868, ls="--", color="green", alpha=0.5, lw=1.5, label="3-Stage P@50=0.868")
for i, v in enumerate(p50_vals):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(names_short)))
ax.set_xticklabels(names_short, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Precision@50 (5-fold CV, RidgeCV)")
ax.set_title("Precision@50 comparison")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(OUT / "W7-2b_fig2_r2_precision.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig2 saved")

# ── 図3: PC 散布図（PC1 vs PC2 に dist を色でプロット）──────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("W-7-2b: PCA of Eigenvalue Spectrum — dist colored", fontsize=12)

ax = axes[0]
sc_ = ax.scatter(X_B_pca[:, 0], X_B_pca[:, 1],
                 c=dists, cmap="RdYlBu_r", alpha=0.5, s=10)
plt.colorbar(sc_, ax=ax, label="dist")
ax.set_xlabel(f"PC1 ({pca_b.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_b.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PC1 vs PC2 (color=dist)")
ax.grid(True, alpha=0.3)

ax = axes[1]
# PC 別の dist 相関バーチャート
pc_r2s = []
for i in range(X_B_pca.shape[1]):
    r_val, _ = stats.pearsonr(X_B_pca[:, i], dists)
    pc_r2s.append(r_val**2)
pc_labels = [f"PC{i+1}\n({pca_b.explained_variance_ratio_[i]*100:.1f}%)"
             for i in range(len(pc_r2s))]
ax.bar(range(len(pc_r2s)), pc_r2s,
       color=["#E53935" if r > 0.1 else "#BDBDBD" for r in pc_r2s], alpha=0.85)
ax.set_xticks(range(len(pc_labels)))
ax.set_xticklabels(pc_labels, fontsize=9)
ax.set_ylabel("R² with dist")
ax.set_title("Each PC's individual R² with dist")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(OUT / "W7-2b_fig3_pca.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig3 saved")

# ══════════════════════════════════════════════════════════════════
# 総合サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  W-7-2b 総合サマリー")
print("=" * 65)

best_name = max(cv_r2_results, key=lambda k: cv_r2_results[k]["ridge"])
best_r2   = cv_r2_results[best_name]["ridge"]
phase1_r2 = 0.287   # Phase1 best (Burau 5点 + Cayley)
phase1_base = 0.272 # Phase1 baseline (trace only)

print(f"\n  Phase1 ベースライン（trace のみ）: CV R²={phase1_base:.4f}")
print(f"  Phase1 最良（Burau 5点+Cayley）:   CV R²={phase1_r2:.4f}")
print(f"  Phase2 最良（{best_name}）: CV R²={best_r2:.4f}")

delta_from_base = best_r2 - phase1_base
delta_from_p1   = best_r2 - phase1_r2

print(f"\n  Phase1 base からの改善:  {delta_from_base:+.4f}")
print(f"  Phase1 best からの改善:  {delta_from_p1:+.4f}")

if delta_from_p1 > 0.02:
    print(f"\n  ★★ 固有値スペクトルが有意な追加情報を持つ！")
    print(f"     → Phase3（LKB 忠実表現）でさらに改善できる可能性あり")
elif delta_from_p1 > 0.005:
    print(f"\n  ★  わずかな改善あり（誤差範囲の可能性）")
    print(f"     → 多重共線性を PCA で除去した際の結果に注目")
else:
    print(f"\n  →  固有値スペクトルは trace に対して追加情報をほとんど持たない")
    print(f"     → Burau 表現の情報量限界が確認された")
    print(f"     → Phase3（LKB 忠実表現）で Burau 非忠実性が原因か最終判定へ")

# Precision 結果
best_p_name = max(prec_results, key=lambda k: prec_results[k][50])
best_p50    = prec_results[best_p_name][50]
print(f"\n  Precision@50: 最良 {best_p_name} → {best_p50:.4f}")
print(f"  3-Stage-25% 参考値: 0.8680  差={best_p50-0.868:+.4f}")

print(f"\n  出力先: {OUT}")
print("=" * 65)
