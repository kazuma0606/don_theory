#!/usr/bin/env sage
# exp_W7-2c.sage — W-7-2 Phase 3
# Lawrence-Krammer-Bigelow (LKB) 忠実表現による多変量解析
#
# LKB は B_K の忠実表現（K≥1 で核が自明）。
# Burau は K≥5 で非忠実 → 同じ Burau 行列を持つ異なる組み紐が存在。
# Phase 3 の問い：
#   「R²≈0.28 の天井は Burau の非忠実性が原因か？」
#   → LKB で R² が有意に上昇すれば：非忠実性が情報ボトルネック
#   → 上昇しなければ：dist の 72% は代数不変量では原理的に予測不可能
#
# 特徴量:
#   G: LKB trace（(x,y) を複数点で評価）
#   H: LKB 固有値スペクトル（15個）
#   I: Burau trace + LKB trace 複合
#   J: LKB trace + SRM（W-7m で有効だった標準表現）
#
# モデル: OLS + LassoCV + RidgeCV
# 評価:  5-fold CV R²,  Precision@N（N=10,20,50,100）

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
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
    print("  [警告] statsmodels 未インストール")
    HAS_SM = False

print("=" * 65)
print("  W-7-2c: LKB 忠実表現 × 多変量回帰 (Phase 3)")
print("  問い: Burau の非忠実性が R²≈0.28 の天井の原因か？")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/W07/results/W7-2c")
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

# 標準表現行列（W-7m で P@50=0.844 を達成）
def standard_rep_matrix(perm_0idx, K):
    M = np.zeros((K-1, K-1))
    sig = perm_0idx;  sK1 = sig[K-1]
    for j in range(K-1):
        sj = sig[j]
        if sj  < K-1: M[sj,  j] += 1.0
        if sK1 < K-1: M[sK1, j] -= 1.0
    return M

# ══════════════════════════════════════════════════════════════════
# LKB 評価点
# (x,y) の組み合わせ: x=q²（group-like parameter），y=t（dilation）
# 慣例的な選択 + 対称な点を使用
# ══════════════════════════════════════════════════════════════════
LKB_EVALS = [
    (QQ(1)/QQ(2), QQ(1)/QQ(2)),   # (x=1/2, y=1/2) — 基本点
    (QQ(1)/QQ(3), QQ(1)/QQ(2)),   # (x=1/3, y=1/2)
    (QQ(2)/QQ(3), QQ(1)/QQ(2)),   # (x=2/3, y=1/2)
    (QQ(1)/QQ(2), QQ(1)/QQ(3)),   # (x=1/2, y=1/3)
    (QQ(1)/QQ(2), QQ(2)/QQ(3)),   # (x=1/2, y=2/3)
]
LKB_LABELS = ["(1/2,1/2)", "(1/3,1/2)", "(2/3,1/2)", "(1/2,1/3)", "(1/2,2/3)"]
LKB_DIM = 15  # K*(K-1)//2 for K=6

# ══════════════════════════════════════════════════════════════════
# データ構築
# ══════════════════════════════════════════════════════════════════
K = 6
print(f"\n[K={K}] 全 {factorial(K)} 件のデータ構築中...")
print(f"  LKB 行列サイズ: {LKB_DIM}×{LKB_DIM}（= K(K-1)/2 = {LKB_DIM}）")
print(f"  評価点 (x,y): {LKB_LABELS}")
BK = BraidGroup(K);  gK = list(BK.generators())

rows = []
for idx_p, perm_iter in enumerate(iter_perms(range(1, K+1))):
    if idx_p % 100 == 0:
        print(f"  {idx_p}/720...", end="\r", flush=True)

    sp    = SagePerm(list(perm_iter))
    b     = BK.one()
    for i in sp.reduced_word(): b = b * gK[i-1]

    # ── Burau trace（Phase1/2 ベースライン）──
    bm_burau = b.burau_matrix()(t=QQ(1)/QQ(2))
    burau_tr = float(bm_burau.trace())

    # ── LKB 行列（シンボリック）を一度だけ取得 ──
    lkb_sym = b.LKB_matrix()
    R_lkb   = lkb_sym.base_ring()
    x_var, y_var = R_lkb.gens()

    # 各評価点での LKB trace & 固有値絶対値ベクトル
    lkb_traces = []
    lkb_eig_abs_all = []  # 評価点(0)の固有値絶対値を保存

    for ei, (xv, yv) in enumerate(LKB_EVALS):
        lkb_eval = lkb_sym.subs({x_var: xv, y_var: yv})
        lkb_np   = np.array([[float(lkb_eval[r][c])
                              for c in range(lkb_eval.ncols())]
                             for r in range(lkb_eval.nrows())], dtype=np.float64)
        lkb_traces.append(float(lkb_np.trace()))
        if ei == 0:
            eigs = np.linalg.eigvals(lkb_np)
            lkb_eig_abs_all = sorted([abs(z) for z in eigs])  # 15個

    # ── LKB スカラー特徴量（評価点0: x=y=1/2）──
    lkb_eval0  = lkb_sym.subs({x_var: QQ(1)/QQ(2), y_var: QQ(1)/QQ(2)})
    lkb_np0    = np.array([[float(lkb_eval0[r][c])
                            for c in range(lkb_eval0.ncols())]
                           for r in range(lkb_eval0.nrows())], dtype=np.float64)
    lkb_frob   = float(np.linalg.norm(lkb_np0, 'fro'))
    lkb_specr  = float(max(lkb_eig_abs_all))
    lkb_eig_std= float(np.std(lkb_eig_abs_all))

    # ── 標準表現行列 (SRM) ──
    order = [x-1 for x in perm_iter]
    srm   = standard_rep_matrix(order, K).ravel()

    ct = list(sp.cycle_type())
    rows.append({
        "label"        : "".join(str(x+1) for x in order),
        "order"        : order,
        "burau_tr"     : burau_tr,
        "lkb_traces"   : lkb_traces,        # 5点評価
        "lkb_eig_abs"  : lkb_eig_abs_all,   # 15個の固有値絶対値
        "lkb_frob"     : lkb_frob,
        "lkb_specr"    : lkb_specr,
        "lkb_eig_std"  : lkb_eig_std,
        "srm"          : srm,               # 25次元
        "cayley"       : K - len(ct),
        "ct_str"       : str(ct),
        "dist"         : compute_dist(order, K),
    })

print()
n_rows = len(rows)
dists  = np.array([r["dist"] for r in rows])
print(f"  完了。dist: [{dists.min():.3f}, {dists.max():.3f}]  mean={dists.mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# 特徴行列の組み立て
# ══════════════════════════════════════════════════════════════════
# Phase 1/2 ベースライン
X_base = np.array([[r["burau_tr"]] for r in rows])                      # (720,1)

# G: LKB trace 1点（基本点 x=y=1/2）
X_G = np.array([[r["lkb_traces"][0]] for r in rows])                    # (720,1)

# H: LKB trace 5点
X_H = np.array([r["lkb_traces"] for r in rows])                         # (720,5)

# I: LKB 固有値絶対値 15個
X_I = np.array([r["lkb_eig_abs"] for r in rows])                        # (720,15)

# J: LKB スカラー特徴量（trace5点 + frob + specr + eig_std）
X_J = np.hstack([
    np.array([r["lkb_traces"] for r in rows]),
    np.array([[r["lkb_frob"], r["lkb_specr"], r["lkb_eig_std"]] for r in rows])
])                                                                        # (720,8)

# K_set: Burau trace + LKB trace(5点) 複合
X_K = np.hstack([X_base, X_H])                                           # (720,6)

# L: LKB trace(5点) + SRM（W-7m 最強特徴 25次元）
X_L = np.hstack([X_H, np.array([r["srm"] for r in rows])])               # (720,30)

# M: Burau + LKB trace + SRM（全情報統合）
X_M = np.hstack([X_base, X_H, np.array([r["srm"] for r in rows])])       # (720,31)

y = dists

feature_sets = {
    "base: Burau trace（Phase1 基準）"   : (X_base, ["burau"]),
    "G: LKB trace（x=y=1/2）"           : (X_G,    ["lkb_tr_0"]),
    "H: LKB trace 5点"                  : (X_H,    LKB_LABELS),
    "I: LKB 固有値絶対値 15個"           : (X_I,    [f"|μ{i}|" for i in range(LKB_DIM)]),
    "J: LKB スカラー 8個"               : (X_J,    LKB_LABELS + ["frob","specr","eig_std"]),
    "K: Burau + LKB trace 6個"          : (X_K,    ["burau"]+LKB_LABELS),
    "L: LKB trace + SRM 30個"           : (X_L,    LKB_LABELS + [f"srm{i}" for i in range(25)]),
    "M: Burau + LKB + SRM 31個（統合）" : (X_M,    ["burau"]+LKB_LABELS+[f"srm{i}" for i in range(25)]),
}

# ══════════════════════════════════════════════════════════════════
# 5-fold CV 設定
# ══════════════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=int(42))

# ══════════════════════════════════════════════════════════════════
# ① 単変量相関：LKB trace 各評価点
# ══════════════════════════════════════════════════════════════════
print("\n[① 単変量 Pearson r — LKB trace vs Burau trace の比較]")
print(f"  {'特徴量':<28}  {'r':>8}  {'p値':>12}  {'R²':>8}")
print("  " + "-"*62)

# Burau ベースライン
r_bur, p_bur = stats.pearsonr([r["burau_tr"] for r in rows], dists)
print(f"  {'Burau trace（Phase1基準）':<28}  {r_bur:+8.4f}  {p_bur:12.3e}  {r_bur**2:8.4f}")
print()

for i, lbl in enumerate(LKB_LABELS):
    vals = [r["lkb_traces"][i] for r in rows]
    r_v, p_v = stats.pearsonr(vals, dists)
    print(f"  LKB trace {lbl:<18}  {r_v:+8.4f}  {p_v:12.3e}  {r_v**2:8.4f}")

print()
r_lkb_frob, _ = stats.pearsonr([r["lkb_frob"]  for r in rows], dists)
r_lkb_spec, _ = stats.pearsonr([r["lkb_specr"] for r in rows], dists)
r_lkb_estd, _ = stats.pearsonr([r["lkb_eig_std"] for r in rows], dists)
print(f"  {'LKB Frobenius norm':<28}  {r_lkb_frob:+8.4f}  —             {r_lkb_frob**2:8.4f}")
print(f"  {'LKB spectral radius':<28}  {r_lkb_spec:+8.4f}  —             {r_lkb_spec**2:8.4f}")
print(f"  {'LKB |eigenvalue| std':<28}  {r_lkb_estd:+8.4f}  —             {r_lkb_estd**2:8.4f}")

# LKB 固有値絶対値の各成分
print(f"\n  [LKB 固有値絶対値 |μ_i| の単変量相関（上位5件）]")
eig_rs = []
for i in range(LKB_DIM):
    vals_i = X_I[:, i]
    r_i, p_i = stats.pearsonr(vals_i, dists)
    eig_rs.append((i, r_i, p_i))
eig_rs_sorted = sorted(eig_rs, key=lambda x: -abs(x[1]))
for i, r_i, p_i in eig_rs_sorted[:5]:
    print(f"  |μ{i:2d}|  r={r_i:+.4f}  p={p_i:.3e}  R²={r_i**2:.4f}")

# ══════════════════════════════════════════════════════════════════
# ② OLS 重回帰（代表的セット）
# ══════════════════════════════════════════════════════════════════
if HAS_SM:
    print("\n[② OLS 重回帰分析（代表的特徴セット）]")
    for name in ["G: LKB trace（x=y=1/2）",
                 "H: LKB trace 5点",
                 "K: Burau + LKB trace 6個"]:
        X, feat_names = feature_sets[name]
        print(f"\n  ── {name} ──")
        X_sc  = StandardScaler().fit_transform(X)
        X_ols = sm.add_constant(X_sc)
        try:
            model = sm.OLS(y, X_ols).fit()
        except Exception as e:
            print(f"  [error: {e}]"); continue

        print(f"  {'変数':<16}  {'係数':>9}  {'t値':>8}  {'p値':>10}  有意")
        print("  " + "-"*56)
        for j, cname in enumerate(["const"] + list(feat_names)):
            coef = model.params[j]; tval = model.tvalues[j]; pval = model.pvalues[j]
            sig  = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else ""))
            print(f"  {cname:<16}  {coef:9.4f}  {tval:8.3f}  {pval:10.4e}  {sig}")
        print(f"  R²={model.rsquared:.4f}  Adj.R²={model.rsquared_adj:.4f}"
              f"  F={model.fvalue:.2f}  F_p={model.f_pvalue:.3e}")

        if X.shape[1] >= 2:
            try:
                vifs = [variance_inflation_factor(X_sc, j) for j in range(X_sc.shape[1])]
                print(f"  VIF: ", end="")
                for fn, vf in zip(feat_names, vifs):
                    flag = "⚠" if vf > 10 else ""
                    print(f"{fn}={vf:.1f}{flag}  ", end="")
                print()
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════
# ③ 5-fold CV R²
# ══════════════════════════════════════════════════════════════════
print("\n[③ 5-fold CV R²]")
print(f"  {'特徴量セット':<40}  {'OLS':>7}  {'Ridge':>7}  {'Lasso':>7}")
print("  " + "-"*65)

PHASE1_BASE = 0.2719   # Burau trace 単独
PHASE1_BEST = 0.2870   # Burau 5点 + Cayley
PHASE2_BEST = 0.2744   # Phase2 最良
SRM_R2      = 0.2839   # W-7m SRM linear (参考)

cv_r2_results = {}
for name, (X, feat_names) in feature_sets.items():
    pipe_ols   = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    pipe_ridge = Pipeline([("sc", StandardScaler()),
                           ("m", RidgeCV(alphas=[0.01,0.1,1,10,100], cv=kf))])
    pipe_lasso = Pipeline([("sc", StandardScaler()),
                           ("m", LassoCV(cv=kf, random_state=int(42), max_iter=5000))])

    r2_ols   = float(np.mean(cross_val_score(pipe_ols,   X, y, cv=kf, scoring="r2")))
    r2_ridge = float(np.mean(cross_val_score(pipe_ridge, X, y, cv=kf, scoring="r2")))
    r2_lasso = float(np.mean(cross_val_score(pipe_lasso, X, y, cv=kf, scoring="r2")))
    cv_r2_results[name] = {"ols": r2_ols, "ridge": r2_ridge, "lasso": r2_lasso}

    delta = r2_ridge - PHASE1_BASE
    star  = " ★★★" if delta > 0.03 else (" ★★" if delta > 0.02 else (" ★" if delta > 0.01 else ""))
    print(f"  {name:<40}  {r2_ols:7.4f}  {r2_ridge:7.4f}  {r2_lasso:7.4f}  (Δ={delta:+.4f}){star}")

print(f"\n  参考（既知値）:")
print(f"  {'Phase1 base (Burau trace)':<40}  —       {PHASE1_BASE:.4f}  —")
print(f"  {'Phase1 best (Burau 5点+Cayley)':<40}  —       {PHASE1_BEST:.4f}  —")
print(f"  {'Phase2 best (scalar 6個)':<40}  —       {PHASE2_BEST:.4f}  —")
print(f"  {'W-7m SRM linear (参考)':<40}  —       {SRM_R2:.4f}  —")

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
hdr = f"  {'特徴量セット':<40}|" + "".join(f"  @{N:<4}|" for N in N_vals)
print(hdr)
print("  " + "-"*(40 + 9*len(N_vals) + 3))

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
    line = f"  {name:<40}|" + "".join(f"  {prec_results[name][N]:.3f}|" for N in N_vals)
    print(line)

print(f"\n  参考（W-7 既知値）:")
print(f"  {'GlobalBurau(1/2)':<40}|  0.600|  0.600|  0.608|  —   |")
print(f"  {'3-Stage-25%':<40}|  —    |  —    |  0.868|  —   |")
print(f"  {'SRM+GF_M30 (W-7n)':<40}|  —    |  0.900|  0.852|  —   |")

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

# ── 図1: LKB trace vs Burau trace（散布図 + 相関構造）────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("W-7-2c: LKB trace vs Burau trace vs dist (K=6)", fontsize=12)

ax = axes[0]
x_lkb = [r["lkb_traces"][0] for r in rows]
r_l, _ = stats.pearsonr(x_lkb, dists)
ax.scatter(x_lkb, dists, alpha=0.15, s=7, color="red")
m, b_ = np.polyfit(x_lkb, dists, 1)
xs = np.linspace(min(x_lkb), max(x_lkb), 100)
ax.plot(xs, m*xs+b_, "k-", lw=1.5)
ax.set_xlabel("LKB trace (x=y=1/2)")
ax.set_ylabel("dist")
ax.set_title(f"LKB trace vs dist\nr={r_l:+.4f}, R²={r_l**2:.4f}")
ax.grid(True, alpha=0.3)

ax = axes[1]
x_bur = [r["burau_tr"] for r in rows]
r_b, _ = stats.pearsonr(x_bur, dists)
ax.scatter(x_bur, dists, alpha=0.15, s=7, color="blue")
m, b_ = np.polyfit(x_bur, dists, 1)
xs = np.linspace(min(x_bur), max(x_bur), 100)
ax.plot(xs, m*xs+b_, "k-", lw=1.5)
ax.set_xlabel("Burau trace (t=1/2)")
ax.set_ylabel("dist")
ax.set_title(f"Burau trace vs dist\nr={r_b:+.4f}, R²={r_b**2:.4f}")
ax.grid(True, alpha=0.3)

ax = axes[2]
r_bl, _ = stats.pearsonr(x_bur, x_lkb)
ax.scatter(x_bur, x_lkb, c=dists, cmap="RdYlBu_r", alpha=0.4, s=8)
ax.set_xlabel("Burau trace"); ax.set_ylabel("LKB trace")
ax.set_title(f"Burau vs LKB trace\nr={r_bl:+.4f}\n(color=dist)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "W7-2c_fig1_lkb_vs_burau.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig1 saved")

# ── 図2: CV R² サマリー（全 Phase 横断比較）────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("W-7-2c: CV R² — Phase 1/2/3 Comprehensive Comparison", fontsize=12)

ax = axes[0]
all_r2_data = [
    ("P1:Burau",      PHASE1_BASE, "#BDBDBD"),
    ("P1:Burau5+Cay", PHASE1_BEST, "#9E9E9E"),
    ("P2:scalar6",    PHASE2_BEST, "#78909C"),
    ("G:LKB_tr1",  cv_r2_results["G: LKB trace（x=y=1/2）"]["ridge"],      "#EF5350"),
    ("H:LKB_tr5",  cv_r2_results["H: LKB trace 5点"]["ridge"],             "#E53935"),
    ("I:LKB_eig15",cv_r2_results["I: LKB 固有値絶対値 15個"]["ridge"],     "#C62828"),
    ("J:LKB_sc8",  cv_r2_results["J: LKB スカラー 8個"]["ridge"],          "#B71C1C"),
    ("K:Bur+LKB",  cv_r2_results["K: Burau + LKB trace 6個"]["ridge"],     "#FF6F00"),
    ("L:LKB+SRM",  cv_r2_results["L: LKB trace + SRM 30個"]["ridge"],      "#4CAF50"),
    ("M:Bur+LKB+SRM",cv_r2_results["M: Burau + LKB + SRM 31個（統合）"]["ridge"],"#1B5E20"),
]
labels_bar = [d[0] for d in all_r2_data]
r2_vals    = [d[1] for d in all_r2_data]
colors_bar = [d[2] for d in all_r2_data]
ax.bar(range(len(r2_vals)), r2_vals, color=colors_bar, alpha=0.85)
ax.axhline(PHASE1_BASE, ls="--", color="gray",  alpha=0.5, lw=1.2, label=f"P1 base={PHASE1_BASE:.3f}")
ax.axhline(PHASE1_BEST, ls=":",  color="gray",  alpha=0.5, lw=1.2, label=f"P1 best={PHASE1_BEST:.3f}")
ax.axhline(SRM_R2,      ls="-.", color="green", alpha=0.5, lw=1.2, label=f"SRM R²={SRM_R2:.3f}")
for i, v in enumerate(r2_vals):
    ax.text(i, v+0.003, f"{v:.3f}", ha="center", fontsize=7, rotation=45)
ax.set_xticks(range(len(labels_bar)))
ax.set_xticklabels(labels_bar, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("5-fold CV R² (RidgeCV)")
ax.set_title("All Phases: CV R²")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(r2_vals)*1.2 + 0.02)

ax = axes[1]
p50_vals = [prec_results[n][50] for n in feature_sets]
p50_names = [n.split(":")[0].strip() for n in feature_sets]
colors_p  = ["#9E9E9E"] + ["#EF5350","#E53935","#C62828","#B71C1C","#FF6F00","#4CAF50","#1B5E20"]
ax.bar(range(len(p50_vals)), p50_vals, color=colors_p[:len(p50_vals)], alpha=0.85)
ax.axhline(0.608, ls="--", color="blue",  alpha=0.5, lw=1.5, label="GlobalBurau P@50=0.608")
ax.axhline(0.868, ls="--", color="green", alpha=0.5, lw=1.5, label="3-Stage P@50=0.868")
ax.axhline(0.852, ls="-.", color="orange",alpha=0.5, lw=1.5, label="SRM+GF P@50=0.852")
for i, v in enumerate(p50_vals):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(p50_names)))
ax.set_xticklabels(p50_names, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Precision@50")
ax.set_title("Precision@50 comparison")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(OUT / "W7-2c_fig2_r2_prec_all.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig2 saved")

# ── 図3: LKB 固有値絶対値の分布（cycle type 別）────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("W-7-2c: LKB Eigenvalue Spectrum by Cycle Type", fontsize=12)

ax = axes[0]
ct_colors = {"[1, 1, 1, 1, 1, 1]":"green","[2, 1, 1, 1, 1]":"blue",
             "[3, 1, 1, 1]":"cyan","[6]":"red"}
for ct, color in ct_colors.items():
    grp_eigs = [np.mean(r["lkb_eig_abs"]) for r in rows if r["ct_str"]==ct]
    grp_dist = [r["dist"]                 for r in rows if r["ct_str"]==ct]
    if grp_eigs:
        ax.scatter(grp_eigs, grp_dist, color=color, alpha=0.5, s=12, label=ct)
ax.set_xlabel("Mean LKB |eigenvalue|"); ax.set_ylabel("dist")
ax.set_title("Mean LKB |eigenvalue| vs dist\n(by cycle type)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
# LKB vs Burau trace 相関をパネル比較
r2_phases = {
    "Burau(Phase1)":    r_bur**2,
    "LKB_tr_1pt":       stats.pearsonr([r["lkb_traces"][0] for r in rows], dists)[0]**2,
    "LKB_frob":         r_lkb_frob**2,
    "LKB_specr":        r_lkb_spec**2,
}
ax.bar(range(len(r2_phases)), list(r2_phases.values()),
       color=["#2196F3","#EF5350","#FF9800","#9C27B0"], alpha=0.85)
ax.set_xticks(range(len(r2_phases)))
ax.set_xticklabels(list(r2_phases.keys()), rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Single-feature R² (with dist)")
ax.set_title("Single-feature R²\nBurau vs LKB scalars")
for i, v in enumerate(r2_phases.values()):
    ax.text(i, v+0.002, f"{v:.4f}", ha="center", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(r2_phases.values())*1.3 + 0.02)

plt.tight_layout()
fig.savefig(OUT / "W7-2c_fig3_lkb_spectrum.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig3 saved")

# ══════════════════════════════════════════════════════════════════
# 総合サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  W-7-2c 総合サマリー（Phase 1–3 横断）")
print("=" * 65)

best_lkb_name = max((k for k in cv_r2_results if k != "base: Burau trace（Phase1 基準）"),
                    key=lambda k: cv_r2_results[k]["ridge"])
best_lkb_r2   = cv_r2_results[best_lkb_name]["ridge"]

print(f"\n  ─ R² 天井の推移 ─")
print(f"  Phase1 base (Burau trace 1点):    {PHASE1_BASE:.4f}")
print(f"  Phase1 best (Burau 5点+Cayley):   {PHASE1_BEST:.4f}")
print(f"  Phase2 best (固有値スカラー):     {PHASE2_BEST:.4f}")
print(f"  Phase3 best ({best_lkb_name}): {best_lkb_r2:.4f}")
print(f"  SRM linear (W-7m 参考):           {SRM_R2:.4f}")

delta_lkb_vs_p1 = best_lkb_r2 - PHASE1_BASE

print(f"\n  LKB 追加によるΔR² (vs Phase1 base): {delta_lkb_vs_p1:+.4f}")

if delta_lkb_vs_p1 > 0.03:
    print(f"\n  ★★★ LKB が R² を有意に改善！")
    print(f"       → Burau の非忠実性が情報ボトルネックであった")
    print(f"       → さらに高次の表現で天井を超えられる可能性あり")
elif delta_lkb_vs_p1 > 0.01:
    print(f"\n  ★   LKB に若干の追加情報あり（ただし小幅）")
    print(f"       → 非忠実性は部分的な原因")
elif delta_lkb_vs_p1 > 0.002:
    print(f"\n  →   LKB の改善はわずか（誤差範囲の可能性）")
    print(f"       → Burau と LKB がほぼ同じ情報を持つ")
    print(f"       → dist の残差 72% は代数不変量では捉えられない可能性が高い")
else:
    print(f"\n  →   LKB は Burau に対して追加情報なし")
    print(f"  ━━ 結論: R²≈0.28 の天井は代数的不変量の限界 ━━")
    print(f"     Burau の非忠実性は天井の原因ではない。")
    print(f"     dist の 72% は置換群の代数的構造では予測不可能な")
    print(f"     dynamical 変動（演算子の具体的な非線形性）に起因する。")

# Precision
best_p50_n  = max(prec_results, key=lambda k: prec_results[k][50])
best_p50_v  = prec_results[best_p50_n][50]
best_p20_n  = max(prec_results, key=lambda k: prec_results[k][20])
best_p20_v  = prec_results[best_p20_n][20]
print(f"\n  Precision 最良:")
print(f"    P@20: {best_p20_n} → {best_p20_v:.4f}  (W-7n SRM+GF: 0.900)")
print(f"    P@50: {best_p50_n} → {best_p50_v:.4f}  (3-Stage: 0.868)")

print(f"\n  出力先: {OUT}")
print("=" * 65)
