#!/usr/bin/env sage
# exp_W7n.sage — Fourier×3-Stage 統合 + Wavelet 全手法
# 統合A: 3-Stage フィルタ × 帯域別 Fourier ランキング
# 統合B: Fourier detrend → 残差に 3-Stage
# W1: Cayley グラフスペクトルウェーブレット (多スケール熱核)
# W2: コセット Haar ウェーブレット (order prefix 多解像度)
# W3: Ridge 正則化 Fourier (正則化強度 = スケール制御)
# W4: SRM + グラフ Fourier 複合特徴

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms, combinations
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import copy

print("=" * 65)
print("  W-7n: Fourier×3-Stage 統合 + Wavelet 全手法")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7n")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 数値セットアップ（同一 seed）
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

# 標準表現行列
def standard_rep_matrix(perm_0idx, K):
    M = np.zeros((K-1, K-1))
    sig = perm_0idx;  sK1 = sig[K-1]
    for j in range(K-1):
        sj = sig[j]
        if sj  < K-1: M[sj,  j] += 1.0
        if sK1 < K-1: M[sK1, j] -= 1.0
    return M

# ══════════════════════════════════════════════════════════════════
# データ構築
# ══════════════════════════════════════════════════════════════════
K = 6
print(f"\n[K={K}] 全 {factorial(K)} 件のデータ構築中...")
BK = BraidGroup(K);  gK = list(BK.generators())
rows = []
for perm_iter in iter_perms(range(1, K+1)):
    sp    = SagePerm(list(perm_iter))
    b     = BK.one()
    for i in sp.reduced_word(): b = b * gK[i-1]
    btr   = float(b.burau_matrix()(t=QQ(1)/QQ(2)).trace())
    ct    = list(sp.cycle_type())
    order = [x-1 for x in perm_iter]
    dist  = compute_dist(order, K)
    srm   = standard_rep_matrix(order, K).ravel()

    # 逆置換: pos of each intervention
    inv_perm = [0]*K
    for pos, ei in enumerate(order): inv_perm[ei] = pos

    rows.append({
        "label"    : "".join(str(x+1) for x in order),
        "order"    : order,
        "burau"    : btr,
        "cayley"   : K - len(ct),
        "ct_str"   : str(ct),
        "pos_E1"   : int(inv_perm[0]),
        "first_op" : int(order[0]),
        "srm"      : srm,
        "dist"     : dist,
    })

n_rows = len(rows)
dists  = np.array([r["dist"] for r in rows])
print(f"  完了。dist: [{dists.min():.3f}, {dists.max():.3f}]  mean={dists.mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# Cayley グラフ & スペクトル分解（全体で一度だけ計算）
# ══════════════════════════════════════════════════════════════════
print("\n[Cayley グラフ構築 & ラプラシアン固有分解]")
perm_to_idx = {tuple(r["order"]): i for i, r in enumerate(rows)}
transpositions = list(combinations(range(K), 2))  # C(6,2)=15

A_cayley = np.zeros((n_rows, n_rows), dtype=np.float32)
for i, r in enumerate(rows):
    perm = list(r["order"])
    for a, b in transpositions:
        new_perm = list(perm); new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
        j = perm_to_idx[tuple(new_perm)]
        A_cayley[i, j] = 1.0

degree = K*(K-1)//2  # 15
L_cayley = degree * np.eye(n_rows) - A_cayley.astype(np.float64)

from numpy.linalg import eigh
eigenvalues, U_eigen = eigh(L_cayley)  # U_eigen[:,k] = k番目固有ベクトル
print(f"  固有値範囲: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
print(f"  最小5固有値: {eigenvalues[:5].round(4)}")
print(f"  固有値の分布（0=連結成分数）: 0が{(eigenvalues < 1e-6).sum()}個")

# ウェーブレット用スペクトルバンド境界を固有値パーセンタイルで決定
nonzero_ev = eigenvalues[eigenvalues > 1e-6]
band_cuts = np.percentile(nonzero_ev, [0, 20, 40, 60, 80, 100])
print(f"  スペクトルバンド境界 (20pct刻み): {band_cuts.round(3)}")

# ══════════════════════════════════════════════════════════════════
# 特徴量行列の準備
# ══════════════════════════════════════════════════════════════════
X_srm  = np.stack([r["srm"] for r in rows])          # (720, 25) 標準表現
X_gf_full = U_eigen                                    # (720, 720) グラフ Fourier 全基底
y_all  = dists

# ══════════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════════
def oracle_labels(rows_sub, N):
    N = int(min(N, len(rows_sub)))
    return {r["label"] for r in sorted(rows_sub, key=lambda r: r["dist"])[:N]}

def precision_at(sel, true):
    return float(len(sel & true)) / float(len(true)) if true else 0.0

def idx_of(rows_sub, global_rows=rows):
    """rows_sub の要素が global_rows 中の何番目か返す"""
    return [perm_to_idx[tuple(r["order"])] for r in rows_sub]

def threestage_orig(rows_sub, N):
    N = int(min(N, len(rows_sub)))
    all_d = np.array([r["dist"] for r in rows_sub])
    thr   = float(np.percentile(all_d, 25))
    ct_m  = defaultdict(list)
    for r in rows_sub: ct_m[r["ct_str"]].append(r["dist"])
    kept  = {ct for ct, v in ct_m.items() if float(np.mean(v)) <= thr}
    if not kept: kept = set(ct_m.keys())
    grps  = defaultdict(list)
    for r in rows_sub:
        if r["ct_str"] in kept: grps[r["ct_str"]].append(r)
    selected = []
    for ct, grp in grps.items():
        bv = np.array([r["burau"] for r in grp])
        dv = np.array([r["dist"]  for r in grp])
        rr = float(stats.pearsonr(bv, dv)[0]) if len(grp)>=3 else 0.0
        dr = -1 if rr>0 else +1
        n_s = int(max(1, round(N*len(grp)/sum(len(g) for g in grps.values()))))
        ranked = sorted(grp, key=lambda r: dr*r["burau"], reverse=True)
        selected.extend(ranked[:n_s])
    if len(selected)>N: selected = sorted(selected, key=lambda r: r["dist"])[:N]
    elif len(selected)<N:
        sl = {r["label"] for r in selected}
        rem = sorted([r for r in rows_sub if r["label"] not in sl],
                     key=lambda r: r["dist"])
        selected.extend(rem[:N-len(selected)])
    return {r["label"] for r in selected[:N]}

# ══════════════════════════════════════════════════════════════════
# Haar ウェーブレット (コセット prefix 多解像度)
# ══════════════════════════════════════════════════════════════════
def haar_predict(rows_tr, rows_te, shrink=8.0, max_level=3):
    """
    介入順序の prefix による階層的平均 (Haar ウェーブレット)
    Level 0: global mean
    Level 1: mean by order[0]
    Level 2: mean by (order[0], order[1])
    Level 3: mean by (order[0], order[1], order[2])
    各レベルで James-Stein 型収縮 (shrink パラメータ)
    """
    y_tr = np.array([r["dist"] for r in rows_tr])
    global_mean = float(y_tr.mean())

    def shrunk(values, parent, lam):
        if not values: return parent
        n = len(values); mu = np.mean(values)
        return n/(n+lam)*mu + lam/(n+lam)*parent

    # level 0
    L0 = defaultdict(list)
    for r, y in zip(rows_tr, y_tr):
        L0[(r["order"][0],)].append(y)
    m0 = {k: shrunk(v, global_mean, shrink) for k,v in L0.items()}

    # level 1
    L1 = defaultdict(list)
    for r, y in zip(rows_tr, y_tr):
        L1[tuple(r["order"][:2])].append(y)
    m1 = {k: shrunk(v, m0.get(k[:1], global_mean), shrink) for k,v in L1.items()}

    # level 2
    L2 = defaultdict(list)
    for r, y in zip(rows_tr, y_tr):
        L2[tuple(r["order"][:3])].append(y)
    m2 = {k: shrunk(v, m1.get(k[:2], m0.get(k[:1], global_mean)), shrink)
          for k,v in L2.items()}

    def pred(r):
        o = tuple(r["order"])
        if max_level >= 3 and o[:3] in m2: return m2[o[:3]]
        if max_level >= 2 and o[:2] in m1: return m1[o[:2]]
        if max_level >= 1 and o[:1] in m0: return m0[o[:1]]
        return global_mean

    return np.array([pred(r) for r in rows_te])

# ══════════════════════════════════════════════════════════════════
# グラフスペクトル予測（訓練データの固有係数から）
# ══════════════════════════════════════════════════════════════════
def graph_spectral_predict(tr_idx, te_idx, y_tr, eigenvalues, U,
                            mode="fourier", n_modes=50, scale=1.0):
    """
    mode="fourier" : 最初の n_modes 固有ベクトルで Ridge 回帰
    mode="heat"    : 熱核フィルタ exp(-scale*λ) で平滑化
    mode="wavelet" : メキシカンハット型フィルタ (λ exp(-scale*λ))
    """
    if mode == "fourier":
        X_tr = U[tr_idx, :n_modes]
        X_te = U[te_idx, :n_modes]
        reg  = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(X_tr, y_tr)
        return reg.predict(X_te)

    elif mode == "heat":
        # α_k = Σ_i y_i U[i,k]  (Fourier係数: 訓練データ全体での推定)
        # 予測: Σ_k exp(-s*λ_k) α_k U[j,k]
        alpha = U[tr_idx].T @ y_tr      # shape (n_all,)
        filt  = np.exp(-scale * eigenvalues)
        return U[te_idx] @ (filt * alpha)

    elif mode == "wavelet":
        # メキシカンハットフィルタ: λ exp(-scale*λ)  (高周波検出)
        alpha = U[tr_idx].T @ y_tr
        filt  = eigenvalues * np.exp(-scale * eigenvalues)
        # バンドパス (特定スケール) の成分: ベースとの差
        filt_low  = np.exp(-scale*2 * eigenvalues)
        filt_band = filt_low - np.exp(-scale*0.5 * eigenvalues)
        return U[te_idx] @ (filt_band * alpha)

def graph_multiscale_predict(tr_idx, te_idx, y_tr, eigenvalues, U):
    """
    複数スケールの heat 予測を Ridge で結合 (Wavelet packet 的アプローチ)
    """
    scales = [0.02, 0.1, 0.5, 2.0, 8.0]
    features = []
    for s in scales:
        alpha = U[tr_idx].T @ y_tr
        filt  = np.exp(-s * eigenvalues)
        y_s   = U @ (filt * alpha)  # 全頂点への適用
        features.append(y_s)
    X_multi = np.stack([f[tr_idx] for f in features], axis=1)
    X_te    = np.stack([f[te_idx] for f in features], axis=1)
    reg = Ridge(alpha=0.1, fit_intercept=True)
    reg.fit(X_multi, y_tr)
    return reg.predict(X_te), features  # features: list of (720,) arrays

def ridge_srm_predict(tr_idx, te_idx, y_tr, X_srm, alpha_reg=1.0):
    """Ridge 正則化 Fourier (標準表現ベース)"""
    reg = Ridge(alpha=alpha_reg, fit_intercept=True)
    reg.fit(X_srm[tr_idx], y_tr)
    return reg.predict(X_srm[te_idx])

def combined_srm_gf_predict(tr_idx, te_idx, y_tr, X_srm, U, n_modes=30):
    """標準表現 + グラフ Fourier 複合特徴"""
    X_gf_tr = U[tr_idx, :n_modes]
    X_gf_te = U[te_idx, :n_modes]
    X_tr    = np.hstack([X_srm[tr_idx], X_gf_tr])
    X_te    = np.hstack([X_srm[te_idx], X_gf_te])
    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(X_tr, y_tr)
    return reg.predict(X_te)

# ══════════════════════════════════════════════════════════════════
# Fourier×3-Stage 統合B: Fourier detrend → 残差に 3-Stage
# ══════════════════════════════════════════════════════════════════
def fourier_detrend_3stage(rows_sub, te_idx_in_sub, tr_idx_in_sub,
                            y_tr_sub, X_srm_sub, y_te_sub, N):
    """
    1. SRM Fourier でグローバルトレンドを除去
    2. 残差に対して 3-Stage フィルタ + Burau ランキング
    3. 予測スコア = Fourier予測 + ε × 3Stage補正
    """
    N = int(min(N, len(te_idx_in_sub)))

    # Fourier fit
    reg = Ridge(alpha=1.0); reg.fit(X_srm_sub[tr_idx_in_sub], y_tr_sub)
    fourier_pred_te = reg.predict(X_srm_sub[te_idx_in_sub])

    # 残差 3-Stage: 残差の小さい順に選ぶ
    rows_te = [rows_sub[i] for i in te_idx_in_sub]
    residuals_te = y_te_sub - fourier_pred_te  # 未知（テスト用なので使えない）

    # CV 内では y_te は使えないので Fourier 予測値のみでスコア付け
    # 残差の代わりに「Fourier予測 × cycle_type補正」を使う
    # cycle_type の mean Fourier_pred に対する相対値
    ct_fourier_tr = defaultdict(list)
    rows_tr_sub = [rows_sub[i] for i in tr_idx_in_sub]
    fp_tr = reg.predict(X_srm_sub[tr_idx_in_sub])
    for r, fp in zip(rows_tr_sub, fp_tr):
        ct_fourier_tr[r["ct_str"]].append(fp)
    ct_fp_mean = {ct: float(np.mean(v)) for ct, v in ct_fourier_tr.items()}

    # 3-Stage フィルタ: Fourier予測 dist が小さいグループを優先
    ct_fp_thr = float(np.percentile(fourier_pred_te, 25))
    kept_ct = set()
    ct_fp_te = defaultdict(list)
    for fp, r in zip(fourier_pred_te, rows_te):
        ct_fp_te[r["ct_str"]].append(fp)
    for ct, v in ct_fp_te.items():
        if float(np.mean(v)) <= ct_fp_thr:
            kept_ct.add(ct)
    if not kept_ct: kept_ct = set(ct_fp_te.keys())

    # フィルタ後を Fourier 予測でランキング
    filtered = [(r, fp) for r, fp in zip(rows_te, fourier_pred_te)
                if r["ct_str"] in kept_ct]
    filtered.sort(key=lambda x: x[1])  # 予測 dist 昇順
    selected = {r["label"] for r, _ in filtered[:N]}
    if len(selected) < N:
        rem = sorted([(r,fp) for r,fp in zip(rows_te, fourier_pred_te)
                      if r["label"] not in selected], key=lambda x: x[1])
        selected.update(r["label"] for r, _ in rem[:N-len(selected)])
    return selected

# ══════════════════════════════════════════════════════════════════
# 5-fold CV
# ══════════════════════════════════════════════════════════════════
N_vals = [10, 20, 50, 100, 150]
kf = KFold(n_splits=5, shuffle=True, random_state=int(42))

STRATEGIES = [
    # ベースライン
    "3Stg(orig)",
    "SRM_linear",    # W-7m の Fourier (LR)
    # グラフ Fourier (固有ベクトル数)
    "GF_M10", "GF_M25", "GF_M50", "GF_M100",
    # グラフ熱核スムージング (スケール)
    "Heat_s002", "Heat_s01", "Heat_s05", "Heat_s2",
    # グラフマルチスケール (Wavelet packet)
    "Wavelet_multi",
    # Haar ウェーブレット (収縮強度 × レベル)
    "Haar_L1", "Haar_L2_s5", "Haar_L2_s15", "Haar_L3_s15",
    # Ridge 正則化 SRM (α=0.01, 0.1, 1, 10, 100)
    "Ridge_a001","Ridge_a01","Ridge_a1","Ridge_a10","Ridge_a100",
    # SRM + グラフ Fourier 複合
    "SRM+GF_M30",
    # 統合A: 3-Stage フィルタ × SRM Fourier ランキング
    "3Stg+SRM",
    # 統合B: Fourier detrend → Fourier-informed 3-Stage
    "FourierFilter3Stg",
    # GlobalBurau (参考)
    "GlobalBurau",
]

cv_prec = {s: {N: [] for N in N_vals} for s in STRATEGIES}

print("\n[5-fold CV] 進捗: ", end="", flush=True)
for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(rows)):
    print(f"fold{fold_idx+1} ", end="", flush=True)
    rows_tr = [rows[i] for i in tr_idx]
    rows_te = [rows[i] for i in te_idx]
    y_tr    = y_all[tr_idx]
    y_te    = y_all[te_idx]

    # ---------- 各予測スコアを計算 ----------
    # SRM Fourier (linear)
    lr_srm = LinearRegression()
    lr_srm.fit(X_srm[tr_idx], y_tr)
    y_srm = lr_srm.predict(X_srm[te_idx])

    # グラフ Fourier (固有ベクトル)
    y_gf = {}
    for M in [10, 25, 50, 100]:
        y_gf[M] = graph_spectral_predict(tr_idx, te_idx, y_tr,
                                          eigenvalues, U_eigen, mode="fourier", n_modes=M)

    # 熱核スムージング
    y_heat = {}
    for s in [0.02, 0.1, 0.5, 2.0]:
        y_heat[s] = graph_spectral_predict(tr_idx, te_idx, y_tr,
                                            eigenvalues, U_eigen, mode="heat", scale=s)

    # マルチスケール wavelet
    y_multi, _ = graph_multiscale_predict(tr_idx, te_idx, y_tr, eigenvalues, U_eigen)

    # Haar ウェーブレット
    y_haar = {
        "L1"   : haar_predict(rows_tr, rows_te, shrink=8.0,  max_level=1),
        "L2s5" : haar_predict(rows_tr, rows_te, shrink=5.0,  max_level=2),
        "L2s15": haar_predict(rows_tr, rows_te, shrink=15.0, max_level=2),
        "L3s15": haar_predict(rows_tr, rows_te, shrink=15.0, max_level=3),
    }

    # Ridge SRM
    y_ridge = {}
    for alpha_r in [0.01, 0.1, 1.0, 10.0, 100.0]:
        y_ridge[alpha_r] = ridge_srm_predict(tr_idx, te_idx, y_tr,
                                              X_srm, alpha_reg=alpha_r)

    # 複合特徴
    y_combo = combined_srm_gf_predict(tr_idx, te_idx, y_tr, X_srm, U_eigen, n_modes=30)

    # GlobalBurau
    bur_te  = np.array([r["burau"] for r in rows_te])
    bur_tr  = np.array([r["burau"] for r in rows_tr])
    r_bur_f = float(stats.pearsonr(bur_tr, y_tr)[0])
    d_bur_f = -1.0 if r_bur_f > 0 else +1.0
    y_gb    = d_bur_f * bur_te

    # 3-Stage フィルタ後 SRM ランキング
    def threestage_then_pred(rows_sub, y_pred_sub, N):
        N = int(min(N, len(rows_sub)))
        all_d = np.array([r["dist"] for r in rows_sub])
        thr   = float(np.percentile(all_d, 25))
        ct_m  = defaultdict(list)
        for r in rows_sub: ct_m[r["ct_str"]].append(r["dist"])
        kept  = {ct for ct, v in ct_m.items() if float(np.mean(v)) <= thr}
        if not kept: kept = set(ct_m.keys())
        pairs = [(r, yp) for r, yp in zip(rows_sub, y_pred_sub)
                 if r["ct_str"] in kept]
        pairs.sort(key=lambda x: x[1])
        selected = {r["label"] for r, _ in pairs[:N]}
        if len(selected) < N:
            rem = sorted([(r,yp) for r,yp in zip(rows_sub, y_pred_sub)
                          if r["label"] not in selected], key=lambda x: x[1])
            selected.update(r["label"] for r,_ in rem[:N-len(selected)])
        return selected

    for N in N_vals:
        n = int(min(N, len(rows_te)))
        true_top = oracle_labels(rows_te, n)

        def run(y_pred, lower_better=True):
            score = -y_pred if lower_better else y_pred
            ranked_idx = np.argsort(-score)[:n]
            return {rows_te[i]["label"] for i in ranked_idx}

        def prc(y_pred, lower_better=True):
            return precision_at(run(y_pred, lower_better), true_top)

        cv_prec["3Stg(orig)"][N].append(
            precision_at(threestage_orig(rows_te, n), true_top))
        cv_prec["SRM_linear"][N].append(prc(y_srm))
        cv_prec["GF_M10"][N].append(prc(y_gf[10]))
        cv_prec["GF_M25"][N].append(prc(y_gf[25]))
        cv_prec["GF_M50"][N].append(prc(y_gf[50]))
        cv_prec["GF_M100"][N].append(prc(y_gf[100]))
        cv_prec["Heat_s002"][N].append(prc(y_heat[0.02]))
        cv_prec["Heat_s01"][N].append(prc(y_heat[0.1]))
        cv_prec["Heat_s05"][N].append(prc(y_heat[0.5]))
        cv_prec["Heat_s2"][N].append(prc(y_heat[2.0]))
        cv_prec["Wavelet_multi"][N].append(prc(y_multi))
        cv_prec["Haar_L1"][N].append(prc(y_haar["L1"]))
        cv_prec["Haar_L2_s5"][N].append(prc(y_haar["L2s5"]))
        cv_prec["Haar_L2_s15"][N].append(prc(y_haar["L2s15"]))
        cv_prec["Haar_L3_s15"][N].append(prc(y_haar["L3s15"]))
        cv_prec["Ridge_a001"][N].append(prc(y_ridge[0.01]))
        cv_prec["Ridge_a01"][N].append(prc(y_ridge[0.1]))
        cv_prec["Ridge_a1"][N].append(prc(y_ridge[1.0]))
        cv_prec["Ridge_a10"][N].append(prc(y_ridge[10.0]))
        cv_prec["Ridge_a100"][N].append(prc(y_ridge[100.0]))
        cv_prec["SRM+GF_M30"][N].append(prc(y_combo))
        cv_prec["3Stg+SRM"][N].append(
            precision_at(threestage_then_pred(rows_te, y_srm, n), true_top))
        # 統合B
        sub_te_idx = list(range(len(rows_te)))
        sub_tr_idx = list(range(len(rows_tr)))  # 実際は全rows_tr
        cv_prec["FourierFilter3Stg"][N].append(
            precision_at(
                fourier_detrend_3stage(
                    rows_te, sub_te_idx, sub_te_idx,  # train=test (近似)
                    y_srm, X_srm[te_idx], y_te, n),   # ← 近似
                true_top))
        cv_prec["GlobalBurau"][N].append(prc(y_gb, lower_better=False))

print("完了")

# 統合B を正しく再実装（train で学習して test に適用）
print("  [統合B 再計算 — 正しい CV 設定]")
for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(rows)):
    rows_tr = [rows[i] for i in tr_idx]
    rows_te = [rows[i] for i in te_idx]
    y_tr    = y_all[tr_idx]
    y_te    = y_all[te_idx]

    reg = Ridge(alpha=1.0); reg.fit(X_srm[tr_idx], y_tr)
    fourier_pred_te = reg.predict(X_srm[te_idx])

    # 訓練データで cycle type ごとの Fourier 予測平均を計算
    fp_tr = reg.predict(X_srm[tr_idx])
    ct_fp_tr = defaultdict(list)
    for r, fp in zip(rows_tr, fp_tr): ct_fp_tr[r["ct_str"]].append(fp)
    ct_fp_mean_tr = {ct: float(np.mean(v)) for ct, v in ct_fp_tr.items()}

    # test での Fourier 予測に対して、訓練から学習した CT 平均で soft フィルタ
    ct_fp_sorted = sorted(ct_fp_mean_tr.values())
    thr_ct = float(np.percentile(ct_fp_sorted, 25)) if ct_fp_sorted else 0.0

    kept_ct = {ct for ct, m in ct_fp_mean_tr.items() if m <= thr_ct}
    if not kept_ct: kept_ct = set(ct_fp_mean_tr.keys())

    for N in N_vals:
        n = int(min(N, len(rows_te)))
        true_top = oracle_labels(rows_te, n)
        # フィルタ後を Fourier 予測でランキング
        pairs_filtered = sorted(
            [(r, fp) for r, fp in zip(rows_te, fourier_pred_te)
             if r["ct_str"] in kept_ct],
            key=lambda x: x[1])
        selected = {r["label"] for r, _ in pairs_filtered[:n]}
        if len(selected) < n:
            rem = sorted([(r, fp) for r, fp in zip(rows_te, fourier_pred_te)
                          if r["label"] not in selected], key=lambda x: x[1])
            selected.update(r["label"] for r, _ in rem[:n-len(selected)])
        # 上書き更新（fold ごとに追記）
        cv_prec["FourierFilter3Stg"][N][fold_idx] = precision_at(selected, true_top)

# ══════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════
print("\n[5-fold CV Precision@N — 全手法]")
DISPLAY_SECTIONS = [
    ("── ベースライン", ["GlobalBurau","3Stg(orig)","SRM_linear"]),
    ("── グラフ Fourier (固有ベクトル数)", ["GF_M10","GF_M25","GF_M50","GF_M100"]),
    ("── 熱核スムージング (スケール小→大)",["Heat_s002","Heat_s01","Heat_s05","Heat_s2"]),
    ("── Wavelet (マルチスケール packet)", ["Wavelet_multi"]),
    ("── Haar ウェーブレット (prefix 多解像度)", ["Haar_L1","Haar_L2_s5","Haar_L2_s15","Haar_L3_s15"]),
    ("── Ridge 正則化 Fourier (α小→大=平滑)", ["Ridge_a001","Ridge_a01","Ridge_a1","Ridge_a10","Ridge_a100"]),
    ("── 複合特徴", ["SRM+GF_M30"]),
    ("── Fourier × 3-Stage 統合", ["3Stg+SRM","FourierFilter3Stg"]),
]
hdr = f"  {'手法':<22}|"
for N in N_vals: hdr += f"  @{N:<4}|"
print(hdr)
ref_3stg = float(np.mean(cv_prec["3Stg(orig)"][50]))
for sec, keys in DISPLAY_SECTIONS:
    print(f"\n  {sec}")
    print("  " + "-"*(22 + 8*len(N_vals) + 2))
    for key in keys:
        line = f"  {key:<22}|"
        for N in N_vals: line += f"  {np.mean(cv_prec[key][N]):.3f}|"
        v50 = float(np.mean(cv_prec[key][50]))
        tag = "  ★★★" if v50 > ref_3stg + 0.005 else ("  ★" if v50 > ref_3stg - 0.005 else "")
        print(line + tag)

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("W-7n: Fourier×3-Stage Integration + Wavelets (5-fold CV)", fontsize=13)

palette = {
    "GlobalBurau"  : ("blue",       "o",  "--", 2.0),
    "3Stg(orig)"   : ("green",      "s",  "-",  2.8),
    "SRM_linear"   : ("darkorange", "^",  "-",  2.0),
    "GF_M10"       : ("#E53935",    "D",  "-",  1.5),
    "GF_M25"       : ("#C62828",    "P",  "-",  1.8),
    "GF_M50"       : ("#B71C1C",    "*",  "-",  2.0),
    "GF_M100"      : ("#7F0000",    "h",  "-",  1.5),
    "Heat_s002"    : ("#9C27B0",    "^",  "-",  1.5),
    "Heat_s01"     : ("#7B1FA2",    "v",  "-",  1.8),
    "Heat_s05"     : ("#6A1B9A",    "D",  "-",  2.0),
    "Heat_s2"      : ("#4A148C",    "P",  "-",  1.5),
    "Wavelet_multi": ("#FF6F00",    "*",  "-",  2.2),
    "Haar_L1"      : ("#1565C0",    "^",  "-",  1.5),
    "Haar_L2_s5"   : ("#0D47A1",    "v",  "-",  1.8),
    "Haar_L2_s15"  : ("#1A237E",    "D",  "-",  2.0),
    "Haar_L3_s15"  : ("#0D1B6E",    "P",  "-",  1.5),
    "Ridge_a001"   : ("#F44336",    "^",  "-",  1.5),
    "Ridge_a01"    : ("#E91E63",    "v",  "-",  1.8),
    "Ridge_a1"     : ("#9C27B0",    "D",  "-",  2.0),
    "Ridge_a10"    : ("#3F51B5",    "P",  "-",  1.8),
    "Ridge_a100"   : ("#2196F3",    "h",  "-",  1.5),
    "SRM+GF_M30"   : ("#FF5722",    "*",  "-",  2.2),
    "3Stg+SRM"     : ("#4CAF50",    "D",  "--", 2.2),
    "FourierFilter3Stg":("#00BCD4", "P",  "-",  2.2),
}

groups_plot = [
    ("Graph Fourier (# modes)",
     ["GlobalBurau","3Stg(orig)","SRM_linear","GF_M10","GF_M25","GF_M50","GF_M100"],
     axes[0,0]),
    ("Heat Kernel Smoothing (scale)",
     ["GlobalBurau","3Stg(orig)","Heat_s002","Heat_s01","Heat_s05","Heat_s2"],
     axes[0,1]),
    ("Wavelet Packet + Haar",
     ["GlobalBurau","3Stg(orig)","Wavelet_multi",
      "Haar_L1","Haar_L2_s5","Haar_L2_s15","Haar_L3_s15"],
     axes[0,2]),
    ("Ridge-regularized Fourier (α)",
     ["GlobalBurau","3Stg(orig)","Ridge_a001","Ridge_a01","Ridge_a1","Ridge_a10","Ridge_a100"],
     axes[1,0]),
    ("Combined Features",
     ["GlobalBurau","3Stg(orig)","SRM_linear","SRM+GF_M30"],
     axes[1,1]),
    ("Fourier × 3-Stage Integration",
     ["GlobalBurau","3Stg(orig)","SRM_linear","3Stg+SRM","FourierFilter3Stg"],
     axes[1,2]),
]

for title, keys, ax in groups_plot:
    for key in keys:
        if key not in palette: continue
        c, mk, ls, lw = palette[key]
        means = [np.mean(cv_prec[key][N]) for N in N_vals]
        stds  = [np.std(cv_prec[key][N])  for N in N_vals]
        ax.plot(N_vals, means, marker=mk, ls=ls, color=c, lw=lw,
                label=key, alpha=0.9)
        ax.fill_between(N_vals, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], color=c, alpha=0.1)
    ax.axhline(ref_3stg, ls=":", color="green", alpha=0.4, lw=1.0)
    ax.set_title(title, fontsize=10); ax.set_xlabel("N"); ax.set_ylabel("Precision@N")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(OUT / "W7n_fig1_precision_groups.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig1 saved: {OUT / 'W7n_fig1_precision_groups.png'}")

# ── 図2: Ridge α スイープ (Precision@N vs α) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("W-7n: Ridge Regularization Sweep — Fourier Smoothing Level", fontsize=12)

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_keys = ["Ridge_a001","Ridge_a01","Ridge_a1","Ridge_a10","Ridge_a100"]
colors_r   = plt.cm.coolwarm(np.linspace(0, 1, len(alphas)))

for ax, N_target, title in [(axes[0], 20, "@N=20"), (axes[1], 50, "@N=50")]:
    p_vals = [float(np.mean(cv_prec[k][N_target])) for k in ridge_keys]
    p_stds = [float(np.std(cv_prec[k][N_target]))  for k in ridge_keys]
    ax.semilogx(alphas, p_vals, "o-", color="purple", lw=2, ms=8)
    ax.fill_between(alphas,
                    [v-s for v,s in zip(p_vals,p_stds)],
                    [v+s for v,s in zip(p_vals,p_stds)],
                    alpha=0.2, color="purple")
    ax.axhline(float(np.mean(cv_prec["3Stg(orig)"][N_target])), ls="--",
               color="green", label="3-Stage-25%")
    ax.axhline(float(np.mean(cv_prec["SRM_linear"][N_target])), ls="-.",
               color="orange", label="SRM (no reg)")
    ax.set_xlabel("Ridge α (larger = smoother)")
    ax.set_ylabel(f"Precision{title}")
    ax.set_title(f"Precision{title} vs Ridge α")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "W7n_fig2_ridge_sweep.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig2 saved: {OUT / 'W7n_fig2_ridge_sweep.png'}")

# ── 図3: Cayley グラフのスペクトル固有値分布 ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("W-7n: Cayley Graph Spectral Properties (K=6)", fontsize=12)

ax = axes[0]
ax.hist(eigenvalues, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
for bc in band_cuts[1:-1]:
    ax.axvline(bc, ls="--", color="red", alpha=0.5)
ax.set_xlabel("Eigenvalue λ"); ax.set_ylabel("Count")
ax.set_title("Laplacian Eigenvalue Distribution\n(red dashes = 20-pct band cuts)")
ax.grid(True, alpha=0.3)

ax = axes[1]
# 固有モード数 M vs P@50 のカーブ
M_vals  = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
gf_p50  = []
for M in M_vals:
    precs = []
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(rows)):
        y_tr = y_all[tr_idx]
        yp   = graph_spectral_predict(tr_idx, te_idx, y_tr,
                                       eigenvalues, U_eigen, mode="fourier", n_modes=M)
        n    = int(min(50, len(te_idx)))
        tt   = oracle_labels([rows[i] for i in te_idx], n)
        ranked = np.argsort(yp)[:n]
        sel = {rows[te_idx[i]]["label"] for i in ranked}
        precs.append(precision_at(sel, tt))
    gf_p50.append(float(np.mean(precs)))

ax.plot(M_vals, gf_p50, "o-", color="red", lw=2, ms=6, label="Graph Fourier P@50")
ax.axhline(ref_3stg, ls="--", color="green", label=f"3-Stage ({ref_3stg:.3f})")
ax.axhline(float(np.mean(cv_prec["SRM_linear"][50])), ls="-.",
           color="orange", label=f"SRM Fourier ({np.mean(cv_prec['SRM_linear'][50]):.3f})")
ax.set_xlabel("Number of eigenvectors M")
ax.set_ylabel("Precision@50 (5-fold CV)")
ax.set_title("Graph Fourier: Optimal # Modes")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "W7n_fig3_spectral.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig3 saved: {OUT / 'W7n_fig3_spectral.png'}")

# ══════════════════════════════════════════════════════════════════
# 総合サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  W-7n 総合サマリー")
print("=" * 65)

all_strats_ranked = sorted(
    [(k, float(np.mean(cv_prec[k][50]))) for k in STRATEGIES],
    key=lambda x: -x[1])

print(f"\n  Precision@50 全手法ランキング:")
val_3stg = float(np.mean(cv_prec["3Stg(orig)"][50]))
for rank, (key, val) in enumerate(all_strats_ranked, 1):
    bar = "█" * int(val * 30)
    tag = " ★★★ 3-Stage超え！" if val > val_3stg + 0.005 else (
          " ★   3-Stage並"   if val > val_3stg - 0.005 else "")
    print(f"  {rank:2}. {key:<24}: {val:.4f}  {bar}{tag}")

best_new = max((k for k in STRATEGIES if k != "3Stg(orig)"),
               key=lambda k: float(np.mean(cv_prec[k][50])))
val_best = float(np.mean(cv_prec[best_new][50]))
print(f"\n  ★ 新手法 最良: {best_new}  P@50={val_best:.4f}")
print(f"    3-Stage(orig):         P@50={val_3stg:.4f}")
if val_best > val_3stg + 0.005:
    print(f"  → 3-Stage を {(val_best-val_3stg)*100:.1f}pt 上回った！")
elif val_best > val_3stg - 0.005:
    print(f"  → 3-Stage に並んだ（±0.5pt 以内）")
else:
    print(f"  → 差 {(val_3stg-val_best)*100:.1f}pt  3-Stage が依然優位")

# 小 N (N=10, 20) で 3-Stage を超えた手法
print(f"\n  [小 N での 3-Stage 超え]")
val_3s_10 = float(np.mean(cv_prec["3Stg(orig)"][10]))
val_3s_20 = float(np.mean(cv_prec["3Stg(orig)"][20]))
for key in STRATEGIES:
    v10 = float(np.mean(cv_prec[key][10]))
    v20 = float(np.mean(cv_prec[key][20]))
    if v10 > val_3s_10 + 0.01 or v20 > val_3s_20 + 0.01:
        print(f"    {key:<24}: @10={v10:.3f}(ref:{val_3s_10:.3f})  "
              f"@20={v20:.3f}(ref:{val_3s_20:.3f})")

# 最適 Ridge α
print(f"\n  [Ridge 最適 α (P@50 最大)]")
ridge_perf = {a: float(np.mean(cv_prec[k][50]))
              for a, k in zip(alphas, ridge_keys)}
best_a = max(ridge_perf, key=ridge_perf.get)
print(f"    最適 α={best_a}  P@50={ridge_perf[best_a]:.4f}")
print(f"    α小 (過学習) → α大 (過平滑化) のトレードオフ:")
for a, k in zip(alphas, ridge_keys):
    bar = "█" * int(ridge_perf[a] * 30)
    print(f"    α={a:<8}: {ridge_perf[a]:.4f}  {bar}")

print("\n" + "=" * 65)
print(f"  W-7n 実験完了  出力先: {OUT}")
print("=" * 65)
