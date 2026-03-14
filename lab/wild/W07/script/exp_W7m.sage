#!/usr/bin/env sage
# exp_W7m.sage — 解析的スクリーニング 全4方向
# 方向1: OPW  (演算子-位置重み付きスコア)
# 方向2: Jac  (介入列の Jacobian トレース・スペクトル)
# 方向3: Coset(最初の介入による層別フィルタ)
# 方向4: Fourier (S_K 上の調和解析 / 標準表現による平滑化)
# 組み合わせ: 3-Stage フィルタ × 各新スコア

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import copy

print("=" * 65)
print("  W-7m: 解析的スクリーニング 全4方向")
print("  方向1:OPW  方向2:Jacobian  方向3:Coset  方向4:Fourier")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7m")
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 数値セットアップ
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
# 方向1 の前処理: 各演算子の Jacobian ノルム (重み wᵢ)
# ══════════════════════════════════════════════════════════════════
print("\n[方向1] 各演算子の Jacobian ノルム（at p0_master, θ=θ_star）")
K = 6
ops_ref = make_ops(K)
eps_J   = 1e-5
op_weights = np.zeros(K)
for idx in range(K):
    J_op = np.zeros((d, d))
    p_ref = ops_ref[idx](p0_master, theta_star)
    for j in range(d):
        delta = np.zeros(d); delta[j] = eps_J
        J_op[:, j] = (ops_ref[idx](p0_master+delta, theta_star) - p_ref) / eps_J
    op_weights[idx] = float(np.linalg.norm(J_op - np.eye(d), 'fro'))

op_weights_norm = op_weights / (op_weights.sum() + 1e-8)
for i, (w, wn) in enumerate(zip(op_weights, op_weights_norm)):
    bar = "█" * int(wn * 40)
    print(f"  E{i+1}: ||J_i - I||_F = {w:.4f}  (weight={wn:.4f})  {bar}")

# ══════════════════════════════════════════════════════════════════
# 方向2 の前処理: 介入列全体の Jacobian（有限差分）
# ══════════════════════════════════════════════════════════════════
def forward_full(order, K, p_init):
    ops = make_ops(K);  p = p_init.copy()
    for _ in range(N_ROUNDS):
        for idx in order:
            p = time_ev(p);  p = ops[idx](p, theta_star)
    return p

def compute_seq_jacobian(order, K, eps=1e-4):
    p0 = p0_master.copy()
    p_ref = forward_full(order, K, p0)
    J = np.zeros((d, d))
    for j in range(d):
        delta = np.zeros(d); delta[j] = eps
        J[:, j] = (forward_full(order, K, p0+delta) - p_ref) / eps
    eigv = np.linalg.eigvals(J)
    return {
        "jac_trace"  : float(np.trace(J)),
        "jac_specrad": float(np.max(np.abs(eigv))),
        "jac_frob"   : float(np.linalg.norm(J, 'fro')),
        "jac_det"    : float(np.real(np.prod(eigv))),
        "jac_entropy": float(-np.sum(np.abs(eigv) * np.log(np.abs(eigv)+1e-15))),
    }

# ══════════════════════════════════════════════════════════════════
# 方向4 の前処理: 標準表現行列
# ══════════════════════════════════════════════════════════════════
def standard_rep_matrix(perm_0idx, K):
    """
    S_K の標準表現行列 (K-1)×(K-1)
    基底: v_j = e_j - e_{K-1}  for j=0,...,K-2
    σ(e_j - e_{K-1}) = e_{σ(j)} - e_{σ(K-1)}
    """
    M = np.zeros((K-1, K-1))
    sig = perm_0idx  # 0-indexed list
    sK1 = sig[K-1]   # σ(K-1)
    for j in range(K-1):
        sj = sig[j]   # σ(j)
        # e_{σ(j)} - e_{σ(K-1)}
        # = (e_{σ(j)}-e_{K-1}) - (e_{σ(K-1)}-e_{K-1})  ... expressed in basis
        if sj < K-1:   M[sj, j]  += 1.0
        if sK1 < K-1:  M[sK1, j] -= 1.0
    return M

# ══════════════════════════════════════════════════════════════════
# データ構築
# ══════════════════════════════════════════════════════════════════
print(f"\n[K={K}] 全 {factorial(K)} 件のデータ構築中（Jacobian含む）...")
BK = BraidGroup(K);  gK = list(BK.generators())
rows = []
for perm_iter in iter_perms(range(1, K+1)):
    sp   = SagePerm(list(perm_iter))
    b    = BK.one()
    for i in sp.reduced_word(): b = b * gK[i-1]
    btr  = float(b.burau_matrix()(t=QQ(1)/QQ(2)).trace())
    ct   = list(sp.cycle_type())
    order= [x-1 for x in perm_iter]   # 0-indexed

    dist = compute_dist(order, K)
    jac  = compute_seq_jacobian(order, K)
    srm  = standard_rep_matrix(order, K)

    # 方向1: OPW スコア (weighted displacement)
    # displacement of Eᵢ = |position_of_Eᵢ_in_order - i|
    inv_perm = [0]*K
    for pos, ei in enumerate(order): inv_perm[ei] = pos
    opw_linear  = -float(sum(op_weights_norm[i]*abs(inv_perm[i]-i) for i in range(K)))
    opw_binary  = -float(sum(op_weights_norm[i]*(1 if inv_perm[i]!=i else 0) for i in range(K)))
    opw_squared = -float(sum(op_weights_norm[i]*(inv_perm[i]-i)**2 for i in range(K)))

    rows.append({
        "label"       : "".join(str(x+1) for x in order),
        "order"       : order,
        "perm_1idx"   : list(perm_iter),
        "burau"       : btr,
        "writhe"      : int(len(sp.reduced_word())),
        "cayley"      : K - len(ct),
        "n_cycles"    : len(ct),
        "n_fixed"     : int(ct.count(1)),
        "max_cycle"   : int(max(ct)),
        "ct_str"      : str(ct),
        "first_op"    : int(order[0]),  # どの演算子が最初か
        "pos_E1"      : int(inv_perm[0]),  # E1が何番目に適用されるか
        # 方向1
        "opw_linear"  : opw_linear,
        "opw_binary"  : opw_binary,
        "opw_squared" : opw_squared,
        # 方向2
        **jac,
        # 方向4 (ベクトル化した標準表現行列)
        "srm"         : srm.ravel(),
        "dist"        : dist,
    })

dists = np.array([r["dist"] for r in rows])
print(f"  完了。dist: [{dists.min():.3f}, {dists.max():.3f}]  mean={dists.mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# 各新特徴量と dist の相関
# ══════════════════════════════════════════════════════════════════
print("\n[新特徴量と dist の相関]")
new_feats = ["opw_linear","opw_binary","opw_squared","pos_E1","first_op",
             "jac_trace","jac_specrad","jac_frob","jac_det","jac_entropy"]
corr_new = {}
for fn in new_feats:
    fv = np.array([float(r[fn]) for r in rows])
    rv, pv = stats.pearsonr(fv, dists)
    corr_new[fn] = float(rv)
    bar_len = int(abs(rv)*30)
    sign = "+" if rv >= 0 else "-"
    print(f"  {fn:<16}: r={rv:+.4f}  p={pv:.2e}  {sign}{'█'*bar_len}")

# 参考: 既存特徴量
print("\n  [参考: 既存特徴量]")
for fn in ["burau","cayley","n_fixed","writhe"]:
    fv = np.array([float(r[fn]) for r in rows])
    rv, pv = stats.pearsonr(fv, dists)
    bar_len = int(abs(rv)*30)
    sign = "+" if rv >= 0 else "-"
    print(f"  {fn:<16}: r={rv:+.4f}  p={pv:.2e}  {sign}{'█'*bar_len}")

# ══════════════════════════════════════════════════════════════════
# 方向3: コセット分析（最初の介入による層別）
# ══════════════════════════════════════════════════════════════════
print("\n[方向3: Coset分析 — 最初の介入 (first_op) による層別]")
coset_groups = defaultdict(list)
for r in rows: coset_groups[r["first_op"]].append(r)

for fop in sorted(coset_groups.keys()):
    grp = coset_groups[fop]
    dv  = np.array([r["dist"] for r in grp])
    bv  = np.array([r["burau"] for r in grp])
    rv  = float(stats.pearsonr(bv, dv)[0]) if len(grp) >= 3 else 0.0
    print(f"  E{fop+1} first: n={len(grp):3d}  mean_dist={dv.mean():.3f}  "
          f"Burau_r={rv:+.3f}")

# ══════════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════════
def oracle_labels(rows_sub, N):
    N = int(min(N, len(rows_sub)))
    return {r["label"] for r in sorted(rows_sub, key=lambda r: r["dist"])[:N]}

def precision_at(sel, true):
    return float(len(sel & true)) / float(len(true)) if true else 0.0

def select_top_by_score(rows_sub, score_fn, N, higher_better=True):
    N = int(min(N, len(rows_sub)))
    ranked = sorted(rows_sub, key=score_fn, reverse=higher_better)
    return {r["label"] for r in ranked[:N]}

# ══════════════════════════════════════════════════════════════════
# スコアリング関数（全データ統計を使う版 — CV内で再計算）
# ══════════════════════════════════════════════════════════════════
def make_scores_from_subset(rows_sub):
    """rows_sub から全スコアを計算して辞書で返す"""
    n = len(rows_sub)
    bur = np.array([r["burau"]       for r in rows_sub])
    cay = np.array([float(r["cayley"]) for r in rows_sub])
    dv  = np.array([r["dist"]        for r in rows_sub])

    # グローバル方向
    r_bur = float(stats.pearsonr(bur, dv)[0]) if n>=3 else 0.0
    d_bur = -1.0 if r_bur > 0 else +1.0

    # Jacobian 特徴量の方向
    jac_keys = ["jac_trace","jac_specrad","jac_frob","jac_entropy"]
    jac_dirs = {}
    for jk in jac_keys:
        jv = np.array([float(r[jk]) for r in rows_sub])
        rv = float(stats.pearsonr(jv, dv)[0]) if n>=3 else 0.0
        jac_dirs[jk] = (-1.0 if rv > 0 else +1.0, float(abs(rv)))

    # OPW 方向
    for ok in ["opw_linear","opw_binary","opw_squared"]:
        ov = np.array([float(r[ok]) for r in rows_sub])
        rv = float(stats.pearsonr(ov, dv)[0]) if n>=3 else 0.0
        jac_dirs[ok] = (-1.0 if rv > 0 else +1.0, float(abs(rv)))

    # pos_E1 方向
    pe1 = np.array([float(r["pos_E1"]) for r in rows_sub])
    rv_pe1 = float(stats.pearsonr(pe1, dv)[0]) if n>=3 else 0.0
    d_pe1 = -1.0 if rv_pe1 > 0 else +1.0

    # cycle type 統計
    ct_stats_loc = defaultdict(list)
    for r in rows_sub: ct_stats_loc[r["ct_str"]].append(r["dist"])
    ct_mean_loc  = {ct: float(np.mean(v)) for ct, v in ct_stats_loc.items()}
    ct_burau_loc = defaultdict(list)
    for r in rows_sub: ct_burau_loc[r["ct_str"]].append(r["burau"])
    ct_br_stats  = {ct: (float(np.mean(v)), float(np.std(v))+1e-8)
                    for ct, v in ct_burau_loc.items()}
    ct_r_loc     = {}
    for ct in ct_stats_loc:
        bv2 = np.array(ct_burau_loc[ct])
        dv2 = np.array(ct_stats_loc[ct])
        ct_r_loc[ct] = float(stats.pearsonr(bv2, dv2)[0]) if len(bv2)>=3 else 0.0

    # coset (first_op) 統計
    co_stats = defaultdict(list)
    for r in rows_sub: co_stats[r["first_op"]].append(r["dist"])
    co_mean  = {co: float(np.mean(v)) for co, v in co_stats.items()}

    # 標準表現行列 (Fourier 用)
    X_srm  = np.stack([r["srm"] for r in rows_sub])  # (n, (K-1)²)
    y_srm  = dv

    return {
        "d_bur": d_bur, "r_bur": r_bur,
        "jac_dirs": jac_dirs,
        "d_pe1": d_pe1,
        "ct_mean": ct_mean_loc, "ct_br_stats": ct_br_stats, "ct_r": ct_r_loc,
        "co_mean": co_mean,
        "X_srm": X_srm, "y_srm": y_srm,
    }

def apply_scores(rows_sub, rows_query, stats_dict, fit_fourier=True):
    """
    stats_dict から学習した各スコアを rows_query に適用。
    戻り値: dict{strategy_name: array of scores (higher=better)}
    """
    sd    = stats_dict
    n_q   = len(rows_query)
    bur_q = np.array([r["burau"] for r in rows_query])
    dv_q  = np.array([r["dist"]  for r in rows_query])

    scores = {}

    # ── GlobalBurau ──────────────────────────────────────────────
    scores["GlobalBurau"]   = sd["d_bur"] * bur_q

    # ── A: Cayley ────────────────────────────────────────────────
    scores["A:Cayley"]      = -np.array([float(r["cayley"]) for r in rows_query])

    # ── B1: pos_E1 ───────────────────────────────────────────────
    scores["B1:pos_E1"]     = sd["d_pe1"] * np.array([float(r["pos_E1"]) for r in rows_query])

    # ── B2: OPW_linear ───────────────────────────────────────────
    d_opw, _ = sd["jac_dirs"]["opw_linear"]
    scores["B2:OPW_linear"] = d_opw * np.array([float(r["opw_linear"]) for r in rows_query])

    # ── B3: OPW_squared ──────────────────────────────────────────
    d_opwsq, _ = sd["jac_dirs"]["opw_squared"]
    scores["B3:OPW_sq"]     = d_opwsq * np.array([float(r["opw_squared"]) for r in rows_query])

    # ── C1: Jac_trace ────────────────────────────────────────────
    d_jt, _ = sd["jac_dirs"]["jac_trace"]
    scores["C1:Jac_trace"]  = d_jt * np.array([float(r["jac_trace"]) for r in rows_query])

    # ── C2: Jac_specrad ──────────────────────────────────────────
    d_js, _ = sd["jac_dirs"]["jac_specrad"]
    scores["C2:Jac_srad"]   = d_js * np.array([float(r["jac_specrad"]) for r in rows_query])

    # ── C3: Jac_frob ─────────────────────────────────────────────
    d_jf, _ = sd["jac_dirs"]["jac_frob"]
    scores["C3:Jac_frob"]   = d_jf * np.array([float(r["jac_frob"]) for r in rows_query])

    # ── C4: Jac_entropy ──────────────────────────────────────────
    d_je, _ = sd["jac_dirs"]["jac_entropy"]
    scores["C4:Jac_ent"]    = d_je * np.array([float(r["jac_entropy"]) for r in rows_query])

    # ── C5: Burau + Jac_trace (複合) ─────────────────────────────
    def norm_arr(arr):
        s = arr.std(); return (arr-arr.mean())/s if s>1e-8 else arr*0.0
    bur_n_q  = norm_arr(bur_q)
    jt_q     = np.array([float(r["jac_trace"]) for r in rows_query])
    lam_jt   = sd["jac_dirs"]["jac_trace"][1] / (abs(sd["r_bur"])+1e-8)
    scores["C5:Bur+Jac"]    = (sd["d_bur"] * norm_arr(bur_q)
                                + lam_jt * d_jt * norm_arr(jt_q))

    # ── D: Coset フィルタスコア ───────────────────────────────────
    # mean_dist の小さい first_op を高スコアに
    co_mean_q = np.array([sd["co_mean"].get(r["first_op"], 999.0) for r in rows_query])
    scores["D:Coset_prior"] = -co_mean_q

    # ── D2: Coset フィルタ + Burau (後段は ranking で処理) ────────
    # → スコアとしては Coset_prior × (1 + Burau_norm / large_const) で近似
    scores["D2:Coset+Bur"]  = -co_mean_q + 0.1 * sd["d_bur"] * norm_arr(bur_q)

    # ── E: Fourier (標準表現行列で線形回帰) ─────────────────────
    if fit_fourier and sd["X_srm"] is not None and len(sd["X_srm"]) >= 10:
        lr = LinearRegression()
        lr.fit(sd["X_srm"], sd["y_srm"])
        X_q = np.stack([r["srm"] for r in rows_query])
        y_hat = lr.predict(X_q)
        scores["E:Fourier"] = -y_hat   # 予測 dist が小さいほど high score
    else:
        scores["E:Fourier"] = np.zeros(n_q)

    return scores

# ──────────────────────────────────────────────────
# 3-Stage フィルタ後に各スコアで ranking する関数
# ──────────────────────────────────────────────────
def threestage_then_score(rows_sub, score_name, all_scores, N):
    """3-Stage-25% でフィルタした後に score でランキング"""
    N = int(min(N, len(rows_sub)))
    all_d = np.array([r["dist"] for r in rows_sub])
    thr   = float(np.percentile(all_d, 25))

    ct_m  = defaultdict(list)
    for r in rows_sub: ct_m[r["ct_str"]].append(r["dist"])
    kept_cts = {ct for ct, dv in ct_m.items() if float(np.mean(dv)) <= thr}
    if not kept_cts: kept_cts = set(ct_m.keys())

    kept_idx = [i for i,r in enumerate(rows_sub) if r["ct_str"] in kept_cts]
    if not kept_idx: kept_idx = list(range(len(rows_sub)))

    sc_kept = [(i, all_scores[score_name][i]) for i in kept_idx]
    sc_kept.sort(key=lambda x: -x[1])

    selected_idx = [i for i, _ in sc_kept[:N]]
    if len(selected_idx) < N:
        remaining = [i for i in range(len(rows_sub)) if i not in set(selected_idx)]
        remaining.sort(key=lambda i: rows_sub[i]["dist"])
        selected_idx.extend(remaining[:N-len(selected_idx)])

    return {rows_sub[i]["label"] for i in selected_idx[:N]}

def threestage_orig(rows_sub, N):
    """W-7i の 3-Stage-25% (Burauによるランキング)"""
    N = int(min(N, len(rows_sub)))
    all_d = np.array([r["dist"] for r in rows_sub])
    thr   = float(np.percentile(all_d, 25))
    ct_m  = defaultdict(list)
    for r in rows_sub: ct_m[r["ct_str"]].append(r["dist"])
    kept_cts = {ct for ct, v in ct_m.items() if float(np.mean(v)) <= thr}
    if not kept_cts: kept_cts = set(ct_m.keys())
    kept = [r for r in rows_sub if r["ct_str"] in kept_cts]
    ct_g  = defaultdict(list)
    for r in kept: ct_g[r["ct_str"]].append(r)
    selected = []
    for ct, grp in ct_g.items():
        bv = np.array([r["burau"] for r in grp])
        dv = np.array([r["dist"]  for r in grp])
        rr = float(stats.pearsonr(bv, dv)[0]) if len(grp)>=3 else 0.0
        dr = -1 if rr>0 else +1
        n_s = int(max(1, round(N*len(grp)/len(kept))))
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
# 5-fold CV: Precision@N
# ══════════════════════════════════════════════════════════════════
N_vals = [10, 20, 50, 100, 150]
kf = KFold(n_splits=5, shuffle=True, random_state=int(42))

BASE_STRATEGIES = ["GlobalBurau","A:Cayley","B1:pos_E1","B2:OPW_linear","B3:OPW_sq",
                   "C1:Jac_trace","C2:Jac_srad","C3:Jac_frob","C4:Jac_ent","C5:Bur+Jac",
                   "D:Coset_prior","D2:Coset+Bur","E:Fourier"]
COMBO_STRATEGIES = ["3Stg(orig)",
                    "3Stg+Jac_trace","3Stg+Jac_frob","3Stg+Jac_ent",
                    "3Stg+pos_E1","3Stg+OPW","3Stg+Fourier"]
ALL_STRATS = BASE_STRATEGIES + COMBO_STRATEGIES

cv_prec = {s: {N: [] for N in N_vals} for s in ALL_STRATS}

print("\n[5-fold CV Precision@N] 進捗: ", end="", flush=True)
for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(rows)):
    print(f"fold{fold_idx+1} ", end="", flush=True)
    rows_tr = [rows[i] for i in tr_idx]
    rows_te = [rows[i] for i in te_idx]

    sd      = make_scores_from_subset(rows_tr)
    # Fourier: train 用の SRM と dist を差し替えて学習済み状態を作る
    sd_full = copy.copy(sd)
    sd_full["X_srm"] = np.stack([r["srm"] for r in rows_tr])
    sd_full["y_srm"] = np.array([r["dist"] for r in rows_tr])

    all_sc  = apply_scores(rows_tr, rows_te, sd_full, fit_fourier=True)

    for N in N_vals:
        n = int(min(N, len(rows_te)))
        true_top = oracle_labels(rows_te, n)

        # ベース戦略
        for sname in BASE_STRATEGIES:
            sc = all_sc[sname]
            ranked = sorted(range(len(rows_te)), key=lambda i: -sc[i])
            sel = {rows_te[i]["label"] for i in ranked[:n]}
            cv_prec[sname][N].append(precision_at(sel, true_top))

        # 3-Stage (orig)
        cv_prec["3Stg(orig)"][N].append(
            precision_at(threestage_orig(rows_te, n), true_top))

        # 3-Stage + 各スコア
        for combo, score_key in [
            ("3Stg+Jac_trace", "C1:Jac_trace"),
            ("3Stg+Jac_frob",  "C3:Jac_frob"),
            ("3Stg+Jac_ent",   "C4:Jac_ent"),
            ("3Stg+pos_E1",    "B1:pos_E1"),
            ("3Stg+OPW",       "B2:OPW_linear"),
            ("3Stg+Fourier",   "E:Fourier"),
        ]:
            sel = threestage_then_score(rows_te, score_key, all_sc, n)
            cv_prec[combo][N].append(precision_at(sel, true_top))
print("完了")

# ══════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════
print("\n[5-fold CV Precision@N — 全手法]")
SECTIONS = [
    ("── ベースライン",       ["GlobalBurau","A:Cayley","3Stg(orig)"]),
    ("── 方向1: OPW",         ["B1:pos_E1","B2:OPW_linear","B3:OPW_sq"]),
    ("── 方向2: Jacobian",    ["C1:Jac_trace","C2:Jac_srad","C3:Jac_frob",
                                "C4:Jac_ent","C5:Bur+Jac"]),
    ("── 方向3: Coset",       ["D:Coset_prior","D2:Coset+Bur"]),
    ("── 方向4: Fourier",     ["E:Fourier"]),
    ("── 3-Stage × 新スコア", ["3Stg+Jac_trace","3Stg+Jac_frob","3Stg+Jac_ent",
                                "3Stg+pos_E1","3Stg+OPW","3Stg+Fourier"]),
]

hdr = f"  {'手法':<20}|"
for N in N_vals: hdr += f"  @{N:<4}|"
print(hdr)

for sec_title, keys in SECTIONS:
    print(f"\n  {sec_title}")
    print("  " + "-"*(20 + 8*len(N_vals) + 2))
    for key in keys:
        line = f"  {key:<20}|"
        for N in N_vals:
            vals = cv_prec[key][N]
            line += f"  {np.mean(vals):.3f}|"
        # 3-Stage(orig) を上回っていたら★
        v50 = np.mean(cv_prec[key][50])
        ref = np.mean(cv_prec["3Stg(orig)"][50])
        tag = " ★" if v50 > ref + 0.005 else ""
        print(line + tag)

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

# ── 図1: Precision@N カーブ（重要手法のみ）──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("W-7m: New Analytical Approaches — Precision@N (5-fold CV)", fontsize=12)

key_methods_left = ["GlobalBurau","3Stg(orig)",
                    "C1:Jac_trace","C3:Jac_frob","C4:Jac_ent","C5:Bur+Jac",
                    "E:Fourier"]
key_methods_right = ["GlobalBurau","3Stg(orig)",
                     "3Stg+Jac_trace","3Stg+Jac_frob","3Stg+Jac_ent",
                     "3Stg+pos_E1","3Stg+OPW","3Stg+Fourier"]
palettes = {
    "GlobalBurau"    : ("blue",       "o",  "--", 2.0),
    "3Stg(orig)"     : ("green",      "s",  "-",  2.5),
    "B1:pos_E1"      : ("teal",       "^",  "-",  1.8),
    "B2:OPW_linear"  : ("darkcyan",   "v",  "-",  1.8),
    "B3:OPW_sq"      : ("cyan",       "<",  "-",  1.5),
    "C1:Jac_trace"   : ("red",        "D",  "-",  1.8),
    "C2:Jac_srad"    : ("salmon",     "x",  "-",  1.5),
    "C3:Jac_frob"    : ("tomato",     "+",  "-",  1.8),
    "C4:Jac_ent"     : ("crimson",    "*",  "-",  2.0),
    "C5:Bur+Jac"     : ("darkred",    "P",  "-",  1.8),
    "D:Coset_prior"  : ("purple",     "h",  "-",  1.8),
    "D2:Coset+Bur"   : ("violet",     "1",  "-",  1.5),
    "E:Fourier"      : ("darkorange", "2",  "-",  2.0),
    "3Stg+Jac_trace" : ("firebrick",  "D",  "-",  2.2),
    "3Stg+Jac_frob"  : ("orangered",  "P",  "-",  2.0),
    "3Stg+Jac_ent"   : ("darkred",    "*",  "-",  2.2),
    "3Stg+pos_E1"    : ("teal",       "^",  "-",  2.0),
    "3Stg+OPW"       : ("darkcyan",   "v",  "-",  2.0),
    "3Stg+Fourier"   : ("chocolate",  "2",  "-",  2.0),
}

for ax, key_list, title in [
    (axes[0], key_methods_left,  "Individual Strategies"),
    (axes[1], key_methods_right, "3-Stage × New Score (Combinations)"),
]:
    for key in key_list:
        if key not in palettes: continue
        c, mk, ls, lw = palettes[key]
        means = [np.mean(cv_prec[key][N]) for N in N_vals]
        stds  = [np.std(cv_prec[key][N])  for N in N_vals]
        ax.plot(N_vals, means, marker=mk, ls=ls, color=c, lw=lw,
                label=key, alpha=0.9)
        ax.fill_between(N_vals, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], color=c, alpha=0.1)
    ax.set_xlabel("N"); ax.set_ylabel("Precision@N")
    ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(OUT / "W7m_fig1_precision.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig1 saved: {OUT / 'W7m_fig1_precision.png'}")

# ── 図2: Precision@50 バーチャート（全手法） ─────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
N_bar   = 50
all_keys_bar = [s for sec_title, keys in SECTIONS for s in keys]
bar_vals = [float(np.mean(cv_prec[k][N_bar])) for k in all_keys_bar]
bar_errs = [float(np.std(cv_prec[k][N_bar]))  for k in all_keys_bar]
sec_colors = {
    "ベースライン"       : ["#2196F3","#888","#4CAF50"],
    "方向1: OPW"         : ["#009688","#00796B","#004D40"],
    "方向2: Jacobian"    : ["#F44336","#EF9A9A","#FF5722","#B71C1C","#7F0000"],
    "方向3: Coset"       : ["#9C27B0","#6A1B9A"],
    "方向4: Fourier"     : ["#FF6F00"],
    "3-Stage × 新スコア" : ["#D32F2F","#C62828","#B71C1C","#009688","#00796B","#E65100"],
}
bar_c = []
for sec_title, keys in SECTIONS:
    sc_list = sec_colors.get(sec_title.replace("── ",""), ["#888"]*10)
    bar_c.extend(sc_list[:len(keys)])

bars = ax.bar(range(len(all_keys_bar)), bar_vals, color=bar_c[:len(all_keys_bar)],
              yerr=bar_errs, capsize=2, alpha=0.85)
ax.set_xticks(range(len(all_keys_bar)))
ax.set_xticklabels(all_keys_bar, rotation=45, ha='right', fontsize=8)
ax.set_ylabel(f"Precision@{N_bar} (5-fold CV)")
ax.set_title(f"W-7m: All Methods — Precision@{N_bar}")
ref_3stg = float(np.mean(cv_prec["3Stg(orig)"][N_bar]))
ref_gb   = float(np.mean(cv_prec["GlobalBurau"][N_bar]))
ax.axhline(ref_3stg, ls="-",  color="#4CAF50", alpha=0.7, lw=1.5, label=f"3-Stage({ref_3stg:.3f})")
ax.axhline(ref_gb,   ls="--", color="#2196F3", alpha=0.7, lw=1.5, label=f"GlobalBurau({ref_gb:.3f})")
for i, v in enumerate(bar_vals):
    ax.text(i, v+0.01, f"{v:.3f}", ha='center', fontsize=6.5, rotation=45)
ax.legend(fontsize=9); ax.set_ylim(0, 1.15)
plt.tight_layout()
fig.savefig(OUT / "W7m_fig2_bar.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig2 saved: {OUT / 'W7m_fig2_bar.png'}")

# ── 図3: 相関マップ（Jacobian × dist / cycle_type 色分け）────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("W-7m: Jacobian Features vs Landscape Distance", fontsize=12)

jac_plot = [("jac_trace","C1:Jac_trace"), ("jac_frob","C3:Jac_frob"),
             ("jac_entropy","C4:Jac_ent")]
ct_color_map = {"[1, 1, 1, 1, 1, 1]":"#1565C0","[2, 1, 1, 1, 1]":"#2196F3",
                "[3, 1, 1, 1]":"#64B5F6","[2, 2, 1, 1]":"#81C784",
                "[4, 1, 1]":"#4CAF50","[3, 2, 1]":"#FFF176",
                "[5, 1]":"#FFB300","[4, 2]":"#FF7043",
                "[3, 3]":"#E53935","[6]":"#B71C1C","[2, 2, 2]":"#880E4F"}

for ax, (feat, score_name) in zip(axes, jac_plot):
    fv   = np.array([float(r[feat]) for r in rows])
    dv   = dists
    cv_r = corr_new.get(feat, 0.0)
    colors_scatter = [ct_color_map.get(r["ct_str"], "#888") for r in rows]
    ax.scatter(fv, dv, c=colors_scatter, alpha=0.4, s=8)
    ax.set_xlabel(feat); ax.set_ylabel("dist")
    p50 = float(np.mean(cv_prec[score_name][50]))
    ax.set_title(f"{feat}\nr={cv_r:.3f}  P@50={p50:.3f}")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(OUT / "W7m_fig3_jac_scatter.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig3 saved: {OUT / 'W7m_fig3_jac_scatter.png'}")

# ══════════════════════════════════════════════════════════════════
# 総合サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  W-7m 総合サマリー")
print("=" * 65)

ref_n = 50
all_results = sorted(
    [(k, float(np.mean(cv_prec[k][ref_n]))) for k in ALL_STRATS],
    key=lambda x: -x[1])

print(f"\n  Precision@{ref_n} ランキング（5-fold CV）:")
for rank, (key, val) in enumerate(all_results, 1):
    bar = "█" * int(val * 30)
    beat = " ★★★" if val > ref_3stg + 0.005 else (" ★" if val > ref_gb + 0.005 else "")
    print(f"  {rank:2}. {key:<22}: {val:.4f}  {bar}{beat}")

best_new = max((k for k in ALL_STRATS if k not in ("3Stg(orig)","GlobalBurau")),
               key=lambda k: float(np.mean(cv_prec[k][ref_n])))
val_best = float(np.mean(cv_prec[best_new][ref_n]))
val_3stg = float(np.mean(cv_prec["3Stg(orig)"][ref_n]))

print(f"\n  ★ 新手法 最良: {best_new}  P@{ref_n}={val_best:.4f}")
print(f"    3-Stage(orig):              P@{ref_n}={val_3stg:.4f}")
if val_best > val_3stg:
    print(f"  → 3-Stage を {(val_best-val_3stg)*100:.1f}pt 上回った！")
else:
    print(f"  → 差 = {(val_3stg-val_best)*100:.1f}pt  (3-Stageが依然優位)")

# Jacobian の発見
best_jac = max(("C1:Jac_trace","C2:Jac_srad","C3:Jac_frob","C4:Jac_ent","C5:Bur+Jac"),
               key=lambda k: float(np.mean(cv_prec[k][ref_n])))
val_bj   = float(np.mean(cv_prec[best_jac][ref_n]))
print(f"\n  Jacobian 系 最良: {best_jac}  P@{ref_n}={val_bj:.4f}")
for jk in ["jac_trace","jac_frob","jac_entropy"]:
    rv = corr_new.get(jk, 0.0)
    print(f"    {jk:<16}: r={rv:+.4f}")

print("\n" + "=" * 65)
print(f"  W-7m 実験完了  出力先: {OUT}")
print("=" * 65)
