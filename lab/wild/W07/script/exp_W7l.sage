#!/usr/bin/env sage
# exp_W7l.sage — 解析的スクリーニング拡張
# A: Cayley 距離単体
# B: Burau + Cayley 正規化（線形・対数ペナルティ、λ解析的導出）
# C: スペクトル Burau（最大固有値・Frobenius ノルム）
# D: 信頼度重み付き合成スコア（3-Stage のソフト版）
# E: Borda 集計（複数弱予測子のランキング集約）
# 全手法 vs 既存ベースライン（GlobalBurau / 3-Stage-25% / Oracle）

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms

print("=" * 65)
print("  W-7l: 解析的スクリーニング拡張（A〜E 全手法）")
print("  A: Cayley距離  B: 正規化Burau  C: スペクトルBurau")
print("  D: 信頼度合成  E: Borda集計")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7l")
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

# ══════════════════════════════════════════════════════════════════
# データ構築: K=6 + スペクトル特徴量
# ══════════════════════════════════════════════════════════════════
K = 6
print(f"\n[K={K}] 全 {factorial(K)} 件のデータ構築中（スペクトル特徴量含む）...")
BK = BraidGroup(K);  gK = list(BK.generators())
rows = []
for perm in iter_perms(range(1, K+1)):
    sp   = SagePerm(list(perm))
    b    = BK.one()
    for i in sp.reduced_word(): b = b * gK[i-1]
    # Burau 行列（t=1/2）
    BM   = b.burau_matrix()(t=QQ(1)/QQ(2))
    BM_np = np.array([[float(BM[i][j]) for j in range(BM.ncols())]
                       for i in range(BM.nrows())])
    btr  = float(BM_np.trace())
    eigvals = np.linalg.eigvals(BM_np)
    spec_rad  = float(np.max(np.abs(eigvals)))
    frob_norm = float(np.linalg.norm(BM_np, 'fro'))
    eig_var   = float(np.var(np.abs(eigvals)))

    ct   = list(sp.cycle_type())
    dist = compute_dist([x-1 for x in perm], K)
    cayley = K - len(ct)          # K - n_cycles

    rows.append({
        "label"       : "".join(str(x) for x in perm),
        "burau"       : btr,
        "writhe"      : int(len(sp.reduced_word())),
        "max_cycle"   : int(max(ct)),
        "n_cycles"    : len(ct),
        "n_fixed"     : int(ct.count(1)),
        "cayley"      : cayley,
        "spec_rad"    : spec_rad,
        "frob"        : frob_norm,
        "eig_var"     : eig_var,
        "ct_str"      : str(ct),
        "dist"        : dist,
    })
dists = np.array([r["dist"] for r in rows])
print(f"  完了。dist: [{dists.min():.3f}, {dists.max():.3f}]  mean={dists.mean():.3f}")

# ══════════════════════════════════════════════════════════════════
# 各特徴量と dist の相関確認
# ══════════════════════════════════════════════════════════════════
print("\n[各特徴量と dist の Pearson 相関]")
features_check = ["burau","writhe","cayley","max_cycle","n_cycles","n_fixed",
                  "spec_rad","frob","eig_var"]
corr_dict = {}
for fn in features_check:
    fv = np.array([float(r[fn]) for r in rows])
    r_val, p_val = stats.pearsonr(fv, dists)
    corr_dict[fn] = float(r_val)
    bar = "█" * int(abs(r_val) * 30)
    sign = "+" if r_val >= 0 else "-"
    print(f"  {fn:<14}: r={r_val:+.4f}  p={p_val:.2e}  {sign}{bar}")

# ══════════════════════════════════════════════════════════════════
# スコアリング関数の定義
# ══════════════════════════════════════════════════════════════════
def normalize(arr):
    s = arr.std()
    return (arr - arr.mean()) / s if s > 1e-10 else arr * 0.0

# cycle type ごとの統計を事前計算
ct_stats = defaultdict(lambda: {"dists": [], "buraus": []})
for r in rows:
    ct_stats[r["ct_str"]]["dists"].append(r["dist"])
    ct_stats[r["ct_str"]]["buraus"].append(r["burau"])
ct_info = {}
for ct, v in ct_stats.items():
    dv = np.array(v["dists"])
    bv = np.array(v["buraus"])
    rr = float(stats.pearsonr(bv, dv)[0]) if len(dv) >= 3 else 0.0
    ct_info[ct] = {
        "mean_dist" : float(dv.mean()),
        "r"         : rr,
        "direction" : -1 if rr > 0 else +1,
        "n"         : len(dv),
    }

# 全体 dist の正規化パラメータ
dist_mean = float(dists.mean())
dist_std  = float(dists.std())

# ── A: Cayley 距離 ──────────────────────────────────────────────
def score_cayley(r):
    """小さいほど良い (恒等置換に近い)"""
    return -float(r["cayley"])

# ── B1: Burau + 線形 Cayley ペナルティ ─────────────────────────
burau_arr  = np.array([r["burau"]  for r in rows])
cayley_arr = np.array([float(r["cayley"]) for r in rows])
r_bur  = corr_dict["burau"]
r_cay  = corr_dict["cayley"]
# λ: 両変数の dist 説明力の比（相関強度比）× スケール補正
# Score = dir_bur × Burau_norm + λ × dir_cay × Cayley_norm
# dir = sign that makes higher score = lower dist
dir_bur = -1.0 if r_bur > 0 else +1.0
dir_cay = -1.0 if r_cay > 0 else +1.0
# λ = |r_cayley| / |r_burau| なら等比重, ここでは比率で自動決定
lam_linear = abs(r_cay) / (abs(r_bur) + 1e-8)
bur_n  = normalize(burau_arr)
cay_n  = normalize(cayley_arr)
print(f"\n[B: 正規化 Burau] λ_linear={lam_linear:.4f}  "
      f"(r_burau={r_bur:.3f}, r_cayley={r_cay:.3f})")

def score_B_linear(r, bur_n=bur_n, cay_n=cay_n):
    idx = next(i for i,x in enumerate(rows) if x["label"]==r["label"])
    return dir_bur * bur_n[idx] + lam_linear * dir_cay * cay_n[idx]

# ── B2: Burau + 対数 Cayley ペナルティ ─────────────────────────
log_cay_arr = np.log1p(cayley_arr)
r_logcay = float(stats.pearsonr(log_cay_arr, dists)[0])
dir_logcay = -1.0 if r_logcay > 0 else +1.0
lam_log    = abs(r_logcay) / (abs(r_bur) + 1e-8)
log_cay_n  = normalize(log_cay_arr)
print(f"  λ_log={lam_log:.4f}  (r_logcay={r_logcay:.3f})")

def score_B_log(r, bur_n=bur_n, log_cay_n=log_cay_n):
    idx = next(i for i,x in enumerate(rows) if x["label"]==r["label"])
    return dir_bur * bur_n[idx] + lam_log * dir_logcay * log_cay_n[idx]

# ── B3: 等重み合成 (λ=1) ─────────────────────────────────────
def score_B_equal(r, bur_n=bur_n, cay_n=cay_n):
    idx = next(i for i,x in enumerate(rows) if x["label"]==r["label"])
    return dir_bur * bur_n[idx] + dir_cay * cay_n[idx]

# ── C1: 最大固有値（スペクトル半径） ────────────────────────────
spec_arr = np.array([r["spec_rad"] for r in rows])
r_spec   = corr_dict["spec_rad"]
dir_spec = -1.0 if r_spec > 0 else +1.0

def score_C_specrad(r): return dir_spec * r["spec_rad"]

# ── C2: Frobenius ノルム ─────────────────────────────────────
frob_arr = np.array([r["frob"] for r in rows])
r_frob   = corr_dict["frob"]
dir_frob = -1.0 if r_frob > 0 else +1.0

def score_C_frob(r): return dir_frob * r["frob"]

# ── C3: 固有値分散 ──────────────────────────────────────────
eig_arr = np.array([r["eig_var"] for r in rows])
r_eig   = corr_dict["eig_var"]
dir_eig = -1.0 if r_eig > 0 else +1.0

def score_C_eigvar(r): return dir_eig * r["eig_var"]

# ── C4: Burau + スペクトル複合 ──────────────────────────────
lam_spec = abs(r_spec) / (abs(r_bur) + 1e-8)
spec_n   = normalize(spec_arr)

def score_C_burau_spec(r, bur_n=bur_n, spec_n=spec_n):
    idx = next(i for i,x in enumerate(rows) if x["label"]==r["label"])
    return dir_bur * bur_n[idx] + lam_spec * dir_spec * spec_n[idx]

# ── D: 信頼度重み付き合成スコア ──────────────────────────────
# CT_mean_dist を [0,1] に正規化
ct_mean_min = min(v["mean_dist"] for v in ct_info.values())
ct_mean_max = max(v["mean_dist"] for v in ct_info.values())

def ct_prior_score(ct_str):
    """CT mean_dist → 低いほど良い → スコアは負方向に変換"""
    md = ct_info[ct_str]["mean_dist"]
    normalized = (md - ct_mean_min) / (ct_mean_max - ct_mean_min + 1e-8)
    return -normalized  # 低dist型 = 高スコア

# within-type 標準化 Burau
burau_by_ct = defaultdict(list)
for r in rows: burau_by_ct[r["ct_str"]].append(r["burau"])
ct_burau_stats = {ct: (np.mean(v), np.std(v)+1e-8)
                  for ct, v in burau_by_ct.items()}

def burau_within_norm(r):
    mu, sig = ct_burau_stats[r["ct_str"]]
    return (r["burau"] - mu) / sig

def score_D_confidence(r):
    ct  = r["ct_str"]
    abs_r  = abs(ct_info[ct]["r"])       # 信頼度 = |within-type r|
    direc  = float(ct_info[ct]["direction"])
    bur_s  = direc * burau_within_norm(r)  # Burau 成分
    ct_s   = ct_prior_score(ct)             # CT 事前スコア
    return abs_r * bur_s + (1.0 - abs_r) * ct_s

# ── E: Borda 集計 ────────────────────────────────────────────
# 各指標でランキングを作り、順位の和（小さいほど良い）で選択
def make_borda_score(rows, scorers_with_higher_better):
    """scorers_with_higher_better: list of (name, fn, higher_is_better)"""
    n = len(rows)
    borda = np.zeros(n)
    for name, fn, higher in scorers_with_higher_better:
        vals = np.array([float(fn(r)) for r in rows])
        if higher:
            ranks = stats.rankdata(-vals)   # 高いほど順位 1
        else:
            ranks = stats.rankdata(vals)    # 低いほど順位 1
        borda += ranks
    return borda   # 小さいほど良い

# Borda の構成要素: Burau, Cayley, max_cycle, spec_rad, frob
borda_components = [
    ("burau",     lambda r: r["burau"],    dir_bur > 0),
    ("cayley",    lambda r: r["cayley"],   dir_cay > 0),
    ("max_cycle", lambda r: r["max_cycle"],False),   # 小さいほど良い
    ("spec_rad",  lambda r: r["spec_rad"], dir_spec > 0),
    ("frob",      lambda r: r["frob"],     dir_frob > 0),
]
print("\n[E: Borda 集計] 構成要素の方向確認:")
for name, fn, higher in borda_components:
    print(f"  {name:<12}: {'高いほど良い' if higher else '低いほど良い'}")

borda_scores = make_borda_score(rows, borda_components)

def score_E_borda(r):
    idx = next(i for i,x in enumerate(rows) if x["label"]==r["label"])
    return -borda_scores[idx]   # 高いスコア = 良い（負にして統一）

# ── E2: Borda (Burau + Cayley のみ) ─────────────────────────
borda2_scores = make_borda_score(rows, [
    ("burau",  lambda r: r["burau"],  dir_bur > 0),
    ("cayley", lambda r: r["cayley"], dir_cay > 0),
])

def score_E2_borda2(r):
    idx = next(i for i,x in enumerate(rows) if x["label"]==r["label"])
    return -borda2_scores[idx]

# ══════════════════════════════════════════════════════════════════
# 既存ベースライン
# ══════════════════════════════════════════════════════════════════
def global_burau_score(r):
    return dir_bur * r["burau"]

def threestage_labels(rows_sub, N):
    N = int(min(N, len(rows_sub)))
    all_d = np.array([r["dist"] for r in rows_sub])
    thr   = float(np.percentile(all_d, 25))
    ct_m  = defaultdict(list)
    for r in rows_sub: ct_m[r["ct_str"]].append(r["dist"])
    kept_cts = {ct for ct, dv in ct_m.items() if np.mean(dv) <= thr}
    if not kept_cts: kept_cts = set(ct_m.keys())
    kept = [r for r in rows_sub if r["ct_str"] in kept_cts]
    ct_g  = defaultdict(list)
    for r in kept: ct_g[r["ct_str"]].append(r)
    selected = []
    for ct, grp in ct_g.items():
        bv = np.array([r["burau"] for r in grp])
        dv = np.array([r["dist"]  for r in grp])
        rr = float(stats.pearsonr(bv, dv)[0]) if len(grp) >= 3 else 0.0
        direc = -1 if rr > 0 else +1
        n_s   = int(max(1, round(N * len(grp) / len(kept))))
        ranked = sorted(grp, key=lambda r: direc * r["burau"], reverse=True)
        selected.extend(ranked[:n_s])
    if len(selected) > N:
        selected = sorted(selected, key=lambda r: r["dist"])[:N]
    elif len(selected) < N:
        sl = {r["label"] for r in selected}
        rem = sorted([r for r in rows_sub if r["label"] not in sl],
                     key=lambda r: r["dist"])
        selected.extend(rem[:N-len(selected)])
    return {r["label"] for r in selected[:N]}

# ══════════════════════════════════════════════════════════════════
# 全スコアリング戦略を辞書化
# ══════════════════════════════════════════════════════════════════
STRATEGIES = {
    # ── 既存ベースライン ──
    "GlobalBurau"  : global_burau_score,
    # 3-Stage は別途 threestage_labels() で処理
    # ── A ──
    "A:Cayley"     : score_cayley,
    # ── B ──
    "B1:Bur+Cay(r)": score_B_linear,
    "B2:Bur+Cay(log)": score_B_log,
    "B3:Bur+Cay(eq)": score_B_equal,
    # ── C ──
    "C1:SpecRad"   : score_C_specrad,
    "C2:Frob"      : score_C_frob,
    "C3:EigVar"    : score_C_eigvar,
    "C4:Bur+Spec"  : score_C_burau_spec,
    # ── D ──
    "D:Confidence" : score_D_confidence,
    # ── E ──
    "E1:Borda(5)"  : score_E_borda,
    "E2:Borda(2)"  : score_E2_borda2,
}

def select_top_N(rows_sub, score_fn, N):
    N = int(min(N, len(rows_sub)))
    ranked = sorted(rows_sub, key=score_fn, reverse=True)
    return {r["label"] for r in ranked[:N]}

def oracle_labels(rows_sub, N):
    N = int(min(N, len(rows_sub)))
    return {r["label"] for r in sorted(rows_sub, key=lambda r: r["dist"])[:N]}

def precision_at(sel, true): return float(len(sel & true)) / float(len(true))

# ══════════════════════════════════════════════════════════════════
# Precision@N（全データ・ノーCVで構造確認）
# ══════════════════════════════════════════════════════════════════
print("\n[Precision@N — 全データ版（構造確認）]")
N_vals = [10, 20, 50, 100, 150]
hdr = f"  {'手法':<18}|"
for N in N_vals: hdr += f"  @{N:<4}|"
print(hdr);  print("  " + "-"*(18 + 8*len(N_vals) + 2))

all_prec_full = {}
for name, sfn in STRATEGIES.items():
    row_str = f"  {name:<18}|"
    all_prec_full[name] = []
    for N in N_vals:
        true_top = oracle_labels(rows, N)
        sel = select_top_N(rows, sfn, N)
        p = precision_at(sel, true_top)
        all_prec_full[name].append(p)
        row_str += f"  {p:.3f}|"
    print(row_str)

# 3-Stage
row_str = f"  {'3Stg-25%':<18}|"
all_prec_full["3Stg-25%"] = []
for N in N_vals:
    true_top = oracle_labels(rows, N)
    sel = threestage_labels(rows, N)
    p = precision_at(sel, true_top)
    all_prec_full["3Stg-25%"].append(p)
    row_str += f"  {p:.3f}|"
print(row_str)

# ══════════════════════════════════════════════════════════════════
# 5-fold CV: Precision@N（汎化性能）
# ══════════════════════════════════════════════════════════════════
from sklearn.model_selection import KFold

print("\n[Precision@N — 5-fold CV（汎化性能）]")
kf = KFold(n_splits=5, shuffle=True, random_state=int(42))

cv_prec = {name: {N: [] for N in N_vals}
           for name in list(STRATEGIES.keys()) + ["3Stg-25%"]}

# Borda スコアは「全データのランク」に依存するため fold ごとに再計算
def make_borda_score_subset(rows_sub, comps):
    n = len(rows_sub)
    borda = np.zeros(n)
    for _, fn, higher in comps:
        vals = np.array([float(fn(r)) for r in rows_sub])
        borda += stats.rankdata(-vals if higher else vals)
    return borda

# スコア関数のうち「全データ統計に依存する」もの → fold ごとに再構築が必要
# B 系, C4, D, E 系は全体の normalize/rank を使っているため
# ここでは「fold の test 集合に対してのみ score を適用」とし
# A (Cayley), C1/C2/C3 は local（各要素のみ）なので fold 非依存

LOCAL_STRATEGIES = ["A:Cayley","C1:SpecRad","C2:Frob","C3:EigVar","GlobalBurau"]

print("  進捗: ", end="", flush=True)
for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(rows)):
    print(f"fold{fold_idx+1} ", end="", flush=True)
    rows_te = [rows[i] for i in te_idx]
    rows_tr = [rows[i] for i in tr_idx]

    # fold ごとに統計を再計算（汎化的に正しい）
    bur_te  = np.array([r["burau"]  for r in rows_te])
    cay_te  = np.array([float(r["cayley"]) for r in rows_te])
    spec_te = np.array([r["spec_rad"] for r in rows_te])
    frob_te = np.array([r["frob"]    for r in rows_te])
    dist_te = np.array([r["dist"]    for r in rows_te])

    # train fold の相関から方向・λ を決める（真の汎化設定）
    bur_tr  = np.array([r["burau"]    for r in rows_tr])
    cay_tr  = np.array([float(r["cayley"]) for r in rows_tr])
    dist_tr = np.array([r["dist"]     for r in rows_tr])
    lcay_tr = np.log1p(cay_tr)
    spec_tr = np.array([r["spec_rad"] for r in rows_tr])

    r_bur_f  = float(stats.pearsonr(bur_tr, dist_tr)[0])
    r_cay_f  = float(stats.pearsonr(cay_tr, dist_tr)[0])
    r_lcay_f = float(stats.pearsonr(lcay_tr, dist_tr)[0])
    r_spec_f = float(stats.pearsonr(spec_tr, dist_tr)[0])
    d_bur_f  = -1.0 if r_bur_f  > 0 else +1.0
    d_cay_f  = -1.0 if r_cay_f  > 0 else +1.0
    d_lcay_f = -1.0 if r_lcay_f > 0 else +1.0
    d_spec_f = -1.0 if r_spec_f > 0 else +1.0
    lam_lin_f = abs(r_cay_f)  / (abs(r_bur_f) + 1e-8)
    lam_log_f = abs(r_lcay_f) / (abs(r_bur_f) + 1e-8)
    lam_spc_f = abs(r_spec_f) / (abs(r_bur_f) + 1e-8)

    def norm_f(arr):
        s = arr.std(); return (arr-arr.mean())/s if s>1e-8 else arr*0.0

    bur_n_f  = norm_f(bur_te)
    cay_n_f  = norm_f(cay_te)
    lcay_n_f = norm_f(np.log1p(cay_te))
    spec_n_f = norm_f(spec_te)

    # CT 統計を train から学習
    ct_m_tr = defaultdict(list); ct_b_tr = defaultdict(list)
    for r in rows_tr:
        ct_m_tr[r["ct_str"]].append(r["dist"])
        ct_b_tr[r["ct_str"]].append(r["burau"])
    ct_info_f = {}
    for ct in ct_m_tr:
        dv = np.array(ct_m_tr[ct]); bv = np.array(ct_b_tr[ct])
        rr = float(stats.pearsonr(bv, dv)[0]) if len(dv) >= 3 else 0.0
        ct_info_f[ct] = {"mean_dist": float(dv.mean()), "r": rr,
                          "direction": -1 if rr > 0 else +1}
    if ct_info_f:
        cmn = min(v["mean_dist"] for v in ct_info_f.values())
        cmx = max(v["mean_dist"] for v in ct_info_f.values())
    else:
        cmn, cmx = 0.0, 1.0

    def ct_ps_f(ct_str):
        if ct_str not in ct_info_f: return 0.0
        md = ct_info_f[ct_str]["mean_dist"]
        return -((md - cmn) / (cmx - cmn + 1e-8))

    bur_by_ct_f = defaultdict(list)
    for r in rows_tr: bur_by_ct_f[r["ct_str"]].append(r["burau"])
    ct_bst_f = {ct: (np.mean(v), np.std(v)+1e-8) for ct, v in bur_by_ct_f.items()}

    # fold スコア関数群
    def sc_Blin(r, idx):
        return d_bur_f*bur_n_f[idx] + lam_lin_f*d_cay_f*cay_n_f[idx]
    def sc_Blog(r, idx):
        return d_bur_f*bur_n_f[idx] + lam_log_f*d_lcay_f*lcay_n_f[idx]
    def sc_Beq(r, idx):
        return d_bur_f*bur_n_f[idx] + d_cay_f*cay_n_f[idx]
    def sc_BSpec(r, idx):
        return d_bur_f*bur_n_f[idx] + lam_spc_f*d_spec_f*spec_n_f[idx]

    def sc_D(r, idx):
        ct = r["ct_str"]
        if ct not in ct_info_f: return 0.0
        abs_r = abs(ct_info_f[ct]["r"])
        direc = float(ct_info_f[ct]["direction"])
        mu, sig = ct_bst_f.get(ct, (0.0, 1.0))
        bur_s = direc * (r["burau"] - mu) / sig
        return abs_r * bur_s + (1.0 - abs_r) * ct_ps_f(ct)

    # Borda on test fold
    borda5_f = make_borda_score_subset(rows_te, [
        ("burau",    lambda r: r["burau"],    d_bur_f  > 0),
        ("cayley",   lambda r: r["cayley"],   d_cay_f  > 0),
        ("max_cyc",  lambda r: r["max_cycle"],False),
        ("spec_rad", lambda r: r["spec_rad"], d_spec_f > 0),
        ("frob",     lambda r: r["frob"],     False),  # 一般に小さいほど良い傾向
    ])
    borda2_f = make_borda_score_subset(rows_te, [
        ("burau",  lambda r: r["burau"],  d_bur_f > 0),
        ("cayley", lambda r: r["cayley"], d_cay_f > 0),
    ])

    FOLD_SCORERS = {
        "GlobalBurau"   : lambda r, i: d_bur_f * r["burau"],
        "A:Cayley"      : lambda r, i: -float(r["cayley"]),
        "B1:Bur+Cay(r)" : sc_Blin,
        "B2:Bur+Cay(log)":sc_Blog,
        "B3:Bur+Cay(eq)": sc_Beq,
        "C1:SpecRad"    : lambda r, i: d_spec_f * r["spec_rad"],
        "C2:Frob"       : lambda r, i: d_frob_f * r["frob"] if False else -r["frob"],
        "C3:EigVar"     : lambda r, i: dir_eig * r["eig_var"],
        "C4:Bur+Spec"   : sc_BSpec,
        "D:Confidence"  : sc_D,
        "E1:Borda(5)"   : lambda r, i: -borda5_f[i],
        "E2:Borda(2)"   : lambda r, i: -borda2_f[i],
    }

    for N in N_vals:
        n = int(min(N, len(rows_te)))
        true_top = oracle_labels(rows_te, n)

        # 3-Stage
        cv_prec["3Stg-25%"][N].append(
            precision_at(threestage_labels(rows_te, n), true_top))

        for name, sfn_f in FOLD_SCORERS.items():
            ranked = sorted(enumerate(rows_te),
                            key=lambda x: sfn_f(x[1], x[0]), reverse=True)
            sel = {rows_te[i]["label"] for i, _ in ranked[:n]}
            cv_prec[name][N].append(precision_at(sel, true_top))

print("完了")

# ══════════════════════════════════════════════════════════════════
# 結果表示
# ══════════════════════════════════════════════════════════════════
print("\n[5-fold CV Precision@N — 全手法比較]")
SECTIONS = [
    ("── 既存ベースライン",   ["GlobalBurau","3Stg-25%"]),
    ("── A: Cayley",          ["A:Cayley"]),
    ("── B: 正規化Burau",     ["B1:Bur+Cay(r)","B2:Bur+Cay(log)","B3:Bur+Cay(eq)"]),
    ("── C: スペクトルBurau", ["C1:SpecRad","C2:Frob","C3:EigVar","C4:Bur+Spec"]),
    ("── D: 信頼度合成",      ["D:Confidence"]),
    ("── E: Borda集計",       ["E1:Borda(5)","E2:Borda(2)"]),
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
        print(line)

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

# ── 図1: 相関係数サマリー ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
feat_names_plot = list(corr_dict.keys())
feat_corrs = [corr_dict[f] for f in feat_names_plot]
colors_bar = []
for f in feat_names_plot:
    if f in ("burau","spec_rad","frob","eig_var"): colors_bar.append("#2196F3")
    elif f in ("cayley","max_cycle","n_cycles","n_fixed"): colors_bar.append("#4CAF50")
    else: colors_bar.append("#FF9800")
bars = ax.bar(feat_names_plot, feat_corrs, color=colors_bar, alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel("Pearson r (feature vs dist)")
ax.set_title("W-7l: Feature Correlations with Landscape Distance")
ax.set_xticklabels(feat_names_plot, rotation=30, ha='right')
for b, v in zip(bars, feat_corrs):
    ax.text(b.get_x()+b.get_width()/2, v+(0.01 if v>=0 else -0.03),
            f"{v:.3f}", ha='center', fontsize=8)
from matplotlib.patches import Patch
leg = [Patch(facecolor="#2196F3", label="Burau/Spectral"),
       Patch(facecolor="#4CAF50", label="Cycle type / Cayley"),
       Patch(facecolor="#FF9800", label="Writhe")]
ax.legend(handles=leg, fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "W7l_fig1_correlations.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig1 saved: {OUT / 'W7l_fig1_correlations.png'}")

# ── 図2: Precision@N カーブ（グループ別） ───────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("W-7l: Precision@N — All Analytical Approaches (5-fold CV)", fontsize=13)

palette = {
    "GlobalBurau"    : ("blue",     "o",  "--", 2.5),
    "3Stg-25%"       : ("green",    "s",  "-",  2.5),
    "A:Cayley"       : ("teal",     "^",  "-",  2.0),
    "B1:Bur+Cay(r)"  : ("purple",   "D",  "-",  1.8),
    "B2:Bur+Cay(log)": ("violet",   "v",  "-",  1.8),
    "B3:Bur+Cay(eq)" : ("indigo",   "P",  "-",  1.8),
    "C1:SpecRad"     : ("red",      "h",  "-",  1.8),
    "C2:Frob"        : ("salmon",   "x",  "-",  1.5),
    "C3:EigVar"      : ("tomato",   "+",  "-",  1.5),
    "C4:Bur+Spec"    : ("crimson",  "1",  "-",  1.8),
    "D:Confidence"   : ("darkorange","*", "-",  2.2),
    "E1:Borda(5)"    : ("brown",    "2",  "-",  1.8),
    "E2:Borda(2)"    : ("sienna",   "3",  "-",  1.8),
}

group_plots = [
    ("Baseline",    ["GlobalBurau","3Stg-25%"],             axes[0,0]),
    ("A: Cayley",   ["GlobalBurau","3Stg-25%","A:Cayley"],  axes[0,1]),
    ("B: Reg Burau",["GlobalBurau","3Stg-25%",
                     "B1:Bur+Cay(r)","B2:Bur+Cay(log)","B3:Bur+Cay(eq)"], axes[0,2]),
    ("C: Spectral", ["GlobalBurau","3Stg-25%",
                     "C1:SpecRad","C2:Frob","C3:EigVar","C4:Bur+Spec"],    axes[1,0]),
    ("D: Confidence",["GlobalBurau","3Stg-25%","D:Confidence"],            axes[1,1]),
    ("E: Borda",    ["GlobalBurau","3Stg-25%",
                     "E1:Borda(5)","E2:Borda(2)"],                         axes[1,2]),
]
for title, keys, ax in group_plots:
    for key in keys:
        c, mk, ls, lw = palette[key]
        means = [np.mean(cv_prec[key][N]) for N in N_vals]
        stds  = [np.std(cv_prec[key][N])  for N in N_vals]
        ax.plot(N_vals, means, marker=mk, ls=ls, color=c, lw=lw,
                label=key, alpha=0.9)
        ax.fill_between(N_vals, [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)], color=c, alpha=0.1)
    ax.set_title(title)
    ax.set_xlabel("N"); ax.set_ylabel("Precision@N")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(OUT / "W7l_fig2_precision_groups.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig2 saved: {OUT / 'W7l_fig2_precision_groups.png'}")

# ── 図3: Precision@50 全手法バーチャート ─────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
N_bar = 50
all_keys_ord = (["GlobalBurau","3Stg-25%","A:Cayley",
                  "B1:Bur+Cay(r)","B2:Bur+Cay(log)","B3:Bur+Cay(eq)",
                  "C1:SpecRad","C2:Frob","C3:EigVar","C4:Bur+Spec",
                  "D:Confidence","E1:Borda(5)","E2:Borda(2)"])
bar_vals = [float(np.mean(cv_prec[k][N_bar])) for k in all_keys_ord]
bar_errs = [float(np.std(cv_prec[k][N_bar]))  for k in all_keys_ord]
bar_col  = (["#2196F3","#4CAF50",             # baseline
              "#009688",                        # A
              "#9C27B0","#7B1FA2","#6A1B9A",   # B
              "#F44336","#EF9A9A","#FFCDD2","#B71C1C",  # C
              "#FF6F00",                        # D
              "#795548","#4E342E"])             # E
bars = ax.bar(range(len(all_keys_ord)), bar_vals, color=bar_col,
              yerr=bar_errs, capsize=3, alpha=0.85)
ax.set_xticks(range(len(all_keys_ord)))
ax.set_xticklabels(all_keys_ord, rotation=40, ha='right', fontsize=9)
ax.set_ylabel(f"Precision@{N_bar} (5-fold CV)")
ax.set_title(f"W-7l: All Analytical Methods — Precision@{N_bar}")
ax.axhline(float(np.mean(cv_prec["GlobalBurau"][N_bar])), ls="--",
           color="#2196F3", alpha=0.5, label="GlobalBurau")
ax.axhline(float(np.mean(cv_prec["3Stg-25%"][N_bar])), ls="--",
           color="#4CAF50", alpha=0.5, label="3-Stage-25%")
for i, v in enumerate(bar_vals):
    ax.text(i, v+0.01, f"{v:.3f}", ha='center', fontsize=7.5)
ax.legend(fontsize=9); ax.set_ylim(0, 1.1)
# セクション区切り
for x, lbl in [(2.5,"A"),(3.5,"B"),(7.5,"C"),(11.5,"D"),(12.5,"E")]:
    ax.axvline(x, color='gray', ls=':', alpha=0.4)
plt.tight_layout()
fig.savefig(OUT / "W7l_fig3_bar_summary.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig3 saved: {OUT / 'W7l_fig3_bar_summary.png'}")

# ══════════════════════════════════════════════════════════════════
# 総合サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  総合サマリー（W-7l）")
print("=" * 65)

ref_n = 50
ranking = sorted(
    [(k, float(np.mean(cv_prec[k][ref_n]))) for k in all_keys_ord],
    key=lambda x: -x[1])

print(f"\n  Precision@{ref_n} ランキング（5-fold CV）:")
for rank, (key, val) in enumerate(ranking, 1):
    bar = "█" * int(val * 30)
    print(f"  {rank:2}. {key:<22}: {val:.4f}  {bar}")

best_new = max((k for k in all_keys_ord if k not in ("GlobalBurau","3Stg-25%")),
               key=lambda k: float(np.mean(cv_prec[k][ref_n])))
val_best_new = float(np.mean(cv_prec[best_new][ref_n]))
val_3stg     = float(np.mean(cv_prec["3Stg-25%"][ref_n]))
val_gb       = float(np.mean(cv_prec["GlobalBurau"][ref_n]))

print(f"\n  [新手法の最良: {best_new}  P@{ref_n}={val_best_new:.4f}]")
if val_best_new > val_3stg:
    print(f"  ★ 3-Stage を上回る新しい解析的手法を発見！")
elif val_best_new > val_gb:
    print(f"  → GlobalBurau を超えたが 3-Stage には届かず")
else:
    print(f"  → GlobalBurau にも届かず（既存手法が優位）")

print(f"\n  [Cayley 距離単体 vs Burau]")
val_cay = float(np.mean(cv_prec["A:Cayley"][ref_n]))
val_bur = float(np.mean(cv_prec["GlobalBurau"][ref_n]))
if val_cay > val_bur:
    print(f"  ★ Cayley ({val_cay:.4f}) > GlobalBurau ({val_bur:.4f})")
    print(f"    → cycle type 情報だけで Burau を超えられる")
else:
    print(f"  Cayley ({val_cay:.4f}) < GlobalBurau ({val_bur:.4f})")
    print(f"  → Burau にはトポロジー情報の上乗せがある")

print("\n" + "=" * 65)
print(f"  W-7l 実験完了  出力先: {OUT}")
print("=" * 65)
