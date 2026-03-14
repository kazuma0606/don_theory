#!/usr/bin/env sage
# exp_W7k.sage — ベイズ系モデル追加比較
# GaussianNB (分類) / BayesianRidge / ARD Regression / Gaussian Process
# W-7j の結果に合流させる形で Precision@N を比較

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from pathlib import Path
from sage.combinat.permutation import Permutation as SagePerm
from itertools import permutations as iter_perms
import copy

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

print("=" * 65)
print("  W-7k: ベイズ系モデル vs 解析的スクリーニング")
print("  GaussianNB / BayesianRidge / ARD / GaussianProcess")
print("=" * 65)

OUT = Path("/mnt/c/Users/yoshi/don_theory/lab/wild/results/W7k")
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
# データ構築: K=6
# ══════════════════════════════════════════════════════════════════
K = 6
print(f"\n[K={K}] 全 {factorial(K)} 件のデータ構築中...")
BK = BraidGroup(K);  gK = list(BK.generators())
rows = []
for perm in iter_perms(range(1, K+1)):
    sp   = SagePerm(list(perm))
    b    = BK.one()
    for i in sp.reduced_word(): b = b * gK[i-1]
    btr  = float(b.burau_matrix()(t=QQ(1)/QQ(2)).trace())
    ct   = list(sp.cycle_type())
    dist = compute_dist([x-1 for x in perm], K)
    rows.append({
        "label"         : "".join(str(x) for x in perm),
        "burau"         : btr,
        "writhe"        : len(sp.reduced_word()),
        "max_cycle_len" : int(max(ct)),
        "n_cycles"      : len(ct),
        "n_fixed"       : int(ct.count(1)),
        "ct_str"        : str(ct),
        "dist"          : dist,
    })
dists = np.array([r["dist"] for r in rows])
print(f"  完了。dist: [{dists.min():.3f}, {dists.max():.3f}]  mean={dists.mean():.3f}")

FEATURE_NAMES = ["burau", "writhe", "max_cycle_len", "n_cycles", "n_fixed"]
X_raw = np.array([[r[f] for f in FEATURE_NAMES] for r in rows])
y_reg = dists  # 回帰用: 連続値

# 分類用ラベル: 下位 25th percentile = 1 (良い順序), それ以外 = 0
thr25 = float(np.percentile(y_reg, 25))
y_cls = (y_reg <= thr25).astype(int)
print(f"  分類ラベル: top-25% threshold={thr25:.3f}  "
      f"positive={y_cls.sum()}件 ({y_cls.mean()*100:.0f}%)")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ══════════════════════════════════════════════════════════════════
# ベイズ系モデル定義
# ══════════════════════════════════════════════════════════════════
kernel_rbf    = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.5)
kernel_matern = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.5)

BAYES_MODELS = {
    # 分類系（確率でランキング）
    "GaussianNB"    : ("cls", GaussianNB()),
    # 回帰系
    "BayesianRidge" : ("reg", BayesianRidge()),
    "ARD"           : ("reg", ARDRegression()),
    "GP-RBF"        : ("reg", GaussianProcessRegressor(kernel=kernel_rbf,
                                                        random_state=int(0),
                                                        normalize_y=True)),
    "GP-Matern"     : ("reg", GaussianProcessRegressor(kernel=kernel_matern,
                                                        random_state=int(0),
                                                        normalize_y=True)),
}

# W-7j の比較モデル（再掲用）
PREV_MODELS = {
    "SVR-RBF"  : ("reg", SVR()),
    "RF"       : ("reg", RandomForestRegressor(random_state=int(0))),
    "GBM"      : ("reg", GradientBoostingRegressor(random_state=int(0))),
    "XGBoost"  : ("reg", xgb.XGBRegressor(random_state=int(0), verbosity=int(0))),
}

ALL_MODELS = {**BAYES_MODELS, **PREV_MODELS}

# ══════════════════════════════════════════════════════════════════
# ユーティリティ: Precision@N
# ══════════════════════════════════════════════════════════════════
def oracle_top_labels(rows_subset, N):
    N = int(min(N, len(rows_subset)))
    return {r["label"] for r in sorted(rows_subset, key=lambda r: r["dist"])[:N]}

def precision_at(sel_labels, true_labels):
    return float(len(sel_labels & true_labels)) / float(len(true_labels))

def global_burau_labels(rows_subset, N):
    N = int(min(N, len(rows_subset)))
    bv = np.array([r["burau"] for r in rows_subset])
    dv = np.array([r["dist"]  for r in rows_subset])
    direction = -1 if stats.pearsonr(bv, dv)[0] > 0 else +1
    sel = sorted(rows_subset, key=lambda r: direction * r["burau"], reverse=True)[:N]
    return {r["label"] for r in sel}

def threestage_labels(rows_subset, N):
    """W-7i の 3-Stage-25% をサブセットに適用"""
    N = int(min(N, len(rows_subset)))
    all_d = np.array([r["dist"] for r in rows_subset])
    dist_thr = float(np.percentile(all_d, 25))
    ct_mean = defaultdict(list)
    for r in rows_subset: ct_mean[r["ct_str"]].append(r["dist"])
    kept_cts = {ct for ct, dv in ct_mean.items() if float(np.mean(dv)) <= dist_thr}
    if not kept_cts:
        kept_cts = set(ct_mean.keys())
    kept = [r for r in rows_subset if r["ct_str"] in kept_cts]
    ct_grp = defaultdict(list)
    for r in kept: ct_grp[r["ct_str"]].append(r)
    selected = []
    for ct, grp in ct_grp.items():
        bv = np.array([r["burau"] for r in grp])
        dv = np.array([r["dist"]  for r in grp])
        rr = stats.pearsonr(bv, dv)[0] if len(grp) >= 3 else 0.0
        direction = -1 if rr > 0 else +1
        n_sel = int(max(1, round(N * len(grp) / len(kept))))
        ranked = sorted(grp, key=lambda r: direction * r["burau"], reverse=True)
        selected.extend(ranked[:n_sel])
    if len(selected) > N:
        selected = sorted(selected, key=lambda r: r["dist"])[:N]
    elif len(selected) < N:
        sel_lbl = {r["label"] for r in selected}
        rem = sorted([r for r in rows_subset if r["label"] not in sel_lbl],
                     key=lambda r: r["dist"])
        selected.extend(rem[:N - len(selected)])
    return {r["label"] for r in selected[:N]}

# ══════════════════════════════════════════════════════════════════
# 5-fold CV: Precision@N
# ══════════════════════════════════════════════════════════════════
print("\n[5-fold CV: Precision@N]")
N_vals = [10, 20, 50, 100, 150]
kf = KFold(n_splits=5, shuffle=True, random_state=int(42))

cv_prec = {name: {N: [] for N in N_vals} for name in ALL_MODELS}
cv_prec["GlobalBurau"] = {N: [] for N in N_vals}
cv_prec["3Stg-25%"]    = {N: [] for N in N_vals}

labels_all = [r["label"] for r in rows]

print("  進捗: ", end="", flush=True)
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"fold{fold_idx+1} ", end="", flush=True)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_reg_tr, y_reg_te = y_reg[train_idx], y_reg[test_idx]
    y_cls_tr = y_cls[train_idx]
    rows_te = [rows[i] for i in test_idx]

    for N in N_vals:
        n = int(min(N, len(rows_te)))
        true_top = oracle_top_labels(rows_te, n)

        # 解析的ベースライン
        cv_prec["GlobalBurau"][N].append(
            precision_at(global_burau_labels(rows_te, n), true_top))
        cv_prec["3Stg-25%"][N].append(
            precision_at(threestage_labels(rows_te, n), true_top))

        # 各モデル
        for name, (task, base_model) in ALL_MODELS.items():
            m = copy.deepcopy(base_model)
            if task == "cls":
                m.fit(X_tr, y_cls_tr)
                # P(クラス=1=良い順序) が高い順に選ぶ
                proba = m.predict_proba(X_te)[:, 1]
                top_idx = np.argsort(-proba)[:n]
            else:
                m.fit(X_tr, y_reg_tr)
                y_pred = m.predict(X_te)
                top_idx = np.argsort(y_pred)[:n]   # dist小さい順
            ml_top = {rows_te[i]["label"] for i in top_idx}
            cv_prec[name][N].append(precision_at(ml_top, true_top))
print("完了")

# ══════════════════════════════════════════════════════════════════
# 5-fold CV: R² / AUC（回帰 / 分類精度）
# ══════════════════════════════════════════════════════════════════
print("\n[5-fold CV: 予測精度]")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

r2_results  = {}
auc_results = {}
kf2 = KFold(n_splits=5, shuffle=True, random_state=int(42))

# 線形 Burau ベースライン
lr_r2 = cross_val_score(LinearRegression(), X[:,[0]], y_reg, cv=kf2, scoring="r2")
r2_results["Linear(Burau)"] = (float(lr_r2.mean()), float(lr_r2.std()))

for name, (task, base_model) in ALL_MODELS.items():
    if task == "reg":
        r2 = cross_val_score(copy.deepcopy(base_model), X, y_reg, cv=kf2, scoring="r2")
        r2_results[name] = (float(r2.mean()), float(r2.std()))
    else:
        # 分類: AUC
        aucs = []
        for tr_idx, te_idx in kf2.split(X):
            m = copy.deepcopy(base_model)
            m.fit(X[tr_idx], y_cls[tr_idx])
            proba = m.predict_proba(X[te_idx])[:, 1]
            aucs.append(roc_auc_score(y_cls[te_idx], proba))
        auc_results[name] = (float(np.mean(aucs)), float(np.std(aucs)))

print(f"\n  回帰 R² (dist 予測):")
print(f"  {'モデル':<16} | R² mean±std")
print("  " + "-"*40)
for name, (mean, std) in sorted(r2_results.items(), key=lambda x: -x[1][0]):
    bar = "█" * int(max(0, mean) * 30)
    tag = " ← Burau単体ベースライン" if name == "Linear(Burau)" else ""
    print(f"  {name:<16} | {mean:+.4f}±{std:.4f}  {bar}{tag}")

print(f"\n  分類 AUC (上位25%識別):")
print(f"  {'モデル':<16} | AUC mean±std")
print("  " + "-"*40)
for name, (mean, std) in sorted(auc_results.items(), key=lambda x: -x[1][0]):
    bar = "█" * int(max(0, (mean-0.5)*60))
    print(f"  {name:<16} | {mean:.4f}±{std:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════
# BayesianRidge / ARD の係数（特徴量への重み）
# ══════════════════════════════════════════════════════════════════
print("\n[BayesianRidge / ARD 係数（全データ学習）]")
for name in ["BayesianRidge", "ARD"]:
    m = copy.deepcopy(ALL_MODELS[name][1])
    m.fit(X, y_reg)
    print(f"\n  {name}:")
    for fn, coef in sorted(zip(FEATURE_NAMES, m.coef_), key=lambda x: -abs(x[1])):
        bar = "█" * int(abs(coef) * 8)
        sign = "+" if coef >= 0 else "-"
        print(f"    {fn:<16}: {sign}{abs(coef):.4f}  {bar}")

# GP の学習済みカーネルパラメータ
print(f"\n[Gaussian Process 学習済みカーネル（全データ）]")
for name in ["GP-RBF", "GP-Matern"]:
    m = copy.deepcopy(ALL_MODELS[name][1])
    m.fit(X, y_reg)
    print(f"  {name}: {m.kernel_}")

# ══════════════════════════════════════════════════════════════════
# 全モデル Precision@N 結果表示
# ══════════════════════════════════════════════════════════════════
print("\n[全モデル Precision@N 比較（5-fold CV mean±std）]")

ALL_KEYS = (["GlobalBurau", "3Stg-25%"]
            + list(BAYES_MODELS.keys())
            + list(PREV_MODELS.keys()))

SECTIONS = [
    ("── 解析的", ["GlobalBurau", "3Stg-25%"]),
    ("── ベイズ系", list(BAYES_MODELS.keys())),
    ("── 従来ML (W-7j 再掲)", list(PREV_MODELS.keys())),
]

for section_title, keys in SECTIONS:
    print(f"\n  {section_title}")
    hdr = f"  {'手法':<16}|"
    for N in N_vals: hdr += f" @{N:<5}|"
    print(hdr)
    print("  " + "-"*(16 + 9*len(N_vals) + 2))
    for key in keys:
        line = f"  {key:<16}|"
        for N in N_vals:
            vals = cv_prec[key][N]
            line += f" {np.mean(vals):.3f} |"
        print(line)

# ══════════════════════════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════════════════════════
print("\n[可視化]")

# ── 図1: Precision@N カーブ（全モデル）──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("W-7k: Precision@N — Bayesian Models vs Analytical (5-fold CV)", fontsize=12)

styles = {
    "GlobalBurau" : ("blue",    "o",  "--", 2.5),
    "3Stg-25%"    : ("green",   "s",  "-",  2.5),
    "GaussianNB"  : ("magenta", "^",  "-",  1.8),
    "BayesianRidge":("purple",  "D",  "-",  1.8),
    "ARD"         : ("indigo",  "v",  "-",  1.8),
    "GP-RBF"      : ("crimson", "P",  "-",  1.8),
    "GP-Matern"   : ("darkorange","h", "-", 1.8),
    "SVR-RBF"     : ("#888",    "x",  ":",  1.2),
    "RF"          : ("#aaa",    "+",  ":",  1.2),
    "GBM"         : ("#ccc",    "1",  ":",  1.2),
    "XGBoost"     : ("#bbb",    "2",  ":",  1.2),
}

# 左: ベイズ系 + 解析的のみ
ax = axes[0]
focus_keys = ["GlobalBurau","3Stg-25%","GaussianNB","BayesianRidge","ARD","GP-RBF","GP-Matern"]
for key in focus_keys:
    c, mk, ls, lw = styles[key]
    means = [np.mean(cv_prec[key][N]) for N in N_vals]
    stds  = [np.std(cv_prec[key][N])  for N in N_vals]
    ax.plot(N_vals, means, marker=mk, ls=ls, color=c, lw=lw, label=key, alpha=0.9)
    ax.fill_between(N_vals, [m-s for m,s in zip(means,stds)],
                    [m+s for m,s in zip(means,stds)], color=c, alpha=0.1)
ax.set_xlabel("N (selected out of 720)")
ax.set_ylabel("Precision@N")
ax.set_title("Focus: Bayesian vs Analytical")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

# 右: 全モデル
ax = axes[1]
for key in ALL_KEYS:
    c, mk, ls, lw = styles[key]
    means = [np.mean(cv_prec[key][N]) for N in N_vals]
    ax.plot(N_vals, means, marker=mk, ls=ls, color=c, lw=lw, label=key, alpha=0.85)
ax.set_xlabel("N (selected out of 720)")
ax.set_ylabel("Precision@N")
ax.set_title("All Models")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(OUT / "W7k_fig1_precision_all.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig1 saved: {OUT / 'W7k_fig1_precision_all.png'}")

# ── 図2: N=50 Precision@N バーチャート (全モデル横断)────────────
fig, ax = plt.subplots(figsize=(12, 5))
N_bar = 50
bar_keys   = ALL_KEYS
bar_values = [float(np.mean(cv_prec[k][N_bar])) for k in bar_keys]
bar_errs   = [float(np.std(cv_prec[k][N_bar]))  for k in bar_keys]
bar_colors = (["#2196F3","#4CAF50"]
              + ["#9C27B0","#673AB7","#3F51B5","#E91E63","#FF5722"]
              + ["#9E9E9E"]*4)
bars = ax.bar(range(len(bar_keys)), bar_values, color=bar_colors,
              yerr=bar_errs, capsize=4, alpha=0.85)
ax.set_xticks(range(len(bar_keys)))
ax.set_xticklabels(bar_keys, rotation=35, ha='right', fontsize=10)
ax.set_ylabel("Precision@50 (5-fold CV)")
ax.set_title(f"W-7k: Precision@{N_bar} — 全モデル比較 (K=6, n=720)")
ax.axhline(float(np.mean(cv_prec["GlobalBurau"][N_bar])), ls="--", color="#2196F3",
           alpha=0.6, label="GlobalBurau baseline")
for i, v in enumerate(bar_values):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=8)

# 区切り線
ax.axvline(1.5, color='gray', ls=':', alpha=0.5)
ax.axvline(6.5, color='gray', ls=':', alpha=0.5)
ax.text(0.5,  0.02, "解析的", ha='center', transform=ax.get_xaxis_transform(),
        fontsize=8, color='gray')
ax.text(4.0,  0.02, "ベイズ系", ha='center', transform=ax.get_xaxis_transform(),
        fontsize=8, color='purple')
ax.text(9.0,  0.02, "従来ML", ha='center', transform=ax.get_xaxis_transform(),
        fontsize=8, color='gray')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
plt.tight_layout()
fig.savefig(OUT / "W7k_fig2_bar_N50.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig2 saved: {OUT / 'W7k_fig2_bar_N50.png'}")

# ── 図3: BayesianRidge 係数 vs ARD 係数（事後スパース性の比較）──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("W-7k: Bayesian Linear Models — Coefficient Comparison", fontsize=12)

for ax, name in zip(axes, ["BayesianRidge", "ARD"]):
    m = copy.deepcopy(ALL_MODELS[name][1])
    m.fit(X, y_reg)
    coefs = m.coef_
    order = np.argsort(np.abs(coefs))[::-1]
    colors_feat = []
    for fn in [FEATURE_NAMES[i] for i in order]:
        if fn in ("max_cycle_len","n_cycles","n_fixed"):
            colors_feat.append("#4CAF50")
        elif fn == "burau":
            colors_feat.append("#2196F3")
        else:
            colors_feat.append("#FF9800")
    ax.barh(range(len(FEATURE_NAMES)), [coefs[i] for i in order],
            color=colors_feat, alpha=0.85)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order])
    ax.set_xlabel("Coefficient (standardized)")
    ax.set_title(name)
    ax.axvline(0, color='black', lw=0.8)
    r2_m, r2_s = r2_results.get(name, (0,0))
    ax.text(0.97, 0.05, f"CV R²={r2_m:.3f}±{r2_s:.3f}",
            transform=ax.transAxes, ha='right', fontsize=9)

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor="#4CAF50", label="cycle type 系"),
                Patch(facecolor="#2196F3", label="Burau trace"),
                Patch(facecolor="#FF9800", label="Writhe")]
axes[1].legend(handles=legend_elems, loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "W7k_fig3_bayes_coefs.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  fig3 saved: {OUT / 'W7k_fig3_bayes_coefs.png'}")

# ══════════════════════════════════════════════════════════════════
# 総合サマリー
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  総合サマリー")
print("=" * 65)

ref_n = 50
print(f"\n  [Precision@{ref_n} ランキング（全モデル）]")
ranking = sorted(
    [(k, float(np.mean(cv_prec[k][ref_n]))) for k in ALL_KEYS + ["3Stg-25%"]],
    key=lambda x: -x[1])
for rank, (key, val) in enumerate(ranking, 1):
    bar = "█" * int(val * 30)
    tag = " ← 解析的" if key in ("GlobalBurau","3Stg-25%") else ""
    tag += " ← ベイズ系" if key in BAYES_MODELS else ""
    print(f"  {rank:2}. {key:<16}: {val:.4f}  {bar}{tag}")

# ベイズ系の最良
best_bayes = max(BAYES_MODELS.keys(),
                 key=lambda k: float(np.mean(cv_prec[k][ref_n])))
val_best_bayes = float(np.mean(cv_prec[best_bayes][ref_n]))
val_3stg       = float(np.mean(cv_prec["3Stg-25%"][ref_n]))
val_gb         = float(np.mean(cv_prec["GlobalBurau"][ref_n]))

print(f"\n  [ベイズ系の最良: {best_bayes}  Precision@{ref_n}={val_best_bayes:.4f}]")
if val_best_bayes > val_3stg:
    print(f"  → ベイズ系が 3-Stage 解析を上回った！")
elif val_best_bayes > val_gb:
    print(f"  → ベイズ系は GlobalBurau を上回るが 3-Stage には届かず")
else:
    print(f"  → ベイズ系も GlobalBurau に届かず。解析的アプローチが優位")

print(f"\n  [特徴量の重みについて]")
br = copy.deepcopy(ALL_MODELS["BayesianRidge"][1]); br.fit(X, y_reg)
ard = copy.deepcopy(ALL_MODELS["ARD"][1]); ard.fit(X, y_reg)
ct_feats = ["max_cycle_len","n_cycles","n_fixed"]
for name, m in [("BayesianRidge", br), ("ARD", ard)]:
    ct_w  = sum(abs(m.coef_[FEATURE_NAMES.index(f)]) for f in ct_feats)
    bur_w = abs(m.coef_[FEATURE_NAMES.index("burau")])
    print(f"  {name}: cycle_type系={ct_w:.4f}  Burau={bur_w:.4f}  "
          + ("→ Burau支配" if bur_w > ct_w else "→ cycle_type支配"))

print("\n" + "=" * 65)
print(f"  W-7k 実験完了  出力先: {OUT}")
print("=" * 65)
