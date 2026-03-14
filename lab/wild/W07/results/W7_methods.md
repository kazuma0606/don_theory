# W-7 実験シリーズ — 実験方法詳細

## 実験の目的

介入操作の非可換性を利用した **介入順序のスクリーニング** において、
順序空間（B_K / S_K）の位相不変量がランドスケープ距離の予測子として機能するか検証する。

---

## 1. 共通セットアップ（全実験共通）

### 状態空間パラメータ

| パラメータ | 値 |
|---|---|
| 状態次元 | d = 24 |
| 介入数 | K ∈ {3, 4, 5, 6} |
| ラウンド数 | N_ROUNDS = 2 |
| パラメータグリッド | N_GRID = 25 × 25 = 625 点 |
| θ グリッド範囲 | θ₁, θ₂ ∈ [−1, 3] |
| 最適パラメータ | θ★ = [1.0, 1.0] |
| 乱数シード | `np.random.default_rng(int(42))` |

### 時間発展演算子

```
U(p) = tanh(A_dyn · p + b_bias) × 0.5 + p × 0.4
```

A_dyn は反対称行列（`A = A' - A'ᵀ`, σ=0.25），b_bias は小ノイズベクトル（σ=0.1）。

### 介入演算子（E₁–E₆）

| 演算子 | 定義 |
|---|---|
| E₁(p; θ) | p + θ₁ · a_vec（線形シフト; a_vec は前d/4次元が1） |
| E₂(p; θ) | (I + θ₂ · M_mat) · p（線形変換; M_mat は正規化ランダム行列） |
| E₃(p) | tanh(1.2·p) × 0.9（非線形圧縮） |
| E₄(p) | tanh(0.8·p + bias₀) × 0.7（バイアス付き圧縮） |
| E₅(p) | 0.6·p + tanh(bias₁) × 0.4（アフィン） |
| E₆(p) | tanh(p + bias₂) × 0.85（バイアス付き非線形） |

E₁–E₂ のみθ依存；E₃–E₆はθ非依存（純粋非線形）。K=3 なら E₁–E₃ のみ使用。

### ランドスケープ距離（dist）の計算

1. 参照順序 `ref = [0,1,…,K-1]` で θ★ による軌跡 `z_t` を計算
2. 評価順序 `order` で 25×25 グリッドの各 θ で損失 `J(order, θ)` を計算：
   ```
   J(order, θ) = mean_t ||traj(order,θ)_t − z_t||²
   ```
3. 距離：`dist = sqrt(mean_{θ} (J(order,θ) − J(ref,θ))²)` （RMSE on grid）

---

## 2. 特徴量の定義

### 2.1 Burau trace（主要特徴量）

K次元組み紐群 B_K において，各置換 σ ∈ S_K に対して左辞書順既約語を用いて組み紐 b を構成し，
unreduced Burau 表現を t=1/2 で評価したトレースを使用：

```sage
b = BK.one()
for i in sp.reduced_word():
    b = b * gK[i-1]
btr = float(b.burau_matrix()(t=QQ(1)/QQ(2)).trace())
```

### 2.2 Cycle type（共役類）

置換 σ の cycle type（K の分割）を文字列キーとして使用。K=6 では 11 種の cycle type が存在：
`[1,1,1,1,1,1]`（恒等置換），`[2,1,1,1,1]`，`[2,2,1,1]`，`[2,2,2]`，
`[3,1,1,1]`，`[3,2,1]`，`[3,3]`，`[4,1,1]`，`[4,2]`，`[5,1]`，`[6]`。

### 2.3 Cayley 距離

```
cayley_dist(σ) = K − n_cycles(σ)
```

K=6 での値は 0（恒等）から 5（K-サイクル）の範囲。

### 2.4 標準表現行列（SRM）

S_K の標準表現（(K−1)×(K−1) 行列）：

```python
def standard_rep_matrix(perm_0idx, K):
    M = np.zeros((K-1, K-1))
    sig = perm_0idx; sK1 = sig[K-1]
    for j in range(K-1):
        sj = sig[j]
        if sj < K-1:  M[sj, j]  += 1.0
        if sK1 < K-1: M[sK1, j] -= 1.0
    return M
```

K=6 で (5×5)=25 次元特徴ベクトルとして使用。

### 2.5 Jones 多項式（W-7f のみ）

```sage
jp = braid.jones_polynomial()
jp_val = float(jp.subs(t=RR(0.25)))
```

最初の呼び出しは Hecke 代数初期化のため warm-up 計算が必要。

### 2.7 Burau 多点評価（W-7-2 Phase 1）

Burau trace を複数の評価点 t ∈ {1/4, 1/3, 1/2, 2/3, 3/4} で評価し，5 次元特徴ベクトルを構成：

```sage
t_vals = [QQ(1)/QQ(4), QQ(1)/QQ(3), QQ(1)/QQ(2), QQ(2)/QQ(3), QQ(3)/QQ(4)]
feats = [float(b.burau_matrix()(t=tv).trace()) for tv in t_vals]
```

| 特徴セット | 説明 | 次元 |
|---|---|---|
| A | Burau trace (t=1/2 のみ，ベースライン) | 1 |
| B | Burau traces at 5 points | 5 |
| C | B + Cayley 距離 | 6 |
| D | B + cycle type ダミー (11D) | 16 |
| E | Lasso 選択後（生存: t=1/4, t=3/4）| 2 |

多重共線性診断に VIF（statsmodels.stats.outliers_influence.variance_inflation_factor）を使用。
評価点間の相関が 0.88–0.997 と極めて高く，VIF = ∞ となる（完全共線性）。

### 2.8 LKB 忠実表現（W-7-2 Phase 3）

Lawrence-Krammer-Bigelow (LKB) 表現は K≥5 でも忠実（Burau は K≥5 で非忠実）。
K=6 では 15×15 行列（次元 K(K-1)/2 = 15）として定義される。

```sage
BK = BraidGroup(K)
b = BK(word)
m_sym = b.LKB_matrix()          # symbolic matrix over Z[x, y]
x_var, y_var = m_sym.base_ring().gens()
m_num = m_sym.subs({x_var: xv, y_var: yv})  # evaluate at (xv, yv)
lkb_trace = float(m_num.trace())
lkb_eigs  = [abs(complex(e)) for e in m_num.eigenvalues()]
```

評価点: (x,y) ∈ {(1/2,1/2), (1/3,1/2), (2/3,1/2), (1/2,1/3), (1/2,2/3)}

| 特徴セット | 説明 | 次元 |
|---|---|---|
| G | LKB trace (1 評価点) | 1 |
| H | LKB trace (5 評価点) | 5 |
| I | LKB 固有値絶対値 (1 評価点) | 15 |
| J | LKB スカラー群 (trace/det/max_eig/frob) × 2 点 | 8 |
| K | Burau (1pt) + LKB trace (5pt) | 6 |
| L | LKB (H: 5pt) + SRM (25D) | 30 |
| M | Burau (1pt) + LKB (H: 5pt) + SRM (25D) | 31 |

### 2.6 Cayley グラフ Laplacian 固有ベクトル（W-7n）

S₆（720頂点）上の Cayley グラフ（全 C(6,2)=15 個の隣接転置で生成）を構築し，
グラフ Laplacian L = D − A（D は次数対角行列，次数=15）を固有分解：

```python
L_cayley = 15 * np.eye(720) - A_cayley.astype(np.float64)
eigenvalues, U_eigen = eigh(L_cayley)
```

固有値は 0 から 30 の範囲（720 個）。

---

## 3. スクリーニング戦略

### 3.1 GlobalBurau

Burau trace と dist の Pearson r を全データで計算し，その符号で「大 Burau 優先」か「小 Burau 優先」かを決定して上位 N 件を返す。

### 3.2 2-Stage（W-7h）

1. Cycle type ごとに within-type r(Burau, dist) を計算し direction を決定
2. 各 cycle type から比例配分で N_ct 件をサンプリング，within-type Burau ランキング適用

### 3.3 3-Stage-25%（W-7i：最良解析的手法）

```
Stage 0: mean_dist が全体の 25th パーセンタイル以下の cycle type のみ保持
Stage 1: 残存 cycle type の要素数に比例して N 件を配分
Stage 2: 型内 direction-aware Burau ランキングで上位を選択
```

25th パーセンタイル閾値でフィルタリングされる cycle type = `{[1,1,1,1,1,1], [2,1,1,1,1], [3,1,1,1]}`，
対応する要素数 = 56/720。

### 3.4 機械学習（W-7j, W-7k）

特徴量：Burau trace，Cayley 距離，cycle type ダミー変数（11次元），SRM（25次元）

- **W-7j**：SVM（RBF），RandomForest，GradientBoosting，XGBoost（全てデフォルトパラメータ）
- **W-7k**：GaussianNB，BayesianRidge，ARD（Relevance Determination），GP（RBF/Matérn カーネル）

評価：5-fold CV，各 fold で Precision@N を計算。

### 3.5 解析的拡張（W-7l）

- **A: Cayley 距離単独**：小 Cayley → 選択
- **B: 正則化 Burau**：Burau / (1 + λ|Cayley − μ|)
- **C: スペクトル半径 Burau**：固有値の最大絶対値を追加特徴量
- **C4: Bur + Spec 複合**：Burau + スペクトル半径の線形和
- **E: Borda 集約**：複数スコアのランキング順位の平均

### 3.6 新方向性（W-7m）

- **B1: OPW（Operator-Position Weighting）**：E₁の位置 pos_E1 を使用したスコア
- **C1: Jacobian trace**：シーケンス Jacobian ∂p_T/∂p_0 を有限差分で推定（24方向の摂動）
- **D1: コセット分解**：S₆/S₄ コセット代表元インデックスを特徴量に
- **E: SRM 線形回帰（S_K Fourier）**：標準表現行列の (K-1)² 要素を特徴量として線形回帰

### 3.8 W-7-2 多変量回帰プロトコル

各特徴セット（A–M）に対して以下を順次適用：

**① VIF 診断**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
```
VIF > 10 で多重共線性あり，∞ で完全共線性。

**② OLS（statsmodels）**
```python
import statsmodels.api as sm
res = sm.OLS(y, sm.add_constant(X)).fit()
# 出力: Adj.R², F-stat, 各変数の係数/t値/p値
```

**③ LassoCV（特徴選択）**
```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_scaled, y)
# 係数 0 の変数を除去し，生存特徴量を確認
```

**④ RidgeCV（正則化回帰）**
```python
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5).fit(X_scaled, y)
cv_r2 = cross_val_score(ridge, X_scaled, y, cv=kf, scoring='r2').mean()
```

**⑤ 固有値特徴量の PCA 直交化（Phase 2）**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=n_components).fit_transform(X_eigen)
# 分散寄与率・PC1 と dist の相関を確認
```

**⑥ Precision@N 評価**
5-fold CV 各 fold のテストデータで Ridge 予測スコアを使いランキングし，P@20 / P@50 を計算。

### 3.7 スペクトル手法（W-7n）

- **Graph Fourier (GF_Mxx)**：Cayley Laplacian の最初の M 固有ベクトルで Ridge 回帰（α=1.0）
- **Heat kernel**：`exp(−s·λ)` フィルタで固有係数を減衰させ平滑化
- **Haar wavelet**：介入順序の prefix（order[0], order[0:2], order[0:3]）による階層的平均 + James-Stein 収縮
- **Wavelet packet**：複数スケール熱核の特徴量を Ridge で結合
- **Ridge SRM (Ridge_α)**：SRM 25次元特徴量に Ridge 正則化（α ∈ {0.01, 0.1, 1, 10, 100}）
- **SRM+GF_M30**：SRM (25次元) + グラフ Fourier (30次元) = 55次元 Ridge 特徴

---

## 4. 評価指標

### 4.1 Precision@N（主要指標）

```
P@N = |{選択した N 件} ∩ {真の上位 N 件}| / N
```

「真の上位 N 件」= dist でソートした上位 N 件（Oracle）。

### 4.2 Pearson 相関 r

Burau trace と dist の Pearson 相関係数（スケーリング則の定量化に使用）。

### 4.3 5-fold CV 設定（W-7j 以降）

K=6 の 720 件を 5 分割交差検証（`KFold(n_splits=5, shuffle=True, random_state=42)`）。
各 fold でテスト分割 (144 件) に対して Precision@N を評価し，5 fold の平均を報告。

---

## 5. 実験ファイル一覧

| ファイル | 内容 |
|---|---|
| exp_W7.sage | B₃ 基礎実験（6 置換） |
| exp_W7b.sage | 純組み紐実験（恒等置換のみ） |
| exp_W7c.sage | B₄ への拡張（24 置換） |
| exp_W7d.sage | B₅（120 置換），sklearn MDS 可視化 |
| exp_W7e.sage | B₆（720 置換），スケーリング則確認 |
| exp_W7f.sage | Jones 多項式 vs Burau trace 比較 |
| exp_W7g.sage | 非線形・環状・クラスタ解析 |
| exp_W7h.sage | 2-Stage スクリーニング |
| exp_W7i.sage | **3-Stage-25%** スクリーニング（最良解析的手法） |
| exp_W7j.sage | ML（SVM/RF/GBM/XGBoost）比較 |
| exp_W7k.sage | ベイズ系（NB/BayesRidge/ARD/GP）比較 |
| exp_W7l.sage | 解析的拡張 A–E |
| exp_W7m.sage | 新方向性 4 種（OPW/Jacobian/Coset/SRM-Fourier） |
| exp_W7n.sage | **Cayley グラフスペクトル + Wavelet** 全手法 |
| exp_W7-2a.sage | W-7-2 Phase 1: Burau 多点評価 + OLS/LassoCV/RidgeCV/VIF |
| exp_W7-2b.sage | W-7-2 Phase 2: Burau 固有値スペクトル + PCA 直交化 |
| exp_W7-2c.sage | W-7-2 Phase 3: LKB 忠実表現 全特徴セット G–M |
