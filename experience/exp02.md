了解です。Claude Code のような **コーディングエージェントに渡すための「実装しやすい要件書」** を作る、という方向ですね。  
これは非常に良い判断で、エージェントにとっては

- 抽象的な文章より  
- **明確な仕様・データ構造・関数インターフェース**  

がある方が圧倒的に実装しやすいです。

そこで、あなたの 4 つの実験を **Claude Code がそのまま読み取って実装できる形式** に再構成します。

以下は「背景を理解しつつ、実装に必要な情報だけを抽出した仕様書」です。

---

# ✅ **Claude Code 用：実験実装仕様書（Implementation Requirements）**

この仕様書は、非可換介入 × 時間発展 × Banach 平滑化の実験をローカルで実行するための **最小限かつ完全な要件** をまとめたものです。

---

# **0. 共通仕様（全実験で使用）**

## **0.0 実行環境**

**CUDA 前提。** CPU フォールバックは不要。

```python
import torch

device = torch.device("cuda")  # CPU フォールバック不要

# 全テンソル・モジュールはここで device に送る
# 例：
#   dynamics = Dynamics(d).to(device)
#   p0_fixed = p0_fixed.to(device)
#   M_base   = M_base.to(device)
#   a        = a.to(device)
#   kernel   = kernel.to(device)
```

大きなグリッドサーチ（Exp 4）や繰り返し rollout（Exp 2, 3, 5）では `torch.cuda.synchronize()` を使った時間計測を推奨。

---

## **0.1 状態空間**
- 状態ベクトル：  
  \[
  p_t \in \mathbb{R}^d
  \]
- デフォルト：  
  - \(d = 64\)  
  - Experiment 3 では \(d \in \{32, 64, 128\}\)

---

## **0.2 時間発展（Dynamics）**
論文 §5.1.2 の定義に従う：

\[
p_{t+1} = A p_t + b + \gamma \tanh(W p_t)
\]

実装：

```python
class Dynamics(nn.Module):
    def __init__(self, d, gamma=0.1):
        super().__init__()
        self.linear = nn.Linear(d, d)        # A p_t + b（bias 込み）
        self.W = nn.Linear(d, d, bias=False) # W p_t（tanh 項）
        self.gamma = gamma

    def forward(self, p):
        return self.linear(p) + self.gamma * torch.tanh(self.W(p))
```

---

## **0.3 介入（Interventions）**

**最適化パラメータ：θ = (θ₁, θ₂) ∈ ℝ² を全実験で統一して使用。**

### **介入 1：加算型 E_A（論文 §5.1.5 の $E_{A,\theta}(p) = p + \theta a$）**
```python
a = torch.zeros(d); a[:16] = 1.0   # 固定ベクトル（次元 0:16 のみ）

def E1(p, theta1):
    return p + theta1 * a
```

### **介入 2：線形変換型 E_B（論文 §5.1.5 の $E_{B,\theta}(p) = B(\theta) p$）**
```python
# B(θ₂) = I + θ₂ M（固定ランダム行列 M で θ₂ に依存する行列を実現）
torch.manual_seed(0)
M_base = torch.randn(d, d) * 0.1

def E2(p, theta2):
    return p + theta2 * (M_base @ p)   # = (I + θ₂ M) p
```

### **非可換性の確保**
- `E1(E2(p)) = p + θ₂ M p + θ₁ a`
- `E2(E1(p)) = p + θ₁ a + θ₂ M(p + θ₁ a) = p + θ₁ a + θ₂ M p + θ₁θ₂ M a`
- 差分：`θ₁θ₂ M a ≠ 0`（M a がゼロでなければ成立）
→ **E1 ∘ E2 ≠ E2 ∘ E1**（θ₁ ≠ 0, θ₂ ≠ 0 のとき）

---

## **0.4 観測（Observation）**

観測関数：

\[
z_t = C(p_t) \in \mathbb{R}^q
\]

実装：

```python
C = nn.Linear(d, q)  # q = 8
def observe(p):
    return C(p)
```

---

## **0.5 rollout 関数（時間発展＋介入＋観測）**

```python
def rollout(p0, theta, dynamics, schedule, T):
    """
    Args:
        p0:       初期状態 (batch, d)
        theta:    パラメータ (θ₁, θ₂) の Tensor, shape (2,)
        dynamics: Dynamics インスタンス
        schedule: dict {t: [(E, theta_idx), ...]}
                  例: {5: [(E1, 0)], 15: [(E2, 1)]}
                  theta_idx は theta[0] or theta[1] を指定
        T:        タイムステップ数
    Returns:
        zs: (T+1, batch, q)
    """
    p = p0
    zs = []
    for t in range(T + 1):
        if t in schedule:
            for (E, idx) in schedule[t]:
                p = E(p, theta[idx])
        zs.append(observe(p))
        if t < T:
            p = dynamics(p)
    return torch.stack(zs, dim=0)

# デフォルトスケジュール（T=20 のとき）
# E1 を t=5 に、E2 を t=15 に適用（非可換性が時間発展と相互作用する）
DEFAULT_SCHEDULE = {5: [(E1, 0)], 15: [(E2, 1)]}
```

---

## **0.6 Banach 的平滑化（時間方向のモルフィア）**

### カーネル例（長さ 5）：

```python
kernel = torch.tensor([1., 4., 6., 4., 1.])
kernel = kernel / kernel.sum()
```

### 時間方向の 1D 畳み込み：

```python
def smooth_time(zs, kernel):
    zs = zs.permute(1, 2, 0)  # (batch, q, T+1)
    k = kernel.view(1, 1, -1).to(zs.device)
    pad = (kernel.numel() // 2, ) * 2
    zs_padded = F.pad(zs, pad, mode='replicate')
    zs_smooth = F.conv1d(zs_padded, k)
    return zs_smooth.permute(2, 0, 1)
```

---

## **0.7 ターゲット軌跡の生成**

ground truth パラメータ θ\* から生成（固定シード）：

```python
THETA_STAR = torch.tensor([1.0, 1.0])   # ground truth

batch = 32                               # デフォルトバッチサイズ（全実験共通）
torch.manual_seed(42)
p0_fixed = torch.randn(batch, d)        # 全実験で共通の初期状態

with torch.no_grad():
    zs_target = rollout(p0_fixed, THETA_STAR, dynamics, DEFAULT_SCHEDULE, T)
    # shape: (T+1, batch, q)，detach 済み
```

---

## **0.8 損失関数**

### 非平滑：

```python
def loss_raw(theta):
    zs = rollout(p0_fixed, theta, dynamics, DEFAULT_SCHEDULE, T)
    return ((zs - zs_target) ** 2).mean()
```

### 平滑（論文 $J_\varepsilon(\theta) = \|f_{E_\theta,\varepsilon} - y_\text{target}\|^2$ に整合）：

```python
def loss_smooth(theta):
    zs = rollout(p0_fixed, theta, dynamics, DEFAULT_SCHEDULE, T)
    zs_s = smooth_time(zs, kernel)
    # ターゲットは平滑化しない（理論の J_ε 定義に従う）
    return ((zs_s - zs_target) ** 2).mean()
```

---

## **0.9 非可換性のサニティチェック（実装前に確認）**

```python
p_test = torch.randn(d)
theta1_test, theta2_test = 1.0, 1.0
diff = (E1(E2(p_test, theta2_test), theta1_test)
      - E2(E1(p_test, theta1_test), theta2_test)).norm()
assert diff > 1e-3, f"非可換性が弱すぎます: {diff:.6f}"
print(f"‖E1∘E2(p) - E2∘E1(p)‖ = {diff:.4f}")
# 期待値: θ₁θ₂ ‖Ma‖ ≈ 1.0 * 1.0 * ‖M_base @ a‖
```

---

# **1. Experiment 1：平滑化の有無による最適化の安定性**

## **目的**
- 平滑化なし → 発散・不安定  
- 平滑化あり → 安定収束  

## **要件**
- 状態次元：d=64
- ダイナミクス：固定（seed=0 で初期化）
- 介入：E1（t=5）, E2（t=15）、スケジュール = DEFAULT_SCHEDULE
- 観測：q=8
- T=20
- θ = (θ₁, θ₂) を 2 次元パラメータとして最適化
- 初期値を 20 通りサンプリング（各 θᵢ ∈ Uniform(-2, 2)）
- 最適化：Adam(lr=1e-2)、500 ステップ

## **評価指標**
- 収束率  
- 最終損失の分散  
- 勾配ノルムの推移  

---

# **2. Experiment 2：ジェネレータのランダム性に対するロバスト性**

## **目的**
- ダイナミクスをランダムに変えても、平滑化の効果が安定して現れるか

## **要件**
- Dynamics を 10 回ランダム生成  
- 各 Dynamics で Experiment 1 を実行  
- 比較：平滑化あり vs なし

---

# **3. Experiment 3：状態次元のロバスト性**

## **目的**
- 次元が変わっても平滑化の効果が維持されるか

## **要件**
- \(d = 32, 64, 128\)  
- 各次元で Experiment 1 を簡略実行  
- ダイナミクス・介入・観測は次元に合わせて再生成

---

# **4. Experiment 4：損失 landscape の可視化**

## **目的**
- 平滑化が landscape を滑らかにすることを視覚的に示す

## **要件**
- θ = (θ₁, θ₂) の 2 次元グリッドサーチ
  - 各軸：50 点、範囲 [-3, 3]
- 非平滑損失 J(θ) とカーネルサイズ ε を変えた平滑損失 J_ε(θ) を計算
- 2D ヒートマップ（左：J, 右：J_ε）で並列比較
- 1 つのダイナミクス・p0_fixed を使用（見せる用）

---

# **5. Experiment 5：ε スケーリング検証（Theorem 1 & Corollary 2）**

## **目的**
- **Theorem 1** の定量的検証：ε を変化させたとき ‖ẑ_ε(θ*) - z(θ*)‖ が O(ε) でスケールすること
- **Corollary 2** の検証：ε→0 のとき argmin J_ε が θ* = (1, 1) に収束すること

## **要件**
- d=64, T=20, q=8（Exp 1 と同じ環境）
- ダイナミクス：固定（seed=0）
- ε（カーネルサイズ）を変化：`eps_list = [1.0, 0.5, 0.2, 0.1, 0.05]`

### **カーネルの定義（ε 依存）**

各 ε に対してカーネル長 `K = max(3, 2*round(ε*10)+1)`（奇数）の二項係数カーネルを使用：

```python
def make_kernel(eps):
    """eps に比例した幅の二項係数カーネルを生成"""
    K = max(3, 2 * round(eps * 10) + 1)   # 奇数幅
    binom = torch.tensor([comb(K-1, k) for k in range(K)], dtype=torch.float32)
    return binom / binom.sum()
```

### **測定 1：軌跡近似誤差（Theorem 1）**

```python
# θ* = (1, 1) で rollout し、ε ごとに ‖ẑ_ε(θ*) - z(θ*)‖₂ を計測
z_true = rollout(p0_fixed, THETA_STAR, dynamics, DEFAULT_SCHEDULE, T)  # 非平滑
for eps in eps_list:
    kernel = make_kernel(eps)
    z_smooth = smooth_time(z_true, kernel)
    approx_err[eps] = (z_smooth - z_true).norm().item()
```

期待値：`approx_err` が eps に対してほぼ線形（log-log プロットで傾き ≈ 1.0）。

### **測定 2：最小解の収束（Corollary 2）**

```python
# 各 ε で J_ε を最適化 → argmin θ_eps を求め ‖θ_eps - θ*‖ を計測
for eps in eps_list:
    kernel = make_kernel(eps)
    theta = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=1e-2)
    for _ in range(500):
        zs = rollout(p0_fixed, theta, dynamics, DEFAULT_SCHEDULE, T)
        loss = ((smooth_time(zs, kernel) - zs_target) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    minimizer_err[eps] = (theta.detach() - THETA_STAR).norm().item()
```

期待値：`minimizer_err` が eps→0 で単調減少（収束）。

## **出力図**

2 パネルの図を生成：

```
左：log(ε) vs log(‖ẑ_ε - z‖)  — 傾き≈1 の直線が Theorem 1 を支持
右：log(ε) vs ‖argmin J_ε - θ*‖  — 右下がりが Corollary 2 を支持
```

---

# 🎯 **Claude Code に渡すときの推奨フォーマット**

Claude Code は以下のような構造を好みます：

1. **目的（What）**  
2. **仕様（Inputs / Outputs）**  
3. **関数インターフェース（Signatures）**  
4. **データ構造（Shapes）**  
5. **依存関係（PyTorch / numpy）**  
6. **実装タスクの分割（Step-by-step）**

上の仕様書はそのまま渡しても動きますが、  
必要なら **Claude Code 用の「プロンプトテンプレート」** も作れます。

---

必要なら次に：

- Claude Code に渡すための **完全プロンプト**  
- 実験 1 の **最小実装コード**  
- 実験全体の **ディレクトリ構成案**  

なども作れます。

どこまで自動化したいですか？