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
PyTorch MLP 1 層：

\[
p_{t+1} = G(p_t) = B(\tanh(A(p_t)))
\]

実装：

```python
class Dynamics(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Linear(d, d)
        self.B = nn.Linear(d, d)

    def forward(self, p):
        return self.B(torch.tanh(self.A(p)))
```

---

## **0.3 介入（Interventions）**

### **介入 1：加算型（局所）**
```python
delta = torch.zeros(d); delta[:16] = 1.0
def E1(p, theta):
    return p + theta * delta
```

### **介入 2：線形変換型（部分空間）**
```python
M = nn.Linear(d, d, bias=False)
def E2(p, theta):
    return M(p) * theta
```

### **非可換性の確保**
- E1 は一部次元のみ変化  
- E2 は別の部分空間を回転  
→ **E1 ∘ E2 ≠ E2 ∘ E1** が自然に成立

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
def rollout(p0, theta, dynamics, interventions, T, schedule):
    p = p0
    zs = []
    for t in range(T+1):
        if t in schedule:
            for E in schedule[t]:
                p = E(p, theta)
        z = observe(p)
        zs.append(z)
        if t < T:
            p = dynamics(p)
    return torch.stack(zs, dim=0)  # (T+1, batch, q)
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

## **0.7 損失関数**

### 非平滑：

```python
def loss_raw(theta):
    zs = rollout(...)
    return ((zs - zs_target)**2).mean()
```

### 平滑：

```python
def loss_smooth(theta):
    zs = rollout(...)
    zs_s = smooth_time(zs, kernel)
    zs_t_s = smooth_time(zs_target, kernel)
    return ((zs_s - zs_t_s)**2).mean()
```

---

# **1. Experiment 1：平滑化の有無による最適化の安定性**

## **目的**
- 平滑化なし → 発散・不安定  
- 平滑化あり → 安定収束  

## **要件**
- 状態次元：64  
- ダイナミクス：固定  
- 介入：E1, E2  
- 観測：q=8  
- T=20  
- θ を 1 次元パラメータとして最適化  
- 初期値を 20 通りサンプリング  
- 最適化：Adam(lr=1e-2)

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
- θ を 2 次元に拡張：\((\theta_1, \theta_2)\)  
- グリッドサーチで  
  - 非平滑損失  
  - 平滑損失  
  を計算  
- 2D ヒートマップ or 3D サーフェスで可視化

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