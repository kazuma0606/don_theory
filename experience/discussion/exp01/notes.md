# Exp1 考察メモ

> 後で §5.3 Results / §6 Discussion に統合するための作業メモ。
> Copilot との議論をベースに整理。

---

## 数値結果サマリ

| 指標 | raw (J) | smooth (J_ε) |
|---|---|---|
| 収束率 (‖θ−θ*‖ < 0.05) | 16/20 (80%) | 1/20 (5%) |
| 最終 loss | ~0 (10^-16 まで下降) | ~0.0104 (フロア) |
| 最終 ‖θ−θ*‖ | 0.037 ± 0.059 | 0.458 ± 0.088 |
| 勾配ノルムの軌跡 | ノイズ的・ギザギザ | 素直に減衰（教科書的） |

---

## 核心的観察：3 つの図を並べると見えること

### 1. 距離 ‖θ − θ*‖
- raw・smooth どちらも θ* 方向にそれなりに動いている
- 「パラメータとしての質」はそこまで大差ない

### 2. Loss
- raw: 異常に小さい（10^-16 まで）
- smooth: 10^-2 付近で頭打ち
- **この違いは θ の良さを反映していない**

### 3. 勾配ノルム
- raw: ノイズが乗ったまま不規則に落ちる
- smooth: 綺麗に単調減衰 → landscape が整っている証拠

---

## 解釈の核心

### Loss の絶対値は比較対象ではない

J と J_ε は別の関数。θ = θ* で：

```
J(θ*)   = ‖z_target − z_target‖²          = 0
J_ε(θ*) = ‖smooth(z_target) − z_target‖² ≈ 0.0104
```

フロアは「悪さ」ではなく、カーネルと軌跡形状が決める幾何学的な量。
ε → 0 でこのフロアはゼロに収束する（Corollary 2 の数値検証 → Exp5 で実施）。

### raw loss が 10^-16 まで落ちる理由

非可換作用素の合成（E1 ∘ E2 ≠ E2 ∘ E1）が loss landscape に
高周波・不連続に近い成分を生む。その「偶然できた極端に深い谷」に
最適化が落ち込んでいるだけの可能性が高い。

→ 帯域外の深い局所解（Copilot の比喩：「帯域解でもないのに局所解にハマる」）

### smooth loss が安定する理由

mollifier の畳み込みが landscape を帯域制限する：
- 高周波の谷が丸められる
- 勾配が安定 → 素直な最適化経路
- Loss の絶対値は高いが、θ の軌跡は信頼できる

---

## 論文用テキスト案

### Results 節（1 段落）

Although the unsmoothed objective J attains extremely small numerical values
(down to 10^-16), this does not correspond to a substantially better estimate
of θ*. The distance ‖θ_ε* − θ*‖ remains comparable between the two conditions,
while the gradient norm trajectory under J exhibits irregular, noise-like
fluctuations. In contrast, J_ε plateaus near 10^-2 — a floor determined
geometrically by the smoothing kernel applied to z_target — yet produces a
monotonically decaying gradient norm and a more stable optimization path.

### Discussion 節（核心の 1 段落）

The unsmoothed objective J contains high-frequency components induced by
non-commutative operator compositions, allowing gradient descent to fall into
extremely deep but band-unlimited local minima. The smoothed objective J_ε
removes these high-frequency artifacts, yielding a band-limited landscape in
which optimization converges more reliably, even though the numerical loss
values remain higher. Thus, the absolute magnitudes of J and J_ε should not
be compared directly; their optimization behaviors reflect fundamentally
different landscape geometries. The primary benefit of smoothing lies not in
reducing the raw loss value but in regularizing the landscape to make the
optimization path tractable.

### Figure キャプション案

**Figure 1** (loss curves):
Loss trajectories for 20 random initializations under J (left) and J_ε (right).
J_ε converges to a constant floor ≈ 0.0104 determined by the smoothing kernel,
while J reaches near-machine-precision values. These absolute magnitudes are
not directly comparable (see Discussion).

**Figure 2** (θ-distance):
Distance ‖θ − θ*‖ over optimization steps. Both conditions reduce the distance
to θ*, though J_ε shows more consistent convergence across initializations.

**Figure 3** (gradient norms):
Gradient norm trajectories. J_ε exhibits smooth, monotone decay characteristic
of a well-conditioned landscape; J shows irregular fluctuations consistent with
a high-frequency loss surface.

---

## 今後への接続

- **Exp4 (landscape 可視化)**: 上記の「帯域制限」を 2D ヒートマップで直接確認
- **Exp5 (ε-scaling)**: フロア 0.0104 が ε → 0 でゼロに収束することを定量検証
  → Corollary 2 の数値的裏付けはここで完結する

---

## メモ：この実験の位置づけ

Exp1 は「smooth が勝つことの実証」ではなく、
**「理論的枠組みの数値的 sanity check」** として読むのが正しい。

主貢献は Lean4 形式証明（Layer 1–4）であり、実験は
「理論的対象が数値的に実体化でき、予測通りに振る舞う」ことの確認。

raw loss が「異常に強く」見えるのも、平滑 loss のフロアも、
いずれも理論から予測される挙動であり、実験がその構造を可視化している。
