# Exp4 考察メモ

> 後で §5.3 Results / §6 Discussion に統合するための作業メモ。

---

## 数値結果サマリ

| 指標 | J（非平滑） | J_ε（平滑） |
|---|---|---|
| グリッド最小値 | 0.000009 | 0.010358 |
| 最小値の位置 | (1.04, 1.04) ≈ θ* | (1.41, 1.29)（バイアスあり） |
| グリッド最大値 | 0.0894 | 0.0369 |
| θ* = (1,1) での値 | 0.000009 | 0.010530 |

---

## 核心的観察

### 1. 値域の圧縮

J の max=0.089 に対して J_ε の max=0.037。
平滑化によって landscape の「高さ」が約 60% 圧縮されている。
これは Exp1 で観察した loss フロアと対応する:
- 低い値（深い谷）は持ち上げられる（フロア 0.0104）
- 高い値（急峻な峰）は押し下げられる
→ **landscape が全体的に均質化・平坦化**

### 2. 最小解のずれ

J_ε の最小は (1.41, 1.29)、θ* = (1.00, 1.00) から距離約 0.50。
Exp1〜3 で一貫して観察された ||θ_ε* − θ*|| ≈ 0.45〜0.66 の範囲に収まる。

landscape 上で見ると:
- J の最小は θ* に正確に乗っている（グリッド解像度の誤差 0.04 以内）
- J_ε の最小は θ* から右上にずれた位置にある
→ 平滑化カーネルが等方的でないため、θ₁, θ₂ 方向で非対称なずれが生じている可能性

### 3. landscape の滑らかさ（視覚的確認）

3D サーフェスで:
- J: ギザギザした高周波成分あり（Exp1 の勾配ノルムのノイズと対応）
- J_ε: なめらかな bowl 形状

ヒートマップ（log scale）で:
- J: 鋭い谷と平坦な領域が混在
- J_ε: 滑らかなグラデーション

→ Exp1 Discussion の「帯域制限された landscape」が視覚的に確認できた

---

## Exp1〜3 との整合性

| 実験 | 観察 | Exp4 との対応 |
|---|---|---|
| Exp1 | J の勾配ノルムがノイズ的 | J の landscape に高周波成分あり |
| Exp1 | J_ε の勾配が素直に減衰 | J_ε の landscape が滑らかな bowl |
| Exp1〜3 | J_ε の最小解が θ* からずれる | J_ε の谷底が (1.41,1.29) にある |
| Exp2,3 | バイアスが dynamics・次元依存 | J_ε の谷底位置が landscape 形状から決まる |

---

## 論文用テキスト案

### Results 節

Figure 4 visualizes the loss landscape J(θ) and J_ε(θ) over a 50×50 grid
in the parameter space θ = (θ₁, θ₂) ∈ [−3, 3]².

The unsmoothed landscape J exhibits a narrow global minimum near θ* = (1, 1)
(grid minimum at (1.04, 1.04), value ≈ 9×10⁻⁶) with a value range spanning
nearly an order of magnitude (0 to 0.089). The 3D surface reveals high-frequency
undulations characteristic of the non-commutative operator interactions.

In contrast, the smoothed landscape J_ε shows a compressed value range
(0.010 to 0.037) with a smooth, bowl-like geometry. The global minimum of J_ε
lies at (1.41, 1.29), displaced from θ* by approximately 0.50 — consistent
with the systematic bias observed in Experiments 1–3. Crucially, J_ε(θ*) = 0.0105
matches the theoretical floor ‖φ_ε ⋆ z_target − z_target‖², confirming that
the smoothing introduces a predictable, ε-dependent offset.

### Discussion 節（Exp1 考察メモと組み合わせる）

The landscape visualization directly confirms the interpretation proposed in
Experiment 1: J contains high-frequency components that create spuriously deep
local minima, while J_ε filters these components to produce a band-limited,
geometrically regular landscape. The compressed value range of J_ε (max=0.037
vs J max=0.089) reflects the averaging effect of the mollifier, which both
raises the global minimum floor and lowers the high-loss regions — a hallmark
of the W^{1,∞} regularization described in Section 3.

### Figure キャプション案

**Figure 4a** (heatmap):
Heatmap of J(θ) (left) and J_ε(θ) (right) over θ ∈ [−3,3]², shown in log
scale. Red dot: θ* = (1,1). White cross: grid minimum. Smoothing compresses
the value range and shifts the minimum away from θ*.

**Figure 4b** (3D surface):
3D surface of J(θ) (left) and J_ε(θ) (right). J exhibits irregular, high-
frequency undulations; J_ε presents a smooth bowl geometry characteristic of
the mollifier-regularized landscape.

---

## Exp5 への接続

Exp4 で見えた「J_ε の谷底が θ* からずれる」量が ε に比例することを
Exp5 で定量確認する（log-log プロットで傾き ≈ 1.0）。
また Exp4 のグリッドを複数の ε で再実行すれば
「ε が小さくなるにつれて谷底が θ* に近づく」動画的な可視化も可能。
（論文には不要だが補足材料として有効）
