# Exp5 考察メモ

> 後で §5.3 Results / §6 Discussion に統合するための作業メモ。

---

## 数値結果サマリ

### 測定 1: 軌跡近似誤差（Theorem 1）

| ε | kernel K | MAE |
|---|---|---|
| 2.0 | 41 | 0.0734 |
| 1.0 | 21 | 0.0676 |
| 0.5 | 11 | 0.0621 |
| 0.2 | 5  | 0.0541 |
| 0.1 | 3  | 0.0462 |

log-log slope = **0.15**（期待値 ~= 1.0）

### 測定 2: 最小解の収束（Corollary 2）

| ε | θ_ε* | ‖θ_ε* − θ*‖ |
|---|---|---|
| 2.0 | (1.587, 1.754) | 0.956 |
| 1.0 | (1.454, 1.485) | 0.664 |
| 0.5 | (1.402, 1.394) | 0.563 |
| 0.2 | (1.396, 1.340) | 0.521 |
| 0.1 | (1.370, 1.291) | 0.471 |

単調減少: **定性的に Corollary 2 を支持** ✓

---

## 核心的観察

### Theorem 1: slope = 0.15 の意味

理論は連続モリファイア φ_ε で ‖f_ε - f‖_∞ = O(ε) を主張する。
数値実験では離散二項カーネル（長さ K = 2·round(ε·10)+1）を用いており、
ε と K の対応は線形ではなく離散的。

slope が 0.15 にとどまる理由：
1. **離散化の粗さ**: K=3 が eps=0.1 の最小（これ以下は K 不変）→ ε→0 の極限が再現できない
2. **ノルムの不一致**: 理論は W^{1,∞} ノルム、測定は時間軌跡の MAE → 異なる量を比較している
3. **proxy の限界**: 時間方向畳み込みはパラメータ空間モリファイアの代理実装

ただし、MAE は ε が大きいほど大きく、ε が小さいほど小さい → **定性的な単調性は成立**。

### Corollary 2: 単調減少の意味

‖θ_ε* − θ*‖ が 0.956 から 0.471 へ単調減少 → Corollary 2 の核心（「ε→0 で θ_ε* → θ*」）を
定性的に支持。ただし eps=0.1 時点でも ‖θ_ε* − θ*‖ ≈ 0.47 と大きく、
理論上の「ε→0 で 0 に収束」が完全には見えない。

これは 500 step の最適化ステップ数が有限であることと、
カーネルの最小サイズ K=3（eps≈0.1 が下限）によるもの。

---

## 理論との正直な照合

| 主張 | 数値結果 | 判定 |
|---|---|---|
| Theorem 1: ‖f_ε - f‖ = O(ε) | slope=0.15（定量不一致） | △（定性的のみ） |
| Corollary 2: ε→0 で θ_ε*→θ* | 単調減少 0.956→0.471 | ○（定性的に成立） |

---

## 論文用テキスト案

### Results 節

**Theorem 1 (trajectory approximation):**
The mean absolute error ‖smooth_ε(z(θ*)) − z(θ*)‖ decreases monotonically
with ε, from 0.073 at ε=2.0 to 0.046 at ε=0.1. However, the log-log slope
is 0.15, below the theoretical O(ε) prediction of slope ≈ 1.0. We attribute
this discrepancy to the gap between the discrete binomial kernel (minimum
size K=3 at ε=0.1) and the continuous mollifier assumed in Theorem 1.

**Corollary 2 (minimizer convergence):**
The distance ‖θ_ε* − θ*‖ decreases monotonically as ε decreases:
0.956 (ε=2.0) → 0.664 (ε=1.0) → 0.563 (ε=0.5) → 0.521 (ε=0.2) → 0.471 (ε=0.1).
This confirms the qualitative prediction of Corollary 2 that the smoothed
minimizer approaches θ* as the smoothing width decreases.

### Discussion 節（limitation として）

The numerical implementation of the mollifier as a discrete 1D convolution
along the time axis introduces a gap between theory and experiment.
The theoretical O(ε) bound in Theorem 1 holds in the W^{1,∞} norm for
continuous mollifiers; the discrete proxy cannot reproduce this scaling
precisely, particularly at small ε where the kernel size saturates at K=3.
A tighter numerical validation would require a continuous-time parameterization
of the trajectory or a finer kernel grid. Nevertheless, the qualitative trends —
monotone decrease of approximation error and minimizer distance with ε —
are consistently observed and provide empirical support for the theoretical framework.

---

## 実験全体を通じた統一的解釈

Exp1〜5 を横断すると一貫した構造が見える：

```
raw  J:  初期値感度あり、dynamics/次元 非依存、loss→0（深い谷）、勾配ノイズ
smooth J_ε:  初期値安定、dynamics/次元 依存のバイアス、loss フロア、勾配滑らか
              ε→0 でバイアスが単調減少（Corollary 2 定性的支持）
```

Exp4 の landscape 可視化がこれを視覚的に統合し、
Exp5 が ε スケーリングとして定量化する。
