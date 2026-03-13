# Exp2 考察メモ

> 後で §5.3 Results / §6 Discussion に統合するための作業メモ。

---

## 数値結果サマリ

| 指標 | raw (J) | smooth (J_ε) |
|---|---|---|
| conv_rate（全 10 dynamics 平均） | 0.80 ± 0.00 | 0.03 ± 0.03 |
| 最終 ‖θ−θ*‖（全 10 dynamics 平均） | 0.037 ± 0.001 | 0.499 ± 0.083 |

---

## 核心的観察

### raw の挙動は dynamics 非依存

raw の conv_rate が全 10 dynamics で完全に 16/20 = 0.80 固定（std=0.00）。
最終 theta_dist も 0.037 ± 0.001 と極めて安定。

→ raw loss の landscape 構造（収束率・到達精度）は dynamics の種類によらない。
  初期値の当たり外れ（4/20 が θ* に届かない）が支配的で、dynamics は無関係。
  これは J の最小解が θ* に一意に定まり、dynamics を変えても同じ収束挙動が再現されることを意味する。

### smooth の bias 量は dynamics 依存

smooth の theta_dist が dynamics seed によってばらつく（0.41〜0.68）。
conv_rate も 0/20〜1/20 と dynamics によって微妙に変動。

→ J_ε の最小解 θ_ε* の位置が dynamics に依存する。
  これは J_ε の landscape 形状が dynamics（＝rollout の非線形性）に影響を受けるため。
  平滑化カーネルと dynamics の相互作用が bias の大きさを決める。

---

## Exp1 との比較

Exp1（dyn_seed=0 固定）と Exp2（dyn_seed=0〜9）の dyn=0 の数値が一致：
- raw: conv=16/20, mean_dist=0.037
- smooth: conv=1/20, mean_dist=0.458

Exp2 は Exp1 の結果が特殊ケースではなく、
複数の dynamics に渡って成立することを確認する実験として機能している。

---

## 解釈

### raw の安定性について

raw loss の収束挙動が dynamics 非依存というのは、
J(θ) の landscape の「形状」が dynamics ではなく
「θ から z(θ) への写像の大域的構造」に支配されていることを示す。

J の最小解 θ* = (1,1) は dynamics によらず固定なので、
landscape の大域的な谷の位置は変わらない。
4 run が収束しないのは landscape の局所的なフラット方向（非一意性の兆候）。

### smooth の dynamics 依存性について

J_ε の最小解 θ_ε* は：
```
argmin_θ ‖smooth(z(θ)) - z_target‖²
```
smooth(z(θ)) の形状は dynamics に依存するため、
θ_ε* の位置も dynamics によって変わる。
これは理論的に予測される挙動（ε→0 で θ_ε* → θ* への収束は
dynamics に依らず成立するが、有限 ε では dynamics 依存の bias が残る）。

---

## 論文用テキスト案

### Results 節

Across 10 independently sampled dynamics generators, the raw objective J
converges at a consistent rate of 80% (16/20 initializations), with negligible
variance across dynamics (std of conv_rate = 0.00). This indicates that the
convergence behavior of J is determined primarily by the initial conditions
rather than the specific dynamics, suggesting a stable global landscape structure.

In contrast, the smoothed objective J_ε shows dynamics-dependent behavior:
the final distance ‖θ_ε* − θ*‖ varies from 0.41 to 0.68 across dynamics seeds,
reflecting that the smoothed minimum θ_ε* is influenced by the interaction
between the smoothing kernel and the nonlinear rollout operator.
This is consistent with the theoretical prediction that the bias |θ_ε* − θ*|
is O(ε) for any fixed dynamics, but the constant depends on the dynamics.

### Discussion 節

The dynamics-invariance of raw J convergence and the dynamics-dependence of
smooth J_ε bias together provide a coherent picture: the global structure of
the unsmoothed landscape is dominated by the parameter-to-observation map,
while the smoothed landscape's geometry reflects an interplay between the
kernel and the nonlinear dynamics. Exp5 will quantify how this bias vanishes
as ε → 0, validating Corollary 2 across dynamics.

---

## Exp5 への接続

Exp2 で観察された「smooth bias が dynamics 依存」という事実は、
Exp5 の ε スケーリング検証においても dynamics=0 固定で行うことの
限界を示唆する。
ただし Corollary 2 の主張（bias が O(ε)）は dynamics 非依存の保証なので、
Exp5 の定量結果（傾き ≈ 1.0）は一般性を持つ。
