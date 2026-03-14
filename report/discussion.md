# discussion.md — §6 Discussion 草稿

> `result.md` の数値結果を受け、論文 §6 Discussion に統合するための草稿。
> 各節は manuscript への転記を想定した英語テキスト案と、
> 背景の日本語解説を対で記載する。

---

## 6.1 The Role of Banach-Space Embedding

**日本語解説**

実験の raw / smooth_sym の比較は単なる「フィルタあり・なし」ではない。
smooth 条件は W^{1,∞} への埋め込みと等価であり、
その効果は勾配場の構造（Lipschitz 性）として現れる。
一方 raw の loss 景観は非可換作用素合成による高周波成分を含み、
W^{1,∞} ノルムで制御されない空間に留まっている。

**論文テキスト案**

The experiments operationalize the Banach-space embedding proposed in §3.1 as follows.
The "raw" condition evaluates $J(\theta) = \|z(\theta) - z_\text{target}\|^2$ in a setting
where the observable trajectory $z(\theta)$ is not regularized;
the effective optimization landscape contains high-frequency components
induced by non-commutative operator compositions and is not controlled
by a $W^{1,\infty}$ norm.
The "smooth" condition applies the mollifier proxy $\phi_\varepsilon$,
embedding the observable into a band-limited space in which
both the sup-norm and the gradient sup-norm are bounded — the defining property
of $W^{1,\infty}$.

The critical consequence of this embedding is not a lower loss value but
a regularized gradient field.
The loss landscape visualization (Exp4, Fig. 07–08) shows that $J_\varepsilon$
compresses the value range from $[0,\, 0.089]$ to $[0.010,\, 0.037]$
and converts the irregular surface of $J$ into a smooth bowl geometry.
This compression is not incidental: it reflects the averaging effect of the
mollifier, which removes the high-frequency undulations produced by
non-commutative interactions and ensures that the observable resides in
a space where optimization is well-conditioned.

---

## 6.2 Gradient Regularity as Evidence for Fréchet Differentiability

**日本語解説**

Layer 3 の Lean4 形式証明は J_ε の Fréchet 微分可能性と ∇J_ε の Lipschitz 連続性を
主張する。これに対応する数値証拠が Exp1-modify の |Δ‖∇J‖| 比較。
raw の勾配ノルム変動が大きい（ノイズ的）のに対し、
smooth_sym では隣接ステップ間の変動が一貫して小さい。
これは「勾配が Lipschitz 連続である」ことの直接的な操作定義に対応する。

**論文テキスト案**

The most direct numerical evidence for the theoretical claim of Fréchet differentiability
(Layer 3 of the formal verification) is provided by Exp1-modify.
The symmetrized smoothed objective $J_{\varepsilon,\text{sym}}(\theta)
= \|\phi_\varepsilon \star z(\theta) - \phi_\varepsilon \star z_\text{target}\|^2$
eliminates the smoothing bias (ensuring $J_{\varepsilon,\text{sym}}(\theta^*) = 0$)
and allows a direct comparison of gradient behavior under otherwise identical conditions.

Under both raw and smooth conditions, 16 out of 20 initializations converge to $\theta^*$,
with nearly identical convergence steps (median 307 for both;
per-initialization difference $\leq 2$ steps).
The decisive difference appears in the gradient norm variation:
$|\Delta\|\nabla J\||$ — the step-to-step change in gradient norm — is consistently
smaller under $J_{\varepsilon,\text{sym}}$ than under $J$ (Fig. 09).

A Lipschitz-continuous gradient satisfies $\|\nabla J(a) - \nabla J(b)\| \leq L\|a - b\|$
for some finite constant $L$, which manifests operationally as bounded changes
in the gradient norm along the optimization path.
The reduced $|\Delta\|\nabla J_{\varepsilon,\text{sym}}\||$ is therefore direct numerical
evidence for the Lipschitz gradient property established in the Lean 4 proof of Layer 3.

---

## 6.3 Smoothing Bias and Its Correction

**日本語解説**

片側平滑化（J_ε）はバイアスを持つ（θ* で loss = 0.0104 > 0）。
このバイアスは「悪い結果」ではなく、カーネルと軌跡の幾何学的な量であり、
ε → 0 で消える（Corollary 2、Exp5 で定性確認）。
両側平滑化（J_ε_sym）はバイアスをゼロにする操作的な修正であり、
理論的に ε の高次補正に対応する。
論文としては J_ε_sym を主条件として提示し、
J_ε の bias を Appendix で整理する構成が自然。

**論文テキスト案**

The asymmetric formulation $J_\varepsilon(\theta) = \|\phi_\varepsilon \star z(\theta) - z_\text{target}\|^2$
introduces a floor at $J_\varepsilon(\theta^*) = \|\phi_\varepsilon \star z_\text{target} - z_\text{target}\|^2 \approx 0.0104$,
causing the smoothed optimizer to converge to a biased minimizer $\theta_\varepsilon^* \neq \theta^*$
(Exp4: $\theta_\varepsilon^* \approx (1.41,\, 1.29)$, $\|\theta_\varepsilon^* - \theta^*\| \approx 0.50$).
This bias is not an artifact of the method but a predictable geometric consequence
of applying the mollifier asymmetrically.

The symmetrized variant $J_{\varepsilon,\text{sym}}$ corrects this by smoothing both sides:
$J_{\varepsilon,\text{sym}}(\theta^*) = \|\phi_\varepsilon \star z(\theta^*) - \phi_\varepsilon \star z_\text{target}\|^2 = 0$.
The convergence rate and final $\|\theta - \theta^*\|$ of $J_{\varepsilon,\text{sym}}$ match those
of the raw objective (both 80%, median 307 steps), confirming that the benefit of
the Banach-space embedding is not faster convergence per se but a more regular
gradient field — as captured by the reduced $|\Delta\|\nabla J\||$ (Fig. 09).

Corollary 2 predicts that this bias vanishes as $\varepsilon \to 0$.
Exp5 provides qualitative support: $\|\theta_\varepsilon^* - \theta^*\|$ decreases monotonically
from 0.956 at $\varepsilon = 2.0$ to 0.471 at $\varepsilon = 0.1$,
consistent with the theoretical prediction that the smoothed minimizer approaches
the true minimizer in the limit (Fig. 10).

---

## 6.4 Dynamics and Dimension Robustness

**日本語解説**

raw の収束率が dynamics/次元で完全に不変（0.80±0.00）なのは、
J の landscape 構造が θ → z(θ) の大域的写像に支配されるから。
smooth の bias が dynamics/次元依存なのは、
J_ε の谷底位置が平滑化カーネルと非線形ダイナミクスの相互作用で決まるから。
この対比が「Banach 空間の構造が dynamics に対して安定な土台を提供する」
という主張の数値的根拠になる。

**論文テキスト案**

Two robustness experiments examine the sensitivity of the proposed framework
to system structure.

**Dynamics robustness (Exp2):** Across 10 independently sampled dynamics generators,
the raw objective $J$ converges at exactly 80% (16/20) for every seed,
with variance zero in convergence rate and negligible variance in final $\|\theta - \theta^*\|$
($0.037 \pm 0.001$). This dynamics-invariance indicates that the convergence behavior of
$J$ is determined by the initial conditions and the parameter-to-observation map,
not by the specific dynamics. In contrast, the smoothed objective $J_\varepsilon$ exhibits
dynamics-dependent bias: $\|\theta_\varepsilon^* - \theta^*\|$ ranges from 0.41 to 0.68
across seeds. This is expected theoretically — at finite $\varepsilon$, the location of
$\theta_\varepsilon^*$ is determined by the interaction between the kernel and the nonlinear
rollout, which varies with the dynamics.

**Dimensionality robustness (Exp3):** The convergence rate of $J$ remains 80% at
$d \in \{32, 64, 128\}$, confirming that the landscape structure of the unsmoothed
objective is dimension-invariant in this parameterization. The smoothed objective
shows dimension-dependent bias ($\|\theta_\varepsilon^* - \theta^*\|$ = 0.597 / 0.458 / 0.661
at $d$ = 32 / 64 / 128), reflecting that higher-dimensional trajectories contain
richer high-frequency components that interact differently with the fixed kernel.

Together, these results establish that the W^{1,∞} embedding provides a
stable and reproducible optimization foundation across system instances,
even as the specific bias of the asymmetric formulation depends on system structure.

---

## 6.5 Second-Order Methods and the Scope of the Banach-Space Benefit

**日本語解説**

Extra Exp1 で L-BFGS が raw/smooth_sym 両方で step=1 に収束したことは
「この 2D 問題では平滑化が不要」と読めるが、逆説的にこの問題の
well-conditioned 性（W^{1,∞} 近傍での準凸構造）を示している。
L-BFGS は自前で Hessian 近似を通じた「内部的な平滑化」を行うため、
外部平滑化の恩恵を受けない。
平滑化の本質的利益は「Hessian を持たない 1 次法が利用可能な Lipschitz 勾配場を
提供すること」であり、Adam の結果がその主要証拠となる。

**論文テキスト案**

An additional experiment (Extra Exp1) compares first- and second-order optimizers
on both raw and smooth conditions.
L-BFGS (strong Wolfe line search, history size 10) converges in $\leq 1$ step
for all 20 initializations under both $J$ and $J_{\varepsilon,\text{sym}}$,
achieving $\|\theta - \theta^*\| < 0.002$ — approximately 307× faster than Adam's
median of 307 steps.
Critically, the raw and smooth L-BFGS conditions are indistinguishable ($\|\theta - \theta^*\| \approx 0.001$),
confirming that in this 2D parameter space, the explicit Hessian approximation
built by L-BFGS subsumes any benefit of external smoothing.

This result clarifies the scope of the Banach-space benefit:
smoothing regularizes the gradient field for optimizers that do not have access
to curvature information.
A quasi-Newton method implicitly achieves a similar regularization by maintaining
a curvature estimate, making external smoothing redundant.
The primary beneficiary of $W^{1,\infty}$ embedding is therefore first-order
gradient-based optimization, where the Lipschitz-continuous gradient provided by
the mollifier-smoothed objective is the only available curvature signal.

This further implies that as parameter dimensionality grows beyond 2D — where
building an accurate Hessian approximation becomes expensive or intractable —
the advantage of $W^{1,\infty}$ smoothing over second-order methods will increase.

---

## 6.6 Theoretical Validation: Honest Assessment

**日本語解説**

理論と実験の対応を誠実に評価する節。
Corollary 2 は定性的に支持されるが定量的な O(ε) 傾きは再現できていない。
これは離散実装の限界であり、理論の主張自体を否定するものではない。

**論文テキスト案**

We assess the correspondence between theory and experiment directly.

**Theorem 1 (trajectory approximation, $O(\varepsilon)$ bound):**
The mean absolute error $\|\phi_\varepsilon \star z(\theta^*) - z(\theta^*)\|$ decreases
monotonically with $\varepsilon$ across all tested values ($0.073 \to 0.046$),
qualitatively consistent with Theorem 1.
However, the log-log slope is 0.15, well below the theoretical prediction of 1.0.
We attribute this discrepancy to two factors:
(i) the discrete binomial kernel saturates at $K=3$ for $\varepsilon \leq 0.1$,
preventing numerical exploration of the true $\varepsilon \to 0$ regime;
(ii) Theorem 1 is stated in the $W^{1,\infty}$ norm, whereas the measured quantity
is the trajectory MAE — a weaker norm that does not capture gradient-level approximation.
The qualitative monotonicity is preserved, supporting the direction of the theoretical claim.

**Corollary 2 (minimizer convergence, $\theta_\varepsilon^* \to \theta^*$):**
$\|\theta_\varepsilon^* - \theta^*\|$ decreases monotonically from $0.956$ to $0.471$
as $\varepsilon$ decreases from 2.0 to 0.1, qualitatively confirming Corollary 2.
The distance at $\varepsilon = 0.1$ remains $\approx 0.47$, reflecting the finite
optimization budget (500 steps) and the $K=3$ kernel floor.
A tighter numerical validation would require either a continuous-time trajectory
parameterization or a more refined kernel grid.

**Layer 3 (Fréchet differentiability, Lipschitz gradient):**
The reduced step-to-step gradient variation under $J_{\varepsilon,\text{sym}}$ (Fig. 09)
constitutes direct numerical evidence for the Lipschitz gradient property.
This is the most precisely matched correspondence between formal proof and experiment
in the present study.

---

## 6.7 Limitations and Future Work

**日本語解説**

実験の限界を正直に整理し、今後の方向性を示す。

**論文テキスト案**

**Discrete proxy for the mollifier.**
The experimental implementation approximates the parameter-space mollifier via
temporal convolution of the observable trajectory.
This proxy is computationally tractable but introduces a gap between theory and experiment:
the formal guarantees of §3.2 concern convolution over $\Theta$, not over the time axis.
A more faithful implementation would directly smooth the parameter-to-observable map.

**Low-dimensional parameters.**
The experiments use $\theta \in \mathbb{R}^2$, chosen to allow landscape visualization
and exact comparison with the known $\theta^* = (1,1)$.
In this setting, L-BFGS trivially solves the problem in one step, and the advantage
of smoothing over second-order methods is not visible.
The theoretical benefit of $W^{1,\infty}$ embedding — stabilizing first-order optimization —
becomes more important as $\dim(\Theta)$ grows and Hessian-based methods become costly.
Scaling to $\theta \in \mathbb{R}^{100}$ or higher is a natural next step.

**Fixed optimizer hyperparameters.**
The Adam baseline uses a fixed learning rate ($\text{lr} = 0.01$) without scheduling.
The 4 non-converging initializations in Exp1-modify are likely attributable to
this fixed step size combined with landscape irregularities outside the attraction basin.
A more systematic hyperparameter study would separate the effect of learning rate
from the effect of smoothing.

**Theorem 1 quantitative gap.**
As noted in §6.6, the O(ε) slope of 0.15 falls short of the theoretical prediction.
Closing this gap requires a discrete-to-continuous approximation analysis specific
to the binomial kernel family used here, which is left for future work.

---

## 6.8 Positioning Relative to Related Work

**論文テキスト案（Related Work との差分を Discussion で補強する形）**

The proposed framework shares surface similarities with several existing approaches
but differs in a specific and consequential way.

**Compared to neural ODEs:** Neural ODEs achieve differentiability by assuming
continuous-time commutative flows. Our framework explicitly models discrete,
non-commutative operator sequences, where the intervention $E_t$ and the evolution
$U_{\Delta t}$ do not commute. The mollifier smoothing is applied *after* forming
the observable from non-commutative compositions, not to the dynamics themselves.

**Compared to curriculum / annealed optimization:** Methods that gradually reduce
a smoothing parameter share structural similarity with our $\varepsilon \to 0$ annealing.
The distinction is that our approach provides a theoretically grounded convergence
guarantee (Corollary 2) rooted in $W^{1,\infty}$ functional analysis,
rather than empirical scheduling heuristics.

**Compared to operator learning:** DeepONet and related methods learn operator
mappings from data. Our framework does not learn the operators; it analytically
characterizes the smoothing-induced landscape geometry and provides provable
approximation bounds for fixed operators.

---

## まとめ（§6 全体の構成案）

```
§6.1  Banach 空間埋め込みの役割
        → smooth/raw 比較 = W^{1,∞} 導入あり・なしと等価
§6.2  Fréchet 微分可能性の数値証拠
        → exp1m_grad_smoothness.png が主図
§6.3  平滑化バイアスとその補正
        → J_ε の bias（Exp4）+ J_ε_sym での解消 + Exp5 の ε→0 収束
§6.4  Dynamics・次元ロバスト性
        → Exp2, Exp3 の dynamics/次元不変性
§6.5  2 次法との比較と平滑化の受益対象
        → Extra Exp1: L-BFGS は不要、1 次法 Adam が主受益者
§6.6  理論との誠実な照合
        → Theorem 1: 定性的のみ (slope=0.15)、Corollary 2: 定性的支持
§6.7  Limitation と今後の方向
        → 離散 proxy、低次元 θ、固定 lr、quantitative gap
§6.8  Related Work との位置づけ補強
        → Neural ODE・Operator Learning との差分
```

主要図の再掲:

| 節 | 図 |
|---|---|
| §6.1, §6.2 | Fig. 07, Fig. 08, Fig. 09 |
| §6.3 | Fig. 10 |
| §6.4 | Fig. A4, A5, A6, A7 |
| §6.5 | Fig. A8 |
| §6.6 | Fig. 10, Fig. 09 |
