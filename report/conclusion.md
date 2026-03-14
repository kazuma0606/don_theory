# conclusion.md — §7 Conclusion 草稿

> `discussion.md` の議論を受け、論文 §7 Conclusion に統合するための草稿。
> Conclusion は新たな主張を追加せず、貢献の再提示・実験の意義・展望の3層で構成する。

---

## 英語本文案

We have presented a functional-analytic framework for optimizing
non-commutative intervention sequences by embedding intervention-induced
observables into the Banach space $W^{1,\infty}$ and applying mollifier-based smoothing.

**Theoretical contributions.**
The framework establishes four formal results:
(i) a non-commutative operator model capturing the order-dependence of
time-indexed interventions (Layer 1);
(ii) a $W^{1,\infty}$ Banach-space embedding that provides uniform, pointwise
control over operator compositions (Layer 2);
(iii) mollifier smoothing that yields a $C^\infty$ surrogate $J_\varepsilon$ whose
gradient is Lipschitz continuous — a property proved via Fréchet differentiability
in the formal Lean 4 verification (Layer 3);
(iv) existence of minimizers for $J_\varepsilon$ and their convergence to minimizers
of $J$ as $\varepsilon \to 0$ (Theorem 2, Corollary 2).

**Empirical findings.**
The numerical experiments confirm the qualitative predictions of this framework.
The symmetrized smoothed objective $J_{\varepsilon,\text{sym}}$ matches the convergence
rate of the unsmoothed objective (80%) while exhibiting markedly reduced
step-to-step gradient variation — direct numerical evidence for the
Lipschitz gradient property established in Layer 3.
Loss landscape visualization reveals that mollifier smoothing compresses the
irregular, high-frequency surface of $J$ into a smooth bowl geometry,
consistent with $W^{1,\infty}$ regularization.
The monotone decrease of $\|\theta_\varepsilon^* - \theta^*\|$ with $\varepsilon$
(from 0.956 to 0.471) provides qualitative support for Corollary 2.
The framework is robust across independently sampled dynamics generators
and state-space dimensions $d \in \{32, 64, 128\}$,
with the convergence behavior of the unsmoothed objective remaining
invariant to both factors.

**Practical implication.**
The primary benefit of the $W^{1,\infty}$ embedding is not a reduction in loss value
but a regularization of the gradient field that makes first-order optimization
tractable in landscapes rendered irregular by non-commutative operator interactions.
This benefit is most pronounced for gradient-based methods that lack curvature
information; quasi-Newton methods, having implicit access to second-order structure,
do not require external smoothing in low-dimensional settings.
As parameter dimensionality and system complexity grow — the regime most relevant
to real-world intervention planning — the advantage of providing a guaranteed
Lipschitz-continuous gradient is expected to increase.

**Broader significance.**
The proposed framework is domain-agnostic.
Although motivated by clinical intervention sequencing, the mathematical structure —
non-commutative operators acting on high-dimensional state spaces, embedded into
$W^{1,\infty}$ via mollifier smoothing — applies to any system in which
discrete, order-dependent operations interact with continuous time evolution.
The Lean 4 formal verification provides machine-checked guarantees for all
core theoretical claims, offering a level of rigor beyond standard proof-by-argument.

**Future directions.**
Immediate extensions include scaling the parameter space beyond $\theta \in \mathbb{R}^2$
to assess the smoothing benefit in high-dimensional optimization,
replacing the discrete temporal-convolution proxy with a parameter-space
mollifier for a tighter theory-experiment correspondence,
and applying the framework to structured real-world intervention datasets.
On the theoretical side, the $W^{2,\infty}$ assumption in the mollifier
convergence proof (Corollary 1) is mathematically necessary and is
explicitly stated as such; the Lean 4 proofs are available at
\url{https://github.com/kazuma0606/don_theory}.

---

## 日本語解説

### 構成の意図

Conclusion は3つの役割を持つ：

1. **再提示**（理論貢献の要約）
   - 4層の理論構造を 1 段落でコンパクトに再述。
   - Lean4 形式証明への言及を忘れない（これが他論文との最大の差別化点）。

2. **実験の意義の確定**
   - 「smooth が勝った」ではなく「Lipschitz 勾配の数値証拠を得た」という読み替えを確定。
   - Corollary 2 の定性的支持と Theorem 1 の定量的 gap について触れすぎず、
     Discussion（§6.6）で詳述済みであることを前提に簡潔に。

3. **展望**
   - θ の高次元化（最重要）
   - 離散 proxy から parameter-space mollifier への移行
   - Lean4 の open task A1（W^{2,∞} 仮定の除去）

### Limitation との役割分担

| 場所 | 書くこと |
|---|---|
| Discussion §6.7 | 具体的な限界（slope=0.15、K=3 下限、固定 lr） |
| Conclusion | 「今後の方向」として前向きに言い換える |

Conclusion で limitation を再掲しない。

### キーフレーズ

- "not a reduction in loss value but a regularization of the gradient field" — 主メッセージ
- "domain-agnostic" — 応用範囲の広さ
- "machine-checked guarantees" — Lean4 の差別化
- "as parameter dimensionality grows" — 今後の重要性を示す伏線
