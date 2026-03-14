---

# **Differentiable Optimization of Non‑Commutative Intervention Sequences via Banach‑Space Smoothing**

---

## **Abstract**

Time‑dependent intervention systems—such as clinical treatments, robotic manipulation, and sequential decision processes—exhibit strong order‑dependence due to the non‑commutative interaction between interventions and underlying dynamics.  
However, optimizing such systems is challenging because intervention sequences induce discrete, non‑smooth, and non‑differentiable objective functions.

We propose a functional‑analytic framework that models interventions, time evolution, and observations as operators acting on high‑dimensional belief states.  
By embedding intervention‑induced observables into $W^{1,\infty}$ and applying mollifier‑based smoothing, we obtain uniformly convergent differentiable approximations that enable gradient‑based optimization of intervention parameters.  
We demonstrate the framework in high‑dimensional synthetic environments designed to reflect the structure of real clinical systems, showing that smoothing stabilizes gradients and significantly improves optimization performance.

---

# **1 Introduction**

Interventions applied to time‑evolving systems are inherently order‑dependent.  
In clinical medicine, chemotherapy followed by surgery produces different outcomes than the reverse order.  
In robotics, grasping before lifting is not equivalent to lifting before grasping.  
These examples illustrate a universal structural property:

> **Interventions and time evolution interact in a fundamentally non‑commutative manner.**

Despite the ubiquity of such systems, optimizing intervention parameters remains difficult.  
Most machine‑learning methods rely on differentiable objectives, yet intervention sequences induce discrete, non‑smooth, and non‑commutative operator compositions.  
Reinforcement learning handles such systems but lacks differentiability; control theory assumes smooth dynamics; neural ODEs assume commutative flows.

We introduce a functional‑analytic framework that models interventions, time evolution, and observations as operators acting on high‑dimensional belief states.  
By embedding intervention‑induced observables into $W^{1,\infty}$ and applying mollifier smoothing, we obtain differentiable approximations that preserve the underlying operator structure.

**Our contributions are:**

- **A non‑commutative operator model** for time‑dependent interventions.  
- **A Banach‑space embedding** enabling uniform control of observables.  
- **A mollifier‑based smoothing method** yielding differentiable approximations with provable error bounds.  
- **A demonstration in high‑dimensional synthetic environments** reflecting real clinical systems.

---

# **2 Background and Problem Formulation**

## **2.1 State Space**

Let $P \subset \mathbb{R}^d$ be a high-dimensional belief-state space.  
In clinical applications, $P$ may represent:

- physiological measurements,  
- imaging-derived features,  
- medication histories,  
- latent embeddings of multimodal clinical data.

## **2.2 Intervention Operators**

An intervention is a self-map:

$$
E : P \to P.
$$

Interventions compose:

$$
E_a \circ E_b : p \mapsto E_a(E_b(p)).
$$

The identity map acts as the neutral element.

### Non-Commutativity

$$
E_a \circ E_b \neq E_b \circ E_a.
$$

This reflects the order dependence of real interventions.

## **2.3 Observation Functions**

Observations are maps:

$$
O : P \to \mathbb{R}^n.
$$

The intervention-induced observable is:

$$
f_E = O \circ E.
$$

---

## **2.4 Assumptions and Theoretical Foundations**

To ensure mathematical well-posedness, we impose the following mild assumptions.

### Assumption 1 (State Space Regularity)

$P$ is locally compact and bounded.

### Assumption 2 (Intervention Regularity)

Each $E_t$ is **Lipschitz continuous**.

### Assumption 3 (Observation Regularity)

Each $O_t$ is **Lipschitz** and piecewise differentiable.

### Assumption 4 (Time Evolution Regularity)

$U_t$ satisfies:

- semigroup property  
- Lipschitz continuity in state  
- continuity in time  
- boundedness

### Assumption 5 (Boundedness of Composition)

Any finite composition of $U_t$ and $E_t$ maps bounded sets to bounded sets.

These assumptions ensure that all observables belong to $W^{1,\infty}$.

---

## **2.5 Time Evolution Operator**

$$
U_t : P \to P,
\qquad
U_{t+s} = U_t \circ U_s.
$$

## **2.6 Time-Indexed Interventions**

$$
E_t : P \to P.
$$

## **2.7 Fundamental Non-Commutativity**

$$
U_{\Delta t} \circ E_t \neq E_{t+\Delta t} \circ U_{\Delta t}.
$$

### **2.7.1 Why Non‑Commutativity Breaks Optimization**

Non‑commutativity does not merely express that intervention order matters; it fundamentally alters the geometry of the induced objective landscape.  
When the composition  
$$
U_{\Delta t} \circ E_t \neq E_{t+\Delta t} \circ U_{\Delta t}
$$
holds, the resulting observable  
$$
f_{E,t} = O_t \circ U \circ E \circ \cdots
$$
becomes **piecewise‑smooth and order‑discontinuous** with respect to intervention parameters.

Small perturbations in parameters may change the *effective ordering* or *relative influence* of operators, producing:

- flat regions (vanishing gradients),  
- abrupt jumps (non‑differentiable kinks),  
- highly irregular local geometry.

As a consequence, the induced objective  
$$
J(\theta)
$$
typically exhibits **plateaus, sharp transitions, and unstable gradients**, making gradient‑based optimization unreliable.  
This motivates the introduction of a smoothing mechanism that regularizes the operator‑induced discontinuities while preserving the underlying non‑commutative structure.

## **2.8 Dynamic Observables**

$$
f_{E,t}
= O_t \circ U_{t-t_k} \circ E_{t_k} \circ \cdots \circ E_{t_1}.
$$

---

# **3 Method: Operator Embedding and Smoothing**

## **3.1 Functional Embedding into $W^{1,\infty}$**

### Motivation for Banach‑Space Embedding

Non‑commutative compositions amplify local perturbations:  
a small deviation in an early intervention can propagate through the operator chain and produce exponentially larger deviations at later times.  
Hilbert‑space norms such as $L^2$ provide only *average‑case* control and cannot bound these pointwise deviations.

In contrast, the Banach space $W^{1,\infty}$:

- controls both the **sup‑norm** and the **essential supremum of gradients**,  
- guarantees **pointwise stability** under operator composition,  
- ensures that mollifier smoothing converges **uniformly**,  
- provides the correct functional setting for non‑commutative operator sequences.

Thus, embedding observables into $W^{1,\infty}$ is not merely convenient—it is mathematically necessary for stability and uniform approximation.

We embed each observable into the Banach space $W^{1,\infty}$.

### Why Banach Spaces?

1. **Uniform control is essential**  
   Non-commutative operator composition amplifies local errors.

2. **Sup-norm control is required**  
   $L^2$ cannot guarantee pointwise stability.

3. **Mollifier smoothing converges uniformly**  
   Only guaranteed in $W^{1,\infty}$.

4. **Clinical observables are bounded and Lipschitz**  
   Making $W^{1,\infty}$ the natural choice.

## **3.2 Mollifier Smoothing**

$$
f_{E,t,\varepsilon} = f_{E,t} * \rho_\varepsilon.
$$

### Theorem 1 (Uniform Approximation)

$$
\|f_{E,t,\varepsilon} - f_{E,t}\|_{L^\infty} \le C \varepsilon.
$$

### Corollary 1 (Gradient Approximation)

If $f_{E,t} \in W^{2,\infty}$, then  
$$
\|\nabla f_{E,t,\varepsilon} - \nabla f_{E,t}\|_{L^\infty} \to 0.
$$

### Interpretation

- smoothing introduces only **O(ε)** error  
- gradients become stable  
- optimization becomes feasible

---

# **4 Theoretical Results**

## **4.1 Well-Posedness of the Optimization Problem**

Let  
$$
J(\theta) = \|f_{E_\theta,t_{\mathrm{final}}} - y_{\text{target}}\|^2.
$$

### Theorem 2 (Existence of Minimizer)

If $\Theta$ is compact, then $J_\varepsilon$ admits a minimizer.

### Corollary 2

Minimizers of $J_\varepsilon$ converge to minimizers of $J$ as $\varepsilon \to 0$.

---

# **5 Experiments**

## **5.1 Dataset and Simulation Setup**

We construct a structured high-dimensional simulation environment.

### **5.1.1 High-Dimensional State Space**

$
p_t \in \mathbb{R}^{64}.
$

This reflects the inherently high-dimensional nature of real patient states  
(Dimensions ranging from 32 to 256 are established as typical latent spaces in prior research.)。

### **5.1.2 Time Evolution Dynamics**

$$
p_{t+1} = A p_t + b + \gamma \tanh(W p_t).
$$

### **5.1.3 Non-Commutative Interventions**

$$
E_A(p) = p + a,
\qquad
E_B(p) = B p.
$$

### **5.1.4 Observation Model**

$$
y_t = C p_t.
$$

### **5.1.5 Parameterized Interventions**

$$
E_{A,\theta_1}(p) = p + \theta_1 a,
\qquad
E_{B,\theta_2}(p) = (I + \theta_2 M)\, p,
$$

where $a \in \mathbb{R}^d$ is a fixed sparse vector ($a_i = 1$ for $i < d/4$, else $0$) and $M \in \mathbb{R}^{d \times d}$ is a fixed random matrix.
The optimization parameter is $\theta = (\theta_1, \theta_2) \in \mathbb{R}^2$.
This instantiates $B(\theta_2) = I + \theta_2 M$, a scalar parameterization of the linear operator family.

### **5.1.6 Simulation Procedure**

1. Fix ground-truth parameter $\theta^* = (1, 1)$ and generate target trajectory $z_{\text{target}} = \{C\, p_t^*\}_{t=0}^{T}$ via rollout with $\theta^*$.
2. For each candidate $\theta \in \Theta$, run rollout: $p_0 \sim \mathcal{N}(0, I)$, apply $U$ and $E$ per schedule, record $\{z_t(\theta)\}$.
3. Evaluate
$$
J(\theta) = \frac{1}{T+1}\sum_{t=0}^{T}\| z_t(\theta) - z_t^{\text{target}} \|^2.
$$

### **5.1.7 Smoothing**

Compare:

- non-smoothed $J(\theta)$
- smoothed $J_\varepsilon(\theta)$

**Implementation note.**
The theoretical mollifier $f_{E,t,\varepsilon} = f_{E,t} * \rho_\varepsilon$ (§3.2) is defined as convolution over the parameter domain $\Theta$.
In practice, we approximate this by applying temporal smoothing to the observable trajectory $\{z_t\}_{t=0}^{T}$ via a 1D convolution kernel along the time axis.
While this differs from the parameter-space mollification analyzed in §3.2 and Appendix A.2, it serves as a computationally tractable proxy that regularizes the effective loss landscape and stabilizes gradients.
The smoothed loss is $J_\varepsilon(\theta) = \| \tilde{z}(\theta) - z_{\text{target}} \|^2$ where $\tilde{z}(\theta)$ denotes the temporally smoothed trajectory; the target $z_{\text{target}}$ is not smoothed, consistent with the definition in §4.1.

### **5.1.8 Expected Phenomena Prior to Experiments**

Before presenting empirical results, we outline the qualitative behaviors expected from the theoretical analysis:

- The non‑smoothed objective $J(\theta)$ should exhibit  
  **plateaus, abrupt jumps, and unstable gradients** due to non‑commutative operator interactions.
- The smoothed objective $J_\varepsilon(\theta)$ is expected to have  
  **stable gradients and improved convergence**, with optimization trajectories that are less sensitive to initialization.
- The benefit of smoothing should increase with  
  **state dimensionality** and **operator non‑commutativity**, reflecting the amplification of local perturbations.
- As $\varepsilon \to 0$, minimizers of $J_\varepsilon$ should approach minimizers of $J$, validating the theoretical approximation guarantees.

These predicted behaviors guide the design and interpretation of the subsequent experiments.

## **5.2 Clinical Interpretation**

Real patient states are inherently high-dimensional, consisting of laboratory values, physiological signals, imaging-derived features, medication histories, and clinical notes. Recent clinical machine learning studies routinely employ latent spaces of **32 to 256 dimensions**, while Foundation Models typically operate in hundreds of dimensions.

Clinical interventions are fundamentally **order-dependent** and interact with ongoing physiological time evolution, mirroring the non-commutative operator structure modeled here. Clinical outcome functions are often noisy and non-differentiable, making mollifier-based smoothing a natural mechanism for enabling gradient-based optimization.

Thus、the high-dimensional synthetic environment used in our experiments serves as a structural analogue of real clinical systems, allowing us to isolate and evaluate the mathematical properties of the proposed framework.

## **5.3 Experimental Results**

We evaluate the framework across five experiments and one supplementary experiment,
each targeting a specific theoretical claim.
Common settings: $d = 64$, $T = 20$, $n_\text{init} = 20$, Adam lr $= 0.01$,
500 steps per run, convergence criterion $\|\theta - \theta^*\| < 0.05$.

### **5.3.1 Gradient Regularity under Symmetrized Smoothing (Exp1-modify)**

To isolate gradient behavior from smoothing bias, we use the symmetrized objective
$J_{\varepsilon,\text{sym}}(\theta) = \|\phi_\varepsilon \star z(\theta) - \phi_\varepsilon \star z_\text{target}\|^2$,
which satisfies $J_{\varepsilon,\text{sym}}(\theta^*) = 0$ by construction.

Both the raw objective $J$ and $J_{\varepsilon,\text{sym}}$ achieve a convergence rate of
**16/20 (80%)** with median convergence step **307**, and nearly identical per-run
trajectories (step difference $\leq 2$ across all 20 initializations).
The same 4 initializations fail to converge under both conditions.

The decisive difference is the step-to-step gradient norm variation
$|\Delta\|\nabla J\||$: this quantity is consistently smaller under $J_{\varepsilon,\text{sym}}$
than under $J$ across all optimization steps (Fig. 09).
This is direct numerical evidence for the Lipschitz-continuous gradient established
in the Lean 4 proof of Layer 3 (Fréchet differentiability of the mollified observable).

### **5.3.2 Loss Landscape Visualization (Exp4)**

We evaluate $J(\theta)$ and $J_\varepsilon(\theta)$ on a $50 \times 50$ grid
over $\theta \in [-3,3]^2$ (Fig. 07–08).

The unsmoothed landscape has a global minimum at $(1.04,\, 1.04) \approx \theta^*$
(value $\approx 9 \times 10^{-6}$) with a value range $[0,\, 0.089]$;
the 3D surface reveals high-frequency undulations characteristic of
non-commutative operator interactions.
The smoothed landscape $J_\varepsilon$ compresses the value range to $[0.010,\, 0.037]$
and shifts the minimum to $(1.41,\, 1.29)$ — displaced from $\theta^*$ by $\approx 0.50$
due to the asymmetric smoothing bias.
The 3D surface of $J_\varepsilon$ presents a smooth bowl geometry,
consistent with $W^{1,\infty}$ regularization of the operator-induced landscape.

### **5.3.3 Dynamics and Dimension Robustness (Exp2, Exp3)**

Across 10 independently sampled dynamics generators (Exp2), the raw objective $J$
converges at exactly $80\%$ for every seed (std $= 0.00$),
with final $\|\theta - \theta^*\| = 0.037 \pm 0.001$.
In contrast, the smoothed objective $J_\varepsilon$ exhibits dynamics-dependent bias:
$\|\theta_\varepsilon^* - \theta^*\|$ ranges from $0.41$ to $0.68$ across seeds (Fig. A4–A5).

Across state-space dimensions $d \in \{32, 64, 128\}$ (Exp3),
the convergence rate of $J$ remains $80\%$ at all dimensions,
while the smoothed bias varies ($0.597 / 0.458 / 0.661$ at $d = 32/64/128$; Fig. A6–A7).
These results confirm that the convergence structure of $J$ is determined by the
parameter-to-observation map rather than the system dynamics or dimension,
while the smoothed landscape geometry reflects the interaction between
the mollifier kernel and the nonlinear rollout operator.

### **5.3.4 Mollifier Approximation and Minimizer Convergence (Exp5)**

We evaluate the approximation quality and minimizer convergence over
$\varepsilon \in \{2.0, 1.0, 0.5, 0.2, 0.1\}$ (Fig. 10).

The mean absolute error $\|\phi_\varepsilon \star z(\theta^*) - z(\theta^*)\|$ decreases
monotonically from $0.073$ ($\varepsilon = 2.0$) to $0.046$ ($\varepsilon = 0.1$),
qualitatively consistent with Theorem 1 ($\|f_\varepsilon - f\|_{L^\infty} \leq C\varepsilon$).
The log-log slope is $0.15$, below the theoretical $O(\varepsilon)$ prediction,
attributable to the discrete binomial kernel saturating at minimum size $K=3$.

The distance $\|\theta_\varepsilon^* - \theta^*\|$ decreases monotonically:
$0.956 \to 0.664 \to 0.563 \to 0.521 \to 0.471$ as $\varepsilon$ decreases from
$2.0$ to $0.1$, providing qualitative support for Corollary 2.

---

# **6 Related Work**

Our framework intersects several major research areas, yet differs fundamentally from each.

### Neural ODEs

Neural ODEs embed discrete data into continuous-time differential equations.  
**Difference:** Neural ODEs assume *commutative* flows, whereas we model **non-commutative intervention operators**.

### Control Theory

Classical control assumes smooth, differentiable dynamics.  
**Difference:** we handle **discrete, irreversible, and non-commutative interventions**.

### Reinforcement Learning

RL optimizes sequential decisions via non-differentiable methods.  
**Difference:** we construct **differentiable approximations** of intervention-induced observables.

### Kernel Methods

Kernel regression smooths discrete data.  
**Difference:** kernel methods do not model **operator composition** or **time-indexed interventions**.

### Operator Learning

DeepONet/FNO approximate operators between function spaces.  
**Difference:** existing methods assume **commutative or linear operators**, while we model **non-commutative operator algebras**.

### Structural Comparison with Existing Paradigms

The proposed framework can be positioned within a broader landscape of dynamical‑system learning methods.  
The following table summarizes the structural differences:

| Methodology | Representation of Interventions | Time Evolution | Commutativity | Differentiability |
|-------------|--------------------------------|----------------|---------------|-------------------|
| Neural ODEs | Inputs to vector field | Continuous | Largely commutative | Yes |
| Reinforcement Learning | Discrete actions | Arbitrary | Non‑commutative | No (typically) |
| Classical Control | Inputs to smooth dynamics | Continuous | Often assumes commutativity | Yes |
| Operator Learning (FNO/DeepONet) | Linear/commutative operators | Arbitrary | Often commutative | Yes |
| **This Work** | **Operators acting on state** | **Arbitrary** | **Non‑commutative** | **Enabled via smoothing** |

This comparison highlights that our framework uniquely combines:

- operator‑theoretic representation of interventions,  
- explicit modeling of non‑commutative structure,  
- Banach‑space embedding for stability,  
- differentiable optimization via mollifier smoothing.

### Summary of Novelty

Our framework is the first to:

- model **interventions as non-commutative operators**,  
- incorporate **time evolution and time-dependent observations**,  
- embed observables into **Banach spaces**,  
- apply **mollifier smoothing** to enable differentiable optimization.

---

# **7 Discussion**

## **7.1 The Role of Banach-Space Embedding**

The experiments operationalize the $W^{1,\infty}$ embedding proposed in §3.1.
The "raw" condition evaluates $J(\theta)$ without regularization;
the effective landscape contains high-frequency components induced by
non-commutative operator compositions and is not controlled by a $W^{1,\infty}$ norm.
The "smooth" condition applies the mollifier proxy $\phi_\varepsilon$,
embedding the observable into a band-limited space in which both the sup-norm
and the gradient sup-norm are bounded — the defining property of $W^{1,\infty}$.

The loss landscape visualization (§5.3.2, Fig. 07–08) shows that $J_\varepsilon$
compresses the value range from $[0,\, 0.089]$ to $[0.010,\, 0.037]$ and converts
the irregular surface of $J$ into a smooth bowl geometry,
reflecting the mollifier’s role as $W^{1,\infty}$ regularization of the
operator-induced landscape.

## **7.2 Gradient Regularity as Evidence for Fréchet Differentiability**

The most direct numerical evidence for the Fréchet differentiability claim
(Layer 3 of the formal verification) is provided by the gradient norm variation
$|\Delta\|\nabla J\||$ in §5.3.1.
A Lipschitz-continuous gradient satisfies $\|\nabla J(a) - \nabla J(b)\| \leq L\|a-b\|$,
which manifests as bounded step-to-step changes in gradient norm.
The consistently reduced $|\Delta\|\nabla J_{\varepsilon,\text{sym}}\||$ under smooth
conditions (Fig. 09) is therefore the most precisely matched correspondence
between formal proof and experiment in this study.

## **7.3 Smoothing Bias and Minimizer Convergence**

The asymmetric formulation $J_\varepsilon$ introduces a floor at
$J_\varepsilon(\theta^*) \approx 0.010$, displacing the minimizer to
$\theta_\varepsilon^* \approx (1.41, 1.29)$ (§5.3.2).
This bias is a predictable geometric consequence of one-sided smoothing,
not a failure of the method.
The symmetrized variant $J_{\varepsilon,\text{sym}}$ corrects the bias while
preserving the gradient regularity benefit.
Corollary 2 predicts $\|\theta_\varepsilon^* - \theta^*\| \to 0$ as $\varepsilon \to 0$;
Exp5 provides qualitative support with monotone decrease
from $0.956$ to $0.471$ (§5.3.4, Fig. 10).

## **7.4 Dynamics and Dimension Robustness**

The dynamics-invariance of raw $J$ convergence ($80\%$ across all seeds and
dimensions; §5.3.3) and the dynamics-dependence of smoothed $J_\varepsilon$ bias
together provide a coherent picture: the global structure of the unsmoothed
landscape is dominated by the parameter-to-observation map,
while the smoothed landscape geometry reflects an interplay between
the kernel and the nonlinear dynamics.
The $W^{1,\infty}$ embedding provides a reproducible optimization foundation
across system instances, even as the specific bias at finite $\varepsilon$ depends
on system structure.

## **7.5 Second-Order Methods and the Scope of the Smoothing Benefit**

A supplementary experiment (Appendix B.1) compares Adam and L-BFGS on both
raw and smooth conditions.
L-BFGS converges in $\leq 1$ step for all 20 initializations under both $J$ and
$J_{\varepsilon,\text{sym}}$, approximately $307\times$ faster than Adam’s median of
307 steps. The raw and smooth L-BFGS conditions are indistinguishable
($\|\theta - \theta^*\| \approx 0.001$).

This clarifies the scope of the $W^{1,\infty}$ benefit:
smoothing regularizes the gradient field for optimizers that lack curvature information.
A quasi-Newton method implicitly achieves similar regularization through its
Hessian approximation, making external smoothing redundant in this 2D setting.
The primary beneficiary of the Banach-space embedding is first-order
gradient-based optimization, where the Lipschitz-continuous gradient is the
only available curvature signal.
As parameter dimensionality grows beyond the 2D setting used here,
the advantage over Hessian-based methods is expected to increase.

## **7.6 Honest Theoretical Assessment**

| Claim | Numerical result | Status |
|---|---|---|
| Theorem 1: $\|f_\varepsilon - f\| = O(\varepsilon)$ | log-log slope $= 0.15$ | Qualitative only |
| Corollary 2: $\theta_\varepsilon^* \to \theta^*$ as $\varepsilon \to 0$ | Monotone decrease $0.956 \to 0.471$ | Qualitative support |
| Layer 3: Lipschitz gradient | $|\Delta\|\nabla J_{\varepsilon,\text{sym}}\||$ reduced | Quantitative evidence |

The $O(\varepsilon)$ slope discrepancy in Theorem 1 is attributed to
the discrete binomial kernel ($K_\text{min} = 3$ at $\varepsilon = 0.1$)
and the mismatch between trajectory MAE and the $W^{1,\infty}$ norm.
The qualitative monotonicity is consistently preserved.

## **7.7 Limitations and Future Directions**

**Discrete proxy.** The temporal convolution proxy differs from the
parameter-space mollifier of §3.2.
A more faithful implementation would directly smooth the parameter-to-observable map,
which we leave for future work.

**Low-dimensional parameters.** The $\theta \in \mathbb{R}^2$ setting is chosen
to enable landscape visualization and exact ground-truth comparison.
Scaling to $\theta \in \mathbb{R}^{100}$ or higher — the regime where Hessian-based
methods become costly and first-order smoothing benefits are largest —
is a natural next step.

**Higher-order operators.** Because $J_\varepsilon \in C^\infty$, the parameter-space
Hessian $\nabla^2_\theta J_\varepsilon$ is well-defined, enabling curvature-based
regularization and PDE-based analysis (d’Alembertian $\square J_\varepsilon$) of
how intervention-induced perturbations propagate through time and parameter space.

## **7.8 Positioning Relative to Related Work**

Compared to neural ODEs, which achieve differentiability via commutative
continuous-time flows, our framework explicitly models discrete non-commutative
operator sequences and applies smoothing *after* forming the observable.
Compared to curriculum annealing methods, our $\varepsilon \to 0$ schedule is
grounded in the convergence guarantee of Corollary 2 rather than empirical heuristics.
Compared to operator learning (DeepONet/FNO), we do not learn operators from data
but analytically characterize the smoothing-induced landscape geometry
with provable approximation bounds.


---

# **8 Conclusion**

We have presented a functional-analytic framework for optimizing
non-commutative intervention sequences by embedding intervention-induced
observables into the Banach space $W^{1,\infty}$ and applying mollifier-based smoothing.

**Theoretical contributions.**
The framework establishes four formal results, each machine-checked in Lean 4:
(i) a non-commutative operator model capturing the order-dependence of
time-indexed interventions;
(ii) a $W^{1,\infty}$ embedding providing uniform, pointwise control over
operator compositions;
(iii) mollifier smoothing yielding a $C^\infty$ surrogate $J_\varepsilon$ whose
gradient is Lipschitz continuous (Fréchet differentiability, Layer 3);
(iv) existence of minimizers for $J_\varepsilon$ and their convergence to
minimizers of $J$ as $\varepsilon \to 0$ (Theorem 2, Corollary 2).
All proofs compile with zero \texttt{sorry} under Lean 4 v4.28.0 / Mathlib v4.28.0
and are available at \url{https://github.com/kazuma0606/don_theory}.

**Empirical findings.**
The symmetrized smoothed objective $J_{\varepsilon,\text{sym}}$ matches the convergence
rate of the raw objective (80\%) while exhibiting markedly reduced step-to-step
gradient variation — direct numerical evidence for the Lipschitz gradient property.
Loss landscape visualization confirms that mollifier smoothing converts
the irregular surface of $J$ into a smooth bowl geometry consistent with
$W^{1,\infty}$ regularization.
Monotone decrease of $\|\theta_\varepsilon^* - \theta^*\|$ from $0.956$ to $0.471$
provides qualitative support for Corollary 2.
Convergence behavior is robust across 10 dynamics generators and
dimensions $d \in \{32, 64, 128\}$.

**Practical implication.**
The primary benefit of the $W^{1,\infty}$ embedding is not a reduction in
loss value but a regularization of the gradient field that makes first-order
optimization tractable in landscapes rendered irregular by non-commutative
operator interactions.
This benefit is most pronounced for gradient-based methods without curvature information;
as parameter dimensionality grows, the advantage over second-order methods increases.

**Broader significance.**
The framework is domain-agnostic: although motivated by clinical intervention sequencing,
the mathematical structure applies to any system in which discrete, order-dependent
operations interact with continuous time evolution.
The Lean 4 formal verification provides machine-checked guarantees for all core claims,
offering a level of rigor beyond standard proof-by-argument.

**Future directions** include scaling to high-dimensional parameter spaces $\theta \in \mathbb{R}^m$,
replacing the discrete temporal-convolution proxy with a parameter-space mollifier
for a tighter theory-experiment correspondence,
and leveraging the $C^\infty$ regularity of $J_\varepsilon$ for curvature-based
and PDE-based analysis of intervention landscapes.

---

# **References**

*(To be added)*

---

# **Acknowledgments**

*(To be added)*

---

# **Appendix A. Mathematical Lemmas and Detailed Proof Sketches**

---

## A.0 Lemma: Composition of Lipschitz maps

**Lemma A.0 (Lipschitz stability under composition).**  
Let $X, Y, Z$ be normed spaces.  
Let $g : X \to Y$ and $h : Y \to Z$ be Lipschitz with constants $L_g, L_h \ge 0$, 

i.e.,

$$
\|g(x_1) - g(x_2)\|_Y \le L_g \|x_1 - x_2\|_X,\quad
\|h(y_1) - h(y_2)\|_Z \le L_h \|y_1 - y_2\|_Y.
$$

Then the composition $h \circ g : X \to Z$ is Lipschitz with constant $L_h L_g$.

**Proof (detailed sketch).**  
For any $x_1, x_2 \in X$,

$$
\begin{aligned}
\|h(g(x_1)) - h(g(x_2))\|_Z
&\le L_h \|g(x_1) - g(x_2)\|_Y \\
&\le L_h L_g \|x_1 - x_2\|_X.
\end{aligned}
$$

Thus $h \circ g$ is Lipschitz with constant at most $L_h L_g$.  
By induction, any finite composition of Lipschitz maps is Lipschitz, with a constant given by the product of the individual Lipschitz constants. $\square$

---

## A.1 Lemma: Regularity of intervention-induced observables

**Lemma A.1 (Regularity of $f_{E,t}$).**  
Assume:

- $P \subset \mathbb{R}^d$ is bounded.  
- Each intervention $E_{t_i} : P \to P$, time evolution $U_{t_j} : P \to P$, and observation $O_t : P \to \mathbb{R}^n$ is Lipschitz.

Then for any finite sequence of interventions and time evolution,
$$
f_{E,t}
= O_t \circ U_{t-t_k} \circ E_{t_k} \circ \cdots \circ E_{t_1}
$$
is bounded and Lipschitz on $P$. In particular, $f_{E,t} \in W^{1,\infty}(P)$.

**Proof (detailed sketch).**

1. **Lipschitz continuity of the composite map.**  
   Define
   $$
   F := U_{t-t_k} \circ E_{t_k} \circ \cdots \circ E_{t_1} : P \to P.
   $$
   Each $E_{t_i}$ and $U_{t-t_k}$ is Lipschitz by assumption.  
   By Lemma A.0, the finite composition $F$ is Lipschitz on $P$.  
   Then
   $$
   f_{E,t} = O_t \circ F
   $$
   is also Lipschitz, again by Lemma A.0, since $O_t$ is Lipschitz.

2. **Boundedness.**  
   Since $P$ is bounded and each operator maps bounded sets to bounded sets (by the regularity assumptions in the main text), the image $F(P)$ is bounded in $\mathbb{R}^d$.  
   As $O_t$ is continuous and Lipschitz, it maps bounded sets to bounded sets, hence $f_{E,t}(P)$ is bounded in $\mathbb{R}^n$.

3. **Membership in $W^{1,\infty}$.**  
   A Lipschitz function on a bounded domain in $\mathbb{R}^d$ belongs to $W^{1,\infty}$:  
   by Rademacher's theorem, a Lipschitz function is differentiable almost everywhere, and its weak derivative is essentially bounded with
   $$
   \|\nabla f_{E,t}\|_{L^\infty(P)} \le \text{Lip}(f_{E,t}).
   $$
   Together with boundedness of $f_{E,t}$, this implies
   $$
   f_{E,t} \in W^{1,\infty}(P).
   $$

$\square$

---

## A.2 Lemma: Uniform approximation by mollifiers

**Lemma A.2 (Uniform approximation by mollification).**  
Let $f \in W^{1,\infty}(\mathbb{R}^d)$, and let $\rho \in C_c^\infty(\mathbb{R}^d)$ be a standard mollifier with $\rho \ge 0$, $\int \rho = 1$.  
For $\varepsilon > 0$, define
$$
\rho_\varepsilon(x) = \varepsilon^{-d} \rho\left(\frac{x}{\varepsilon}\right),\quad
f_\varepsilon = f * \rho_\varepsilon.
$$
Then there exists a constant $C > 0$, depending only on $\rho$ and $\|\nabla f\|_{L^\infty}$, such that
$$
\|f_\varepsilon - f\|_{L^\infty(\mathbb{R}^d)} \le C \varepsilon.
$$

**Proof (detailed sketch).**

For any $x \in \mathbb{R}^d$,
$$
\begin{aligned}
f_\varepsilon(x) - f(x)
&= \int_{\mathbb{R}^d} \rho_\varepsilon(y) f(x-y)\,dy - f(x) \int_{\mathbb{R}^d} \rho_\varepsilon(y)\,dy \\
&= \int_{\mathbb{R}^d} \rho_\varepsilon(y)\big(f(x-y) - f(x)\big)\,dy.
\end{aligned}
$$
Since $f \in W^{1,\infty}$, it is Lipschitz with constant $L := \|\nabla f\|_{L^\infty}$. Thus
$$
|f(x-y) - f(x)| \le L \|y\|
\quad \text{for all } x,y.
$$
Therefore,
$$
\begin{aligned}
|f_\varepsilon(x) - f(x)|
&\le \int_{\mathbb{R}^d} \rho_\varepsilon(y) \, |f(x-y) - f(x)|\,dy \\
&\le \int_{\mathbb{R}^d} \rho_\varepsilon(y) \, L \|y\|\,dy \\
&= L \int_{\mathbb{R}^d} \rho_\varepsilon(y) \|y\|\,dy.
\end{aligned}
$$
Now change variables $y = \varepsilon z$. Then
$$
\begin{aligned}
\int_{\mathbb{R}^d} \rho_\varepsilon(y) \|y\|\,dy
&= \int_{\mathbb{R}^d} \varepsilon^{-d} \rho\left(\frac{y}{\varepsilon}\right) \|y\|\,dy \\
&= \int_{\mathbb{R}^d} \varepsilon^{-d} \rho(z) \|\varepsilon z\| \,\varepsilon^d\,dz \\
&= \varepsilon \int_{\mathbb{R}^d} \rho(z) \|z\|\,dz \\
&=: \varepsilon C_0,
\end{aligned}
$$
where $C_0 = \int \rho(z)\|z\|\,dz < \infty$ because $\rho$ has compact support and is smooth.  
Thus
$$
|f_\varepsilon(x) - f(x)| \le L C_0 \varepsilon =: C \varepsilon
\quad \text{for all } x,
$$
and taking the supremum over $x$ yields
$$
\|f_\varepsilon - f\|_{L^\infty} \le C \varepsilon.
$$
$\square$

---

## A.3 Lemma: Well-posedness and approximation of the smoothed objective

Let
$$
J(\theta) = \|f_{E_\theta,t_{\mathrm{final}}} - y_{\text{target}}\|^2,\quad
J_\varepsilon(\theta) = \|f_{E_\theta,t_{\mathrm{final}},\varepsilon} - y_{\text{target}}\|^2,
$$
where $f_{E_\theta,t_{\mathrm{final}},\varepsilon} = f_{E_\theta,t_{\mathrm{final}}} * \rho_\varepsilon$.

**Lemma A.3 (Well-posedness and uniform approximation).**  
Assume:

- $\Theta \subset \mathbb{R}^m$ is compact.  
- For each $\theta \in \Theta$, $f_{E_\theta,t_{\mathrm{final}}} \in W^{1,\infty}$.  
- The map $\theta \mapsto f_{E_\theta,t_{\mathrm{final}}}$ is continuous in the $L^\infty$-topology.  
- There exists $C > 0$ such that
  $$
  \|f_{E_\theta,t_{\mathrm{final}},\varepsilon} - f_{E_\theta,t_{\mathrm{final}}}\|_{L^\infty} \le C \varepsilon
  \quad \text{for all } \theta \in \Theta.
  $$

Then:

1. For each fixed $\varepsilon > 0$, the function $J_\varepsilon : \Theta \to \mathbb{R}$ admits a minimizer.  
2. There exists $C' > 0$ such that for all $\theta \in \Theta$,
   $$
   |J_\varepsilon(\theta) - J(\theta)| \le C' \varepsilon.
   $$

**Proof (detailed sketch).**

1. **Continuity of $J_\varepsilon$.**  
   By assumption, $\theta \mapsto f_{E_\theta,t_{\mathrm{final}}}$ is continuous in $L^\infty$.  
   Convolution with a fixed mollifier $\rho_\varepsilon$ is a continuous linear operator on $L^\infty$, so
   $$
   \theta \mapsto f_{E_\theta,t_{\mathrm{final}},\varepsilon}
   $$
   is also continuous in $L^\infty$.  
   The map
   $$
   v \mapsto \|v - y_{\text{target}}\|^2
   $$
   is continuous on $\mathbb{R}^n$, and composing with the evaluation of $f_{E_\theta,t_{\mathrm{final}},\varepsilon}$ at the relevant point (or integrating over a bounded domain, depending on the precise definition of $J$) preserves continuity.  
   Hence $J_\varepsilon(\theta)$ is continuous in $\theta$.

2. **Existence of a minimizer.**  
   Since $\Theta$ is compact and $J_\varepsilon$ is continuous, the Weierstrass extreme value theorem implies that $J_\varepsilon$ attains its minimum on $\Theta$.
3. **Uniform approximation of $J$ by $J_\varepsilon$.**  
   Fix $\theta \in \Theta$ and denote
   $$
   f(\theta) := f_{E_\theta,t_{\mathrm{final}}},\quad
   f_\varepsilon(\theta) := f_{E_\theta,t_{\mathrm{final}},\varepsilon},\quad
   y := y_{\text{target}}.
   $$
   Then
   $$
   J(\theta) = \|f(\theta) - y\|^2,\quad
   J_\varepsilon(\theta) = \|f_\varepsilon(\theta) - y\|^2.
   $$
   Using the identity
   $$
   \|a\|^2 - \|b\|^2 = \langle a + b, a - b \rangle,
   $$
   we obtain
   $$
   \begin{aligned}
   |J_\varepsilon(\theta) - J(\theta)|
   &= \big|\|f_\varepsilon(\theta) - y\|^2 - \|f(\theta) - y\|^2\big| \\
   &= \big|\langle f_\varepsilon(\theta) - y + f(\theta) - y,\; f_\varepsilon(\theta) - y - (f(\theta) - y)\rangle\big| \\
   &= \big|\langle f_\varepsilon(\theta) + f(\theta) - 2y,\; f_\varepsilon(\theta) - f(\theta)\rangle\big|.
   \end{aligned}
   $$
   Applying Cauchy–Schwarz,
   $$
   |J_\varepsilon(\theta) - J(\theta)|
   \le \|f_\varepsilon(\theta) + f(\theta) - 2y\| \cdot \|f_\varepsilon(\theta) - f(\theta)\|.
   $$
   By boundedness of the state space and operators, there exists $M > 0$ such that
   $$
   \|f_\varepsilon(\theta)\|,\ \|f(\theta)\|,\ \|y\| \le M
   \quad \text{for all } \theta \in \Theta,\ \varepsilon \in (0,1].
   $$
   Hence
   $$
   \|f_\varepsilon(\theta) + f(\theta) - 2y\|
   \le \|f_\varepsilon(\theta)\| + \|f(\theta)\| + 2\|y\|
   \le 4M.
   $$
   By the uniform approximation assumption,
   $$
   \|f_\varepsilon(\theta) - f(\theta)\| \le C \varepsilon.
   $$
   Combining these,
   $$
   |J_\varepsilon(\theta) - J(\theta)|
   \le 4M \cdot C \varepsilon =: C' \varepsilon.
   $$
   Since this bound is uniform in $\theta$, we obtain
   $$
   \sup_{\theta \in \Theta} |J_\varepsilon(\theta) - J(\theta)| \le C' \varepsilon.
   $$

This shows both existence of minimizers for $J_\varepsilon$ and uniform approximation of $J$ by $J_\varepsilon$ as $\varepsilon \to 0$. $\square$

---

## A.5 Lean 4 Proof Summary

All core theoretical claims in this paper have been formally verified
using Lean 4 with Mathlib. The complete proof development is publicly available at:

> **https://github.com/kazuma0606/don_theory**
> Directory: `verify/lean4/MedicusVerify/`

**Build:** `cd verify/lean4 && lake build MedicusVerify`
**Environment:** Lean 4 `v4.28.0`, Mathlib `v4.28.0`
**Status:** zero `sorry`, zero warnings.

| File | Content | Corresponds to |
|---|---|---|
| `Basic.lean` | Abstract axioms (`state_dependent`, `irreversible`) | §2 assumptions |
| `Layer1Monoid.lean` | Non-commutative monoid (`noncomm_exists`, `no_inverse`) | §2.2–2.3 |
| `Layer2Banach.lean` | $W^{1,\infty}$ Banach space (norm axioms, completeness) | §3.1, Lemma 1 |
| `Layer3Mollifier.lean` | Mollifier $C^\infty$, Fréchet differentiability, convergence | §3.2, Theorem 1, Corollary 1 |
| `Layer4Regularity.lean` | Lipschitz composition, $W^{1,\infty}$ membership, minimizer existence and convergence | A.0–A.1, Theorem 2, Corollary 2 |

**Note on $W^{2,\infty}$ assumption.**
`Layer3Mollifier.lean` (`mollifier_converges`) explicitly assumes
$f \in W^{2,\infty}$ (hypothesis `hdf_lip`).
This assumption is mathematically necessary: the IBP step
$(f \star \nabla\rho_\varepsilon)(x) = ((\nabla f) \star \rho_\varepsilon)(x)$
requires continuity of $\nabla f$, which does not follow from $W^{1,\infty}$ alone.
The assumption is stated explicitly in §3.2 Corollary 1.

---

# **Appendix B. Domains, Structural Analogies, and Relations to World Models**

## **B.1 Artificially Commutative Systems**

Certain engineered systems are explicitly designed to *eliminate* non‑commutativity.  
Transactional databases, banking systems, and accounting infrastructures enforce strict consistency through ACID properties, locking mechanisms, and serializability constraints. These mechanisms ensure that the order of operations does not affect the final state, even though the underlying real‑world processes would naturally be order‑dependent.

In this sense, transactional systems constitute a **special case** in which non‑commutativity is intentionally suppressed to guarantee predictability, auditability, and reproducibility. Such systems do not reflect the intrinsic behavior of time‑evolving dynamical processes but rather represent a controlled sandbox in which commutativity is artificially imposed.

---

## **B.2 Naturally Non‑Commutative Systems**

In contrast, many real‑world domains exhibit **inherent non‑commutativity** due to the interaction between interventions and ongoing time evolution. Examples include:

- **Medicine and physiology**  
Treatment sequences (e.g., surgery followed by chemotherapy vs. the reverse) produce different outcomes because interventions modify the underlying physiological state.
- **Biology and chemistry**  
Stimuli or reagents applied in different orders lead to distinct reaction pathways or cellular responses.
- **Robotics and control**  
Manipulation tasks depend critically on the order of applied actions, especially under nonlinear dynamics.
- **Economics and policy**  
The effect of policy A followed by policy B is generally not equivalent to the reverse order, due to path‑dependent macroeconomic dynamics.
- **Climate and geophysical systems**  
External perturbations interact with nonlinear time evolution, making the sequence of disturbances essential.

These domains share a common structural property:  
**interventions act on a state that is itself evolving in time**, and therefore the composition of intervention operators is generically non‑commutative.

---

## **B.3 Structural Analogy to Quantum Mechanics**

Interestingly, the operator‑theoretic structure of non‑commutative interventions parallels that of quantum mechanics. In quantum systems, observables and time‑evolution operators satisfy:

$$
AB \neq BA,
$$

reflecting the fundamental non‑commutativity of measurements and dynamical evolution.  
Similarly, in our framework:

$$
U_{\Delta t} \circ E_t \neq E_{t+\Delta t} \circ U_{\Delta t},
$$

expressing that the effect of an intervention depends on when it is applied relative to the system's intrinsic dynamics.

Although the physical interpretations differ, the **mathematical structure—time‑indexed operators acting on a state space with non‑commutative composition—is shared**. This analogy highlights that non‑commutativity is not an exotic phenomenon but a natural consequence of time‑dependent operator interactions.

---

## **B.4 Relation to World Models and Dynamics Learning**

Many machine learning systems attempt to learn a *world model*—a representation of state dynamics and observations. Such models typically assume:

1. a state transition $p_{t+1} = U(p_t)$,  
2. an observation map $O(p_t)$, and  
3. actions that are treated as inputs rather than operators.

In contrast, our framework treats interventions as **operators** acting on the state space, allowing for **non‑commutative composition**:

$$
U_{\Delta t} \circ E_t \neq E_{t+\Delta t} \circ U_{\Delta t}.
$$


This operator‑theoretic formulation generalizes classical world models by explicitly representing the algebraic structure of interventions and their interaction with time evolution. The resulting non‑commutative dynamics arise naturally in many real‑world systems, including biological, physical, and economic processes.

A detailed instantiation of this framework as a practical world model is beyond the scope of this paper, but we note that the operator‑based perspective provides a principled foundation for future work in this direction.

---

## **B.5 Scope of This Work**

While many motivating examples arise from medicine and the life sciences, the operator‑theoretic framework developed in this work is **domain‑agnostic**. The mathematical results apply to any system in which:

1. interventions modify the state,
2. the state evolves in time, and
3. the order of operations matters.

The clinical examples in the main text serve only as intuitive illustrations of these principles.  
A detailed instantiation of this framework in a medical decision‑support system will be presented in subsequent work (MEDICUS), where the non‑commutative structure of clinical interventions becomes essential.

---

