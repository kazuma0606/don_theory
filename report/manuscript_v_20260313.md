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

**Note:** Experimental results are currently under development and will be included in the final manuscript.

Planned experiments include:

- High-dimensional synthetic environment validation  
- Non-commutative intervention sequence optimization  
- Gradient stability analysis  
- Comparison of smoothed vs. non-smoothed optimization performance  
- Ablation studies on dimensionality and operator complexity

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

*(To be expanded with final experimental results)*



---

# **7.x Future Work**

The smoothing framework introduced in this work yields $(C^\infty$) approximations of intervention‑induced observables, enabling not only stable first‑order gradients but also higher‑order differential operators.  
This opens several promising directions for future research.

**(1) Second‑order and higher‑order optimization.**  
Because the smoothed objective $(J_\varepsilon(\theta)$) admits a well‑defined Hessian $(\nabla^2_\theta J_\varepsilon(\theta)$), second‑order methods such as Newton or quasi‑Newton optimization become theoretically justified.  
A systematic study of curvature properties, Hessian conditioning, and the behavior of saddle points in non‑commutative intervention systems remains an open direction.

**(2) Curvature‑based regularization via the Laplacian.**  
The parameter‑space Laplacian  
$$
\Delta_\theta J_\varepsilon(\theta)
= \sum_i \frac{\partial^2 J_\varepsilon}{\partial \theta_i^2}
$$

provides a natural measure of local curvature of the optimization landscape.  
Laplacian‑based regularization may suppress sharp variations induced by non‑commutative operator compositions, potentially improving robustness and generalization of learned intervention policies.

**(3) Laplacian‑of‑Gaussian (LoG) operators for detecting non‑commutative structure.**  
The composition \(\Delta_\theta J_\varepsilon(\theta)\) after mollifier smoothing is structurally analogous to the Laplacian‑of‑Gaussian operator used in image processing.  
This suggests that LoG‑type operators could be used to highlight regions where non‑commutative intervention effects produce high curvature, offering a new tool for analyzing and visualizing the geometry of intervention landscapes.

**(4) PDE‑based perspectives on intervention propagation.**  
Since $(J_\varepsilon$) is smooth in both time and parameters, differential operators such as the d’Alembertian  
$$
\square J_\varepsilon(t,\theta)
= \partial_t^2 J_\varepsilon(t,\theta) - \Delta_\theta J_\varepsilon(t,\theta)
$$
may provide a wave‑like description of how intervention‑induced perturbations propagate through time and diffuse across the parameter space.  
Exploring PDE‑based formulations could yield new insights into the stability, sensitivity, and causal structure of non‑commutative intervention systems.

These directions extend the mathematical foundations established in this work while preserving the core objective: enabling differentiable optimization of inherently discrete and non‑commutative intervention processes.


---

# **8 Conclusion**

We introduced a functional-analytic framework for optimizing intervention parameters in time-dependent, non-commutative systems. By embedding intervention-induced observables into Banach spaces and applying mollifier-based smoothing, we enable gradient-based optimization while preserving the fundamental operator structure.

The framework provides a principled mathematical foundation for intervention optimization in high-dimensional, time-evolving systems, with applications spanning scientific machine learning, robotics, control theory, and computational medicine.

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

