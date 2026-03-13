### Appendix A. Mathematical lemmas and detailed proof sketches

---

## A.0 Lemma: Composition of Lipschitz maps

**Lemma A.0 (Lipschitz stability under composition).**  
Let $(X, Y, Z$) be normed spaces.  
Let $(g : X \to Y$) and $(h : Y \to Z$) be Lipschitz with constants $(L_g, L_h \ge 0$), 

i.e.,

$
\|g(x_1) - g(x_2)\|_Y \le L_g \|x_1 - x_2\|_X,\quad
\|h(y_1) - h(y_2)\|_Z \le L_h \|y_1 - y_2\|_Y.
$

Then the composition $(h \circ g : X \to Z$) is Lipschitz with constant $(L_h L_g$).

**Proof (detailed sketch).**  
For any $(x_1, x_2 \in X$),

$
\begin{aligned}
\|h(g(x_1)) - h(g(x_2))\|_Z
&\le L_h \|g(x_1) - g(x_2)\|_Y \\
&\le L_h L_g \|x_1 - x_2\|_X.
\end{aligned}
$

Thus $(h \circ g$) is Lipschitz with constant at most $(L_h L_g$).  
By induction, any finite composition of Lipschitz maps is Lipschitz, with a constant given by the product of the individual Lipschitz constants. $(\square$)

---

## A.1 Lemma: Regularity of intervention-induced observables

**Lemma A.1 (Regularity of $(f_{E,t}$)).**  
Assume:

- $(P \subset \mathbb{R}^d$) is bounded.  
- Each intervention $(E_{t_i} : P \to P$), time evolution $(U_{t_j} : P \to P$), and observation $(O_t : P \to \mathbb{R}^n$) is Lipschitz.  

Then for any finite sequence of interventions and time evolution,
$
f_{E,t}
= O_t \circ U_{t-t_k} \circ E_{t_k} \circ \cdots \circ E_{t_1}
$
is bounded and Lipschitz on $(P$). In particular, $(f_{E,t} \in W^{1,\infty}(P)$).

**Proof (detailed sketch).**

1. **Lipschitz continuity of the composite map.**  
   Define
   $
   F := U_{t-t_k} \circ E_{t_k} \circ \cdots \circ E_{t_1} : P \to P.
   $
   Each $(E_{t_i}$) and $(U_{t-t_k}$) is Lipschitz by assumption.  
   By Lemma A.0, the finite composition $(F$) is Lipschitz on $(P$).  
   Then
   $
   f_{E,t} = O_t \circ F
   $
   is also Lipschitz, again by Lemma A.0, since $(O_t$) is Lipschitz.

2. **Boundedness.**  
   Since $(P$) is bounded and each operator maps bounded sets to bounded sets (by the regularity assumptions in the main text), the image $(F(P)$) is bounded in $(\mathbb{R}^d$).  
   As $(O_t$) is continuous and Lipschitz, it maps bounded sets to bounded sets, hence $(f_{E,t}(P)$) is bounded in $(\mathbb{R}^n$).

3. **Membership in $(W^{1,\infty}$).**  
   A Lipschitz function on a bounded domain in $(\mathbb{R}^d$) belongs to $(W^{1,\infty}$):  
   by Rademacher’s theorem, a Lipschitz function is differentiable almost everywhere, and its weak derivative is essentially bounded with
   $
   \|\nabla f_{E,t}\|_{L^\infty(P)} \le \text{Lip}(f_{E,t}).
   $
   Together with boundedness of $(f_{E,t}$), this implies
   $
   f_{E,t} \in W^{1,\infty}(P).
   $
$(\square$)

---

## A.2 Lemma: Uniform approximation by mollifiers

**Lemma A.2 (Uniform approximation by mollification).**  
Let $(f \in W^{1,\infty}(\mathbb{R}^d)$), and let $(\rho \in C_c^\infty(\mathbb{R}^d)$) be a standard mollifier with $(\rho \ge 0$), $(\int \rho = 1$).  
For $(\varepsilon > 0$), define
$
\rho_\varepsilon(x) = \varepsilon^{-d} \rho\left(\frac{x}{\varepsilon}\right),\quad
f_\varepsilon = f * \rho_\varepsilon.
$
Then there exists a constant $(C > 0$), depending only on $(\rho$) and $(\|\nabla f\|_{L^\infty}$), such that
$
\|f_\varepsilon - f\|_{L^\infty(\mathbb{R}^d)} \le C \varepsilon.
$

**Proof (detailed sketch).**

For any $(x \in \mathbb{R}^d$),
$
\begin{aligned}
f_\varepsilon(x) - f(x)
&= \int_{\mathbb{R}^d} \rho_\varepsilon(y) f(x-y)\,dy - f(x) \int_{\mathbb{R}^d} \rho_\varepsilon(y)\,dy \\
&= \int_{\mathbb{R}^d} \rho_\varepsilon(y)\big(f(x-y) - f(x)\big)\,dy.
\end{aligned}
$
Since $(f \in W^{1,\infty}$), it is Lipschitz with constant $(L := \|\nabla f\|_{L^\infty}$). Thus
$
|f(x-y) - f(x)| \le L \|y\|
\quad \text{for all } x,y.
$
Therefore,
$
\begin{aligned}
|f_\varepsilon(x) - f(x)|
&\le \int_{\mathbb{R}^d} \rho_\varepsilon(y) \, |f(x-y) - f(x)|\,dy \\
&\le \int_{\mathbb{R}^d} \rho_\varepsilon(y) \, L \|y\|\,dy \\
&= L \int_{\mathbb{R}^d} \rho_\varepsilon(y) \|y\|\,dy.
\end{aligned}
$
Now change variables $(y = \varepsilon z$). Then
$
\begin{aligned}
\int_{\mathbb{R}^d} \rho_\varepsilon(y) \|y\|\,dy
&= \int_{\mathbb{R}^d} \varepsilon^{-d} \rho\left(\frac{y}{\varepsilon}\right) \|y\|\,dy \\
&= \int_{\mathbb{R}^d} \varepsilon^{-d} \rho(z) \|\varepsilon z\| \,\varepsilon^d\,dz \\
&= \varepsilon \int_{\mathbb{R}^d} \rho(z) \|z\|\,dz \\
&=: \varepsilon C_0,
\end{aligned}
$
where $(C_0 = \int \rho(z)\|z\|\,dz < \infty$) because $(\rho$) has compact support and is smooth.  
Thus
$
|f_\varepsilon(x) - f(x)| \le L C_0 \varepsilon =: C \varepsilon
\quad \text{for all } x,
$
and taking the supremum over $(x$) yields
$
\|f_\varepsilon - f\|_{L^\infty} \le C \varepsilon.
$
$(\square$)

---

## A.3 Lemma: Well-posedness and approximation of the smoothed objective

Let
$
J(\theta) = \|f_{E_\theta,t_{\mathrm{final}}} - y_{\text{target}}\|^2,\quad
J_\varepsilon(\theta) = \|f_{E_\theta,t_{\mathrm{final}},\varepsilon} - y_{\text{target}}\|^2,
$
where $(f_{E_\theta,t_{\mathrm{final}},\varepsilon} = f_{E_\theta,t_{\mathrm{final}}} * \rho_\varepsilon$).

**Lemma A.3 (Well-posedness and uniform approximation).**  
Assume:

- $(\Theta \subset \mathbb{R}^m$) is compact.  
- For each $(\theta \in \Theta$), $(f_{E_\theta,t_{\mathrm{final}}} \in W^{1,\infty}$).  
- The map $(\theta \mapsto f_{E_\theta,t_{\mathrm{final}}}$) is continuous in the $(L^\infty$)-topology.  
- There exists $(C > 0$) such that
  $
  \|f_{E_\theta,t_{\mathrm{final}},\varepsilon} - f_{E_\theta,t_{\mathrm{final}}}\|_{L^\infty} \le C \varepsilon
  \quad \text{for all } \theta \in \Theta.
  $

Then:

1. For each fixed $(\varepsilon > 0$), the function $(J_\varepsilon : \Theta \to \mathbb{R}$) admits a minimizer.  
2. There exists $(C' > 0$) such that for all $(\theta \in \Theta$),
   $
   |J_\varepsilon(\theta) - J(\theta)| \le C' \varepsilon.
   $

**Proof (detailed sketch).**

1. **Continuity of $(J_\varepsilon$).**  
   By assumption, $(\theta \mapsto f_{E_\theta,t_{\mathrm{final}}}$) is continuous in $(L^\infty$).  
   Convolution with a fixed mollifier $(\rho_\varepsilon$) is a continuous linear operator on $(L^\infty$), so
   $
   \theta \mapsto f_{E_\theta,t_{\mathrm{final}},\varepsilon}
   $
   is also continuous in $(L^\infty$).  
   The map
   $
   v \mapsto \|v - y_{\text{target}}\|^2
   $
   is continuous on $(\mathbb{R}^n$), and composing with the evaluation of $(f_{E_\theta,t_{\mathrm{final}},\varepsilon}$) at the relevant point (or integrating over a bounded domain, depending on the precise definition of $(J$)) preserves continuity.  
   Hence $(J_\varepsilon(\theta)$) is continuous in $(\theta$).

2. **Existence of a minimizer.**  
   Since $(\Theta$) is compact and $(J_\varepsilon$) is continuous, the Weierstrass extreme value theorem implies that $(J_\varepsilon$) attains its minimum on $(\Theta$).

3. **Uniform approximation of $(J$) by $(J_\varepsilon$).**  
   Fix $(\theta \in \Theta$) and denote
   $
   f(\theta) := f_{E_\theta,t_{\mathrm{final}}},\quad
   f_\varepsilon(\theta) := f_{E_\theta,t_{\mathrm{final}},\varepsilon},\quad
   y := y_{\text{target}}.
   $
   Then
   $
   J(\theta) = \|f(\theta) - y\|^2,\quad
   J_\varepsilon(\theta) = \|f_\varepsilon(\theta) - y\|^2.
   $
   Using the identity
   $
   \|a\|^2 - \|b\|^2 = \langle a + b, a - b \rangle,
   $
   we obtain
   $
   \begin{aligned}
   |J_\varepsilon(\theta) - J(\theta)|
   &= \big|\|f_\varepsilon(\theta) - y\|^2 - \|f(\theta) - y\|^2\big| \\
   &= \big|\langle f_\varepsilon(\theta) - y + f(\theta) - y,\; f_\varepsilon(\theta) - y - (f(\theta) - y)\rangle\big| \\
   &= \big|\langle f_\varepsilon(\theta) + f(\theta) - 2y,\; f_\varepsilon(\theta) - f(\theta)\rangle\big|.
   \end{aligned}
   $
   Applying Cauchy–Schwarz,
   $
   |J_\varepsilon(\theta) - J(\theta)|
   \le \|f_\varepsilon(\theta) + f(\theta) - 2y\| \cdot \|f_\varepsilon(\theta) - f(\theta)\|.
   $
   By boundedness of the state space and operators, there exists $(M > 0$) such that
   $
   \|f_\varepsilon(\theta)\|,\ \|f(\theta)\|,\ \|y\| \le M
   \quad \text{for all } \theta \in \Theta,\ \varepsilon \in (0,1].
   $
   Hence
   $
   \|f_\varepsilon(\theta) + f(\theta) - 2y\|
   \le \|f_\varepsilon(\theta)\| + \|f(\theta)\| + 2\|y\|
   \le 4M.
   $
   By the uniform approximation assumption,
   $
   \|f_\varepsilon(\theta) - f(\theta)\| \le C \varepsilon.
   $
   Combining these,
   $
   |J_\varepsilon(\theta) - J(\theta)|
   \le 4M \cdot C \varepsilon =: C' \varepsilon.
   $
   Since this bound is uniform in $(\theta$), we obtain
   $
   \sup_{\theta \in \Theta} |J_\varepsilon(\theta) - J(\theta)| \le C' \varepsilon.
   $

This shows both existence of minimizers for $(J_\varepsilon$) and uniform approximation of $(J$) by $(J_\varepsilon$) as $(\varepsilon \to 0$). $(\square$)