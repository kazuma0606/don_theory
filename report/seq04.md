
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

\[
AB \neq BA,
\]

reflecting the fundamental non‑commutativity of measurements and dynamical evolution.  
Similarly, in our framework:

\[
U_{\Delta t} \circ E_t \neq E_{t+\Delta t} \circ U_{\Delta t},
\]

expressing that the effect of an intervention depends on when it is applied relative to the system’s intrinsic dynamics.

Although the physical interpretations differ, the **mathematical structure—time‑indexed operators acting on a state space with non‑commutative composition—is shared**. This analogy highlights that non‑commutativity is not an exotic phenomenon but a natural consequence of time‑dependent operator interactions.

---

## **B.4 Relation to World Models and Dynamics Learning**

Many machine learning systems attempt to learn a *world model*—a representation of state dynamics and observations. Such models typically assume:

1. a state transition \(p_{t+1} = U(p_t)\),  
2. an observation map \(O(p_t)\), and  
3. actions that are treated as inputs rather than operators.

In contrast, our framework treats interventions as **operators** acting on the state space, allowing for **non‑commutative composition**:

\[
U_{\Delta t} \circ E_t \neq E_{t+\Delta t} \circ U_{\Delta t}.
\]

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
