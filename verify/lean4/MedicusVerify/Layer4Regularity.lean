/-
  MedicusVerify.Layer4Regularity
  ==============================
  Layer 4: Regularity of intervention observables and optimization results.
  Corresponds to Appendix A.0–A.3 of manuscript_v_20260313.md.

  Theorems:
  - lipschitz_comp_of_lipschitz  (A2.1) Lemma A.0: Lipschitz composition stability
  - observable_in_W1inf          (A2.2) Lemma A.1: f_{E,t} ∈ W^{1,∞} from Lipschitz + differentiable
  - J_eps_minimizer_exists       (A2.3) Theorem 2: minimizer of J_ε exists on compact Θ
  - minimizers_approximate       (A2.4) Corollary 2: minimizers of J_ε approximate minimizers of J
-/

import Mathlib.Topology.EMetricSpace.Lipschitz
import Mathlib.Topology.Order.Compact
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.MeanValue
import MedicusVerify.Layer2Banach

open scoped NNReal
open Filter Topology Set

noncomputable section

-- ============================================================
-- A2.1: Lemma A.0 — Lipschitz composition stability
-- ============================================================

/-- Lemma A.0: h ∘ g is Lipschitz with constant Kh * Kg.
    Direct corollary of `LipschitzWith.comp`. -/
theorem lipschitz_comp_of_lipschitz
    {α β γ : Type*}
    [PseudoEMetricSpace α] [PseudoEMetricSpace β] [PseudoEMetricSpace γ]
    {Kg Kh : ℝ≥0} {g : α → β} {h : β → γ}
    (hg : LipschitzWith Kg g) (hh : LipschitzWith Kh h) :
    LipschitzWith (Kh * Kg) (h ∘ g) :=
  hh.comp hg

/-- Inductive extension of Lemma A.0:
    prepending one more Lipschitz map preserves the product Lipschitz bound. -/
theorem lipschitz_comp_cons
    {α β γ : Type*}
    [PseudoEMetricSpace α] [PseudoEMetricSpace β] [PseudoEMetricSpace γ]
    {K_rest K_head : ℝ≥0} {rest : α → β} {head : β → γ}
    (h_rest : LipschitzWith K_rest rest)
    (h_head : LipschitzWith K_head head) :
    LipschitzWith (K_head * K_rest) (head ∘ rest) :=
  lipschitz_comp_of_lipschitz h_rest h_head

-- ============================================================
-- A2.2: Lemma A.1 — f_{E,t} ∈ W^{1,∞}
-- ============================================================

namespace MedicusMin

/-- Auxiliary: A Lipschitz function has bounded derivative.
    ‖deriv f x‖ ≤ K for all x.
    Proof: converse mean value inequality (`norm_deriv_le_of_lipschitz`). -/
lemma deriv_le_of_lipschitzWith {f : ℝ → ℝ} {K : ℝ≥0}
    (hf_lip : LipschitzWith K f) (x : ℝ) :
    |deriv f x| ≤ (K : ℝ) := by
  have h := norm_deriv_le_of_lipschitz hf_lip (x₀ := x)
  rwa [Real.norm_eq_abs] at h

/-- Lemma A.1: If f : ℝ → ℝ is differentiable, Lipschitz with constant K,
    and bounded in absolute value by B, then f ∈ MedicusMin (W^{1,∞}).

    This formalises: composition of Lipschitz operators (via Lemma A.0)
    + bounded domain → the resulting observable is in W^{1,∞}.

    Proof:
    (1) BddAbove |f|:      from hypothesis hB.
    (2) BddAbove |deriv f|: from deriv_le_of_lipschitzWith — bounded by K.
    (3) Differentiability:  from hypothesis hf_diff. -/
theorem observable_in_W1inf
    {f : ℝ → ℝ} {K : ℝ≥0} {B : ℝ}
    (hf_diff : Differentiable ℝ f)
    (hf_lip  : LipschitzWith K f)
    (hB      : ∀ x, |f x| ≤ B) :
    ∃ (m : MedicusMin), m.val = f :=
  ⟨⟨f, hf_diff,
    ⟨B, fun _ ⟨x, hx⟩ => hx ▸ hB x⟩,
    ⟨K, fun _ ⟨x, hx⟩ => hx ▸ by
      have := deriv_le_of_lipschitzWith hf_lip x
      exact_mod_cast this⟩⟩, rfl⟩

end MedicusMin

-- ============================================================
-- A2.3: Theorem 2 — Existence of minimizer of J_ε
-- ============================================================

/-- Theorem 2 (Existence of Minimizer): If S ⊆ Θ is compact and nonempty,
    and J_ε : Θ → ℝ is continuous on S, then J_ε has a minimizer in S.

    Proof: Weierstrass extreme value theorem (`IsCompact.exists_isMinOn`). -/
theorem J_eps_minimizer_exists
    {Θ : Type*} [TopologicalSpace Θ]
    {S : Set Θ} (hS : IsCompact S) (hS_ne : S.Nonempty)
    {J_eps : Θ → ℝ} (hJ : ContinuousOn J_eps S) :
    ∃ θ_min ∈ S, IsMinOn J_eps S θ_min :=
  hS.exists_isMinOn hS_ne hJ

-- ============================================================
-- A2.4: Corollary 2 — Minimizers of J_ε approximate minimizers of J
-- ============================================================

/-- Corollary 2 (Minimizer Approximation): If J_ε approximates J uniformly with
    sup-error ≤ C' * ε, then any minimizer θ_ε of J_ε satisfies
      J(θ_ε) ≤ J(θ*) + 2 * C' * ε
    for every minimizer θ* of J.

    Proof:
      J(θ_ε) ≤ J_ε(θ_ε) + C'ε   [uniform bound, upper half]
             ≤ J_ε(θ*) + C'ε    [θ_ε minimises J_ε on S]
             ≤ J(θ*) + 2C'ε     [uniform bound, upper half] -/
theorem minimizers_approximate
    {Θ : Type*} {S : Set Θ}
    {J J_eps : Θ → ℝ} {C' ε : ℝ}
    (hunif    : ∀ θ ∈ S, |J_eps θ - J θ| ≤ C' * ε)
    {θ_eps   : Θ} (hθ_mem  : θ_eps ∈ S)  (hθ_min  : IsMinOn J_eps S θ_eps)
    {θ_star  : Θ} (hθ_star : θ_star ∈ S) :
    J θ_eps ≤ J θ_star + 2 * C' * ε := by
  -- h1: J(θ_ε) ≤ J_ε(θ_ε) + C'ε
  -- From |J_ε - J| ≤ C'ε: -(C'ε) ≤ J_ε(θ_ε) - J(θ_ε), i.e. J ≤ J_ε + C'ε
  have h1 : J θ_eps ≤ J_eps θ_eps + C' * ε := by
    have := (abs_le.mp (hunif θ_eps hθ_mem)).1; linarith
  -- h2: J_ε(θ_ε) ≤ J_ε(θ*) because θ_ε minimises J_ε
  have h2 : J_eps θ_eps ≤ J_eps θ_star := hθ_min hθ_star
  -- h3: J_ε(θ*) ≤ J(θ*) + C'ε
  have h3 : J_eps θ_star ≤ J θ_star + C' * ε := by
    have := (abs_le.mp (hunif θ_star hθ_star)).2; linarith
  linarith

/-- Quantitative bound on the minimum value error:
    |J(θ_ε) - J(θ*)| ≤ 2 * C' * ε,
    where θ_ε minimises J_ε and θ* minimises J.

    Combined with J(θ*) ≤ J(θ_ε) (since θ* minimises J) this gives
    0 ≤ J(θ_ε) - J(θ*) ≤ 2C'ε, so both the minimum value and the minimiser
    converge as ε → 0 (the latter when the minimiser is unique). -/
theorem minimum_values_converge
    {Θ : Type*} {S : Set Θ}
    {J J_eps : Θ → ℝ} {C' ε : ℝ}
    (hunif      : ∀ θ ∈ S, |J_eps θ - J θ| ≤ C' * ε)
    {θ_eps  : Θ} (hθ_eps_mem  : θ_eps ∈ S)  (hθ_eps_min  : IsMinOn J_eps S θ_eps)
    {θ_star : Θ} (hθ_star_mem : θ_star ∈ S) (hθ_star_min : IsMinOn J S θ_star) :
    |J θ_eps - J θ_star| ≤ 2 * C' * ε := by
  have hub : J θ_eps ≤ J θ_star + 2 * C' * ε :=
    minimizers_approximate hunif hθ_eps_mem hθ_eps_min hθ_star_mem
  have hlb : J θ_star ≤ J θ_eps := hθ_star_min hθ_eps_mem
  rw [abs_le]
  constructor
  · linarith
  · linarith

end
