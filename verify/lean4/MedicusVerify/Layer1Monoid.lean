/-
  MedicusVerify.Layer1Monoid
  ==========================
  Layer 1: Non-commutative monoid proofs.

  Theorems:
  - Monoid instance    (Task 2.1)
  - noncomm_exists     (Task 2.2) *** TOP PRIORITY ***
  - no_inverse         (Task 2.3)

  Ref: verify/PROOF_SPEC.md section 1
-/

import MedicusVerify.Basic

namespace MedicalIntervention

variable {P : Type*} [TopologicalSpace P]

-- ============================================================
-- Task 2.1: Monoid instance
-- Function composition satisfies monoid axioms automatically.
-- ============================================================

instance : Monoid (MedicalIntervention P) where
  mul a b         := compose a b
  one             := idIntervention
  mul_assoc a b c := funext (fun _ => rfl)
  one_mul a       := funext (fun _ => rfl)
  mul_one a       := funext (fun _ => rfl)

-- ============================================================
-- Task 2.2: noncomm_exists *** TOP PRIORITY ***
-- Derived from state_dependent axiom.
-- ============================================================

/-- Non-commutativity theorem:
    There exists a pair of interventions whose outcome depends on order. -/
theorem noncomm_exists :
    ∃ (a b : MedicalIntervention P), a * b ≠ b * a := by
  obtain ⟨a, b, p, h⟩ := state_dependent P
  exact ⟨a, b, fun heq => h (congr_fun heq p)⟩

-- ============================================================
-- Task 2.3: no_inverse
-- Derived from irreversible axiom.
-- ============================================================

/-- No-inverse theorem:
    There exists an intervention with no left inverse (irreversibility). -/
theorem no_inverse :
    ∃ (a : MedicalIntervention P),
      ¬∃ (b : MedicalIntervention P), ∀ p, b (a p) = p := by
  obtain ⟨a, ha⟩ := irreversible P
  exact ⟨a, fun ⟨b, hb⟩ => by
    obtain ⟨p, hp⟩ := ha b
    exact hp (hb p)⟩

end MedicalIntervention
