/-
  MedicusVerify.Basic
  ===================
  MEDICUS basic type definitions and axioms.

  Policy:
  - PatientState is abstract (not fixed to R^3)
  - noncomm_exists uses abstract existence proof (no dataset)
  Ref: verify/PROOF_SPEC.md sections 0-1
-/

import Mathlib.Topology.Basic
import Mathlib.Algebra.Group.Basic

-- ============================================================
-- Section 0: Basic type definitions
-- ============================================================

/-- Medical intervention: a self-map on the patient state space.
    The patient state space P is abstract (internal structure unspecified). -/
def MedicalIntervention (P : Type*) := P → P

namespace MedicalIntervention

variable {P : Type*}

/-- Composition of interventions (apply b first, then a) -/
def compose (a b : MedicalIntervention P) : MedicalIntervention P :=
  fun p => a (b p)

/-- Identity intervention ("do nothing") -/
def idIntervention : MedicalIntervention P := id

-- ============================================================
-- Section 1: Axioms
-- Mathematical encoding of clinical facts.
-- Each axiom is motivated in the paper's Introduction.
-- ============================================================

/-- Axiom 1: State-dependent intervention exists.
    Clinical basis: chemotherapy suppresses immunity, which alters
    radiosensitivity. Hence the effect of intervention a depends on
    whether b was applied first. -/
axiom state_dependent :
  ∀ (P : Type*) [TopologicalSpace P],
  ∃ (a b : MedicalIntervention P) (p : P),
    a (b p) ≠ b (a p)

/-- Axiom 2: Irreversible intervention exists.
    Clinical basis: surgical resection and radiation cause tissue changes
    that cannot be undone by any subsequent intervention. -/
axiom irreversible :
  ∀ (P : Type*) [TopologicalSpace P],
  ∃ (a : MedicalIntervention P),
    ∀ (b : MedicalIntervention P), ∃ (p : P), b (a p) ≠ p

end MedicalIntervention
