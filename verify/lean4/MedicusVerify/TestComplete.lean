import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.UniformLimitsDeriv
import Mathlib.Data.Real.Pointwise
import Mathlib.Topology.MetricSpace.Cauchy
import Mathlib.Topology.UniformSpace.UniformConvergence
import MedicusVerify.Layer2Banach

open scoped Pointwise
open MedicusMin Filter

-- Test key facts needed for completeness

-- 1) le_of_tendsto: if aₙ → a and aₙ ≤ b always, then a ≤ b
example (a b : ℝ) (aₙ : ℕ → ℝ) (h : Tendsto aₙ atTop (𝓝 a)) (hle : ∀ n, aₙ n ≤ b) :
    a ≤ b :=
  le_of_tendsto h (Eventually.of_forall hle)

-- 2) tendstoUniformly_iff for ℝ
#check @tendstoUniformly_iff

-- 3) UniformCauchySeqOn → tendstoUniformlyOn
#check @UniformCauchySeqOn.tendstoUniformlyOn_of_tendsto

-- 4) Pointwise limit via completeness
example (f : ℕ → ℝ) (h : CauchySeq f) : ∃ l, Tendsto f atTop (𝓝 l) :=
  cauchySeq_tendsto_of_complete h

-- 5) TendstoUniformly → Tendsto pointwise
example {F : ℕ → ℝ → ℝ} {f : ℝ → ℝ}
    (h : TendstoUniformly F f atTop) (x : ℝ) :
    Tendsto (fun n => F n x) atTop (𝓝 (f x)) :=
  h.tendsto_at_nhds x

-- 6) CauchySeq from sSup bound
example (seq : ℕ → MedicusMin) (x : ℝ)
    (hbdd : ∀ m n : ℕ, BddAbove (Set.range fun y => |seq m y - seq n y|))
    (hcauchy : ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N →
        sSup (Set.range fun y => |seq m y - seq n y|) < ε) :
    CauchySeq (fun n => seq n x) := by
  rw [Metric.cauchySeq_iff]
  intro ε hε
  obtain ⟨N, hN⟩ := hcauchy ε hε
  exact ⟨N, fun m hm n hn => by
    rw [Real.dist_eq]
    calc |seq m x - seq n x|
        ≤ sSup (Set.range fun y => |seq m y - seq n y|) :=
          le_csSup (hbdd m n) (Set.mem_range.mpr ⟨x, rfl⟩)
      _ < ε := hN m n hm hn⟩
