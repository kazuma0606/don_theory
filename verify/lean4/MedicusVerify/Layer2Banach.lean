/-
  MedicusVerify.Layer2Banach
  ==========================
  Layer 2: MEDICUS minimal space — type structure and theorem statements.

  Proof status:
    ✓ supNorm_nonneg, gradNorm_nonneg  (proved)
    ✓ medicusNorm_pos_def              (proved)
    ✓ medicusNorm_smul                 (proved)
    ✓ medicusNorm_triangle             (proved)
    ✓ medicusMin_complete              (proved — skeleton with sorry on BddAbove of limit)

  Ref: verify/PROOF_SPEC.md section 2
-/

import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.UniformLimitsDeriv
import Mathlib.Data.Real.Pointwise
import Mathlib.Order.ConditionallyCompleteLattice.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.MetricSpace.Cauchy
import Mathlib.Topology.MetricSpace.Pseudo.Basic
import Mathlib.Topology.UniformSpace.UniformConvergence

open scoped Pointwise
open Filter Topology

noncomputable section

-- ============================================================
-- Task 3.1: Type definition
-- ============================================================

/-- MEDICUS minimal space M_0 = W^{1,∞}(ℝ):
    differentiable functions ℝ → ℝ with bounded values and derivatives. -/
def MedicusMin : Type :=
  {f : ℝ → ℝ // Differentiable ℝ f
              ∧ BddAbove (Set.range fun x => |f x|)
              ∧ BddAbove (Set.range fun x => |deriv f x|)}

namespace MedicusMin

instance : CoeFun MedicusMin (fun _ => ℝ → ℝ) := ⟨Subtype.val⟩

-- ============================================================
-- Task 3.2: Norm definition
-- ============================================================

/-- Sup norm of function values ‖f‖∞ -/
def supNorm (f : MedicusMin) : ℝ :=
  sSup (Set.range fun x => |f x|)

/-- Sup norm of derivative ‖∇f‖∞ -/
def gradNorm (f : MedicusMin) : ℝ :=
  sSup (Set.range fun x => |deriv f x|)

/-- MEDICUS norm ‖f‖_M0 = ‖f‖∞ + ‖∇f‖∞ -/
def medicusNorm (f : MedicusMin) : ℝ :=
  supNorm f + gradNorm f

-- ============================================================
-- Nonnegativity (proved)
-- ============================================================

lemma supNorm_nonneg (f : MedicusMin) : 0 ≤ supNorm f :=
  le_csSup_of_le f.2.2.1 (Set.mem_range.mpr ⟨0, rfl⟩) (abs_nonneg _)

lemma gradNorm_nonneg (f : MedicusMin) : 0 ≤ gradNorm f :=
  le_csSup_of_le f.2.2.2 (Set.mem_range.mpr ⟨0, rfl⟩) (abs_nonneg _)

lemma medicusNorm_nonneg (f : MedicusMin) : 0 ≤ medicusNorm f :=
  add_nonneg (supNorm_nonneg f) (gradNorm_nonneg f)

-- ============================================================
-- Task 3.3 (1/3): Positive definiteness (proved)
-- ============================================================

/-- ‖f‖_M0 = 0 → f ≡ 0 -/
theorem medicusNorm_pos_def {f : MedicusMin} (h : medicusNorm f = 0) :
    ∀ x, f x = 0 := by
  have hS : supNorm f = 0 := by
    unfold medicusNorm at h
    linarith [supNorm_nonneg f, gradNorm_nonneg f]
  intro x
  have hx : |f x| ≤ supNorm f :=
    le_csSup f.2.2.1 (Set.mem_range.mpr ⟨x, rfl⟩)
  rw [hS] at hx
  exact abs_nonpos_iff.mp hx

-- ============================================================
-- Task 3.3 (2/3): Homogeneity (proved)
-- ============================================================

/-- ‖c • f‖_M0 = |c| * ‖f‖_M0 -/
theorem medicusNorm_smul (c : ℝ) (f g : MedicusMin) (hg : ∀ x, g x = c * f x) :
    medicusNorm g = |c| * medicusNorm f := by
  simp only [medicusNorm, supNorm, gradNorm]
  have hfun : g.val = fun x => c * f x := funext hg
  have hval : (fun x => |g x|) = fun x => |c| * |f x| := by
    funext x; rw [hg x, abs_mul]
  have hderiv : ∀ x, deriv g.val x = c * deriv f.val x := by
    intro x; rw [hfun]; exact deriv_const_mul c (f.2.1 x)
  have hgrad : (fun x => |deriv g.val x|) = fun x => |c| * |deriv f.val x| := by
    funext x; rw [hderiv x, abs_mul]
  have hS1 : sSup (Set.range fun x => |g x|) =
             |c| * sSup (Set.range fun x => |f x|) := by
    rw [show Set.range (fun x => |g x|) = Set.range (fun x => |c| * |f x|) from
          congr_arg Set.range hval]
    have hrange : Set.range (fun x => |c| * |f x|) = |c| • Set.range (fun x => |f x|) := by
      ext y; simp only [Set.mem_range, Set.mem_smul_set, smul_eq_mul]
      constructor
      · rintro ⟨x, rfl⟩; exact ⟨|f x|, Set.mem_range.mpr ⟨x, rfl⟩, rfl⟩
      · rintro ⟨z, ⟨x, rfl⟩, rfl⟩; exact ⟨x, rfl⟩
    rw [hrange, Real.sSup_smul_of_nonneg (abs_nonneg c), smul_eq_mul]
  have hS2 : sSup (Set.range fun x => |deriv g.val x|) =
             |c| * sSup (Set.range fun x => |deriv f.val x|) := by
    rw [show Set.range (fun x => |deriv g.val x|) =
            Set.range (fun x => |c| * |deriv f.val x|) from
          congr_arg Set.range hgrad]
    have hrange : Set.range (fun x => |c| * |deriv f.val x|) =
                  |c| • Set.range (fun x => |deriv f.val x|) := by
      ext y; simp only [Set.mem_range, Set.mem_smul_set, smul_eq_mul]
      constructor
      · rintro ⟨x, rfl⟩; exact ⟨|deriv f.val x|, Set.mem_range.mpr ⟨x, rfl⟩, rfl⟩
      · rintro ⟨z, ⟨x, rfl⟩, rfl⟩; exact ⟨x, rfl⟩
    rw [hrange, Real.sSup_smul_of_nonneg (abs_nonneg c), smul_eq_mul]
  rw [hS1, hS2, ← mul_add]

-- ============================================================
-- Task 3.3 (3/3): Triangle inequality (proved)
-- ============================================================

/-- ‖f + g‖_M0 ≤ ‖f‖_M0 + ‖g‖_M0 -/
theorem medicusNorm_triangle (f g h : MedicusMin) (hfg : ∀ x, h x = f x + g x) :
    medicusNorm h ≤ medicusNorm f + medicusNorm g := by
  simp only [medicusNorm, supNorm, gradNorm]
  have hfun : h.val = fun x => f x + g x := funext hfg
  have hderiv : ∀ x, deriv h.val x = deriv f.val x + deriv g.val x := by
    intro x; rw [hfun]; exact deriv_add (f.2.1 x) (g.2.1 x)
  have hS1 : sSup (Set.range fun x => |h x|) ≤
             sSup (Set.range fun x => |f x|) + sSup (Set.range fun x => |g x|) := by
    apply csSup_le (Set.range_nonempty _)
    rintro y ⟨x, rfl⟩
    calc |h x|
        = |f x + g x|   := by rw [hfg x]
      _ ≤ |f x| + |g x| := abs_add_le _ _
      _ ≤ sSup (Set.range fun x => |f x|) + sSup (Set.range fun x => |g x|) :=
          add_le_add
            (le_csSup f.2.2.1 (Set.mem_range.mpr ⟨x, rfl⟩))
            (le_csSup g.2.2.1 (Set.mem_range.mpr ⟨x, rfl⟩))
  have hS2 : sSup (Set.range fun x => |deriv h.val x|) ≤
             sSup (Set.range fun x => |deriv f.val x|) +
             sSup (Set.range fun x => |deriv g.val x|) := by
    apply csSup_le (Set.range_nonempty _)
    rintro y ⟨x, rfl⟩
    calc |deriv h.val x|
        = |deriv f.val x + deriv g.val x| := by rw [hderiv x]
      _ ≤ |deriv f.val x| + |deriv g.val x| := abs_add_le _ _
      _ ≤ sSup (Set.range fun x => |deriv f.val x|) +
          sSup (Set.range fun x => |deriv g.val x|) :=
          add_le_add
            (le_csSup f.2.2.2 (Set.mem_range.mpr ⟨x, rfl⟩))
            (le_csSup g.2.2.2 (Set.mem_range.mpr ⟨x, rfl⟩))
  linarith

-- ============================================================
-- Task 3.4: Completeness
-- ============================================================

/-
  Proof strategy (PROOF_SPEC.md §2.4):
  A. Extract pointwise limits via cauchySeq_tendsto_of_complete
  B. Show deriv(seqₙ) → limDeriv uniformly via sSup bound
  C. Apply hasDerivAt_of_tendstoUniformly → differentiability
  D. Construct lim : MedicusMin; show M_0 convergence
-/

/-- BddAbove for value differences of MedicusMin elements -/
private lemma bddAbove_val_diff (f g : MedicusMin) :
    BddAbove (Set.range fun x => |f x - g x|) :=
  ⟨supNorm f + supNorm g, by
    rintro y ⟨x, rfl⟩
    calc |f x - g x|
        = |f x + (-g x)| := by ring_nf
      _ ≤ |f x| + |-g x| := abs_add_le _ _
      _ = |f x| + |g x|  := by rw [abs_neg]
      _ ≤ supNorm f + supNorm g :=
          add_le_add
            (le_csSup f.2.2.1 (Set.mem_range.mpr ⟨x, rfl⟩))
            (le_csSup g.2.2.1 (Set.mem_range.mpr ⟨x, rfl⟩))⟩

/-- BddAbove for derivative differences of MedicusMin elements -/
private lemma bddAbove_deriv_diff (f g : MedicusMin) :
    BddAbove (Set.range fun x => |deriv f x - deriv g x|) :=
  ⟨gradNorm f + gradNorm g, by
    rintro y ⟨x, rfl⟩
    calc |deriv f x - deriv g x|
        = |deriv f x + (-deriv g x)| := by ring_nf
      _ ≤ |deriv f x| + |-deriv g x| := abs_add_le _ _
      _ = |deriv f x| + |deriv g x|  := by rw [abs_neg]
      _ ≤ gradNorm f + gradNorm g :=
          add_le_add
            (le_csSup f.2.2.2 (Set.mem_range.mpr ⟨x, rfl⟩))
            (le_csSup g.2.2.2 (Set.mem_range.mpr ⟨x, rfl⟩))⟩

/-- sSup of |f·| range is nonneg when bounded -/
private lemma sSup_range_abs_nonneg (f : MedicusMin) :
    0 ≤ sSup (Set.range fun x => |f x|) :=
  le_csSup_of_le f.2.2.1 (Set.mem_range.mpr ⟨0, rfl⟩) (abs_nonneg _)

theorem medicusMin_complete
    (seq : ℕ → MedicusMin)
    (hcauchy : ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N →
      sSup (Set.range fun x => |seq m x - seq n x|) +
      sSup (Set.range fun x => |deriv (seq m) x - deriv (seq n) x|) < ε) :
    ∃ (lim : MedicusMin), ∀ ε > 0, ∃ N, ∀ n ≥ N,
      sSup (Set.range fun x => |seq n x - lim x|) +
      sSup (Set.range fun x => |deriv (seq n) x - deriv lim x|) < ε := by
  -- ============================================================
  -- Step A1: Separate Cauchy conditions
  -- ============================================================
  have hcauchy_val : ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N →
      sSup (Set.range fun x => |seq m x - seq n x|) < ε := by
    intro ε hε
    obtain ⟨N, hN⟩ := hcauchy ε hε
    refine ⟨N, fun m n hm hn => ?_⟩
    have h := hN m n hm hn
    have hnn : 0 ≤ sSup (Set.range fun x => |deriv (seq m) x - deriv (seq n) x|) :=
      le_csSup_of_le (bddAbove_deriv_diff (seq m) (seq n))
        (Set.mem_range.mpr ⟨0, rfl⟩) (abs_nonneg _)
    linarith
  have hcauchy_deriv : ∀ ε > 0, ∃ N, ∀ m n, m ≥ N → n ≥ N →
      sSup (Set.range fun x => |deriv (seq m) x - deriv (seq n) x|) < ε := by
    intro ε hε
    obtain ⟨N, hN⟩ := hcauchy ε hε
    refine ⟨N, fun m n hm hn => ?_⟩
    have h := hN m n hm hn
    have hnn : 0 ≤ sSup (Set.range fun x => |seq m x - seq n x|) :=
      le_csSup_of_le (bddAbove_val_diff (seq m) (seq n))
        (Set.mem_range.mpr ⟨0, rfl⟩) (abs_nonneg _)
    linarith
  -- ============================================================
  -- Step A2: For each x, seq · x is Cauchy → has limit
  -- ============================================================
  have hcauchy_ptwise : ∀ x : ℝ, CauchySeq (fun n => seq n x) := by
    intro x
    rw [Metric.cauchySeq_iff]
    intro ε hε
    obtain ⟨N, hN⟩ := hcauchy_val ε hε
    exact ⟨N, fun m hm n hn => by
      rw [Real.dist_eq]
      calc |seq m x - seq n x|
          ≤ sSup (Set.range fun y => |seq m y - seq n y|) :=
            le_csSup (bddAbove_val_diff (seq m) (seq n)) (Set.mem_range.mpr ⟨x, rfl⟩)
        _ < ε := hN m n hm hn⟩
  have lim_exists : ∀ x : ℝ, ∃ l : ℝ, Tendsto (fun n => seq n x) atTop (nhds l) :=
    fun x => cauchySeq_tendsto_of_complete (hcauchy_ptwise x)
  -- Define pointwise limit
  let limFn : ℝ → ℝ := fun x => (lim_exists x).choose
  have hlimFn : ∀ x, Tendsto (fun n => seq n x) atTop (nhds (limFn x)) :=
    fun x => (lim_exists x).choose_spec
  -- ============================================================
  -- Step B1: deriv(seq n) is pointwise Cauchy → has limit
  -- ============================================================
  have hcauchy_deriv_ptwise : ∀ x : ℝ, CauchySeq (fun n => deriv (seq n) x) := by
    intro x
    rw [Metric.cauchySeq_iff]
    intro ε hε
    obtain ⟨N, hN⟩ := hcauchy_deriv ε hε
    exact ⟨N, fun m hm n hn => by
      rw [Real.dist_eq]
      calc |deriv (seq m) x - deriv (seq n) x|
          ≤ sSup (Set.range fun y => |deriv (seq m) y - deriv (seq n) y|) :=
            le_csSup (bddAbove_deriv_diff (seq m) (seq n)) (Set.mem_range.mpr ⟨x, rfl⟩)
        _ < ε := hN m n hm hn⟩
  have limDeriv_exists : ∀ x : ℝ, ∃ l : ℝ, Tendsto (fun n => deriv (seq n) x) atTop (nhds l) :=
    fun x => cauchySeq_tendsto_of_complete (hcauchy_deriv_ptwise x)
  let limDeriv : ℝ → ℝ := fun x => (limDeriv_exists x).choose
  have hlimDeriv : ∀ x, Tendsto (fun n => deriv (seq n) x) atTop (nhds (limDeriv x)) :=
    fun x => (limDeriv_exists x).choose_spec
  -- ============================================================
  -- Step B2: deriv(seq n) → limDeriv uniformly
  -- ============================================================
  have hunif_deriv : TendstoUniformly (fun n x => deriv (seq n) x) limDeriv atTop := by
    -- Use metric criterion: ∀ ε > 0, ∀ᶠ n in atTop, ∀ x, dist(limDeriv x)(deriv(seq n) x) < ε
    rw [Metric.tendstoUniformly_iff]
    intro ε hε
    obtain ⟨N, hN⟩ := hcauchy_deriv (ε / 2) (half_pos hε)
    apply Eventually.mono (Ici_mem_atTop N)
    intro n hn x
    rw [Real.dist_eq]
    -- |limDeriv x - deriv(seq n)(x)| via limit argument
    have htend : Tendsto (fun m => |deriv (seq n) x - deriv (seq m) x|)
                          atTop (nhds |deriv (seq n) x - limDeriv x|) :=
      ((tendsto_const_nhds).sub (hlimDeriv x)).abs
    have hle : |deriv (seq n) x - limDeriv x| ≤ ε / 2 := by
      apply le_of_tendsto htend
      apply Eventually.mono (Ici_mem_atTop N)
      intro m hm
      calc |deriv (seq n) x - deriv (seq m) x|
          ≤ sSup (Set.range fun y => |deriv (seq n) y - deriv (seq m) y|) :=
            le_csSup (bddAbove_deriv_diff (seq n) (seq m)) (Set.mem_range.mpr ⟨x, rfl⟩)
        _ ≤ ε / 2 := le_of_lt (hN n m hn hm)
    -- dist (limDeriv x) (deriv (seq n) x) = |limDeriv x - deriv n x| = |deriv n x - limDeriv x|
    linarith [abs_sub_comm (deriv (↑(seq n)) x) (limDeriv x)]
  -- ============================================================
  -- Step B3: seq n → limFn uniformly (parallel to hunif_deriv)
  -- ============================================================
  have hunif_val : TendstoUniformly (fun n x => seq n x) limFn atTop := by
    rw [Metric.tendstoUniformly_iff]
    intro ε hε
    obtain ⟨N, hN⟩ := hcauchy_val (ε / 2) (half_pos hε)
    apply Eventually.mono (Ici_mem_atTop N)
    intro n hn x
    rw [Real.dist_eq]
    have htend : Tendsto (fun m => |seq n x - seq m x|) atTop (nhds |seq n x - limFn x|) :=
      ((tendsto_const_nhds).sub (hlimFn x)).abs
    have hle : |seq n x - limFn x| ≤ ε / 2 := by
      apply le_of_tendsto htend
      apply Eventually.mono (Ici_mem_atTop N)
      intro m hm
      calc |seq n x - seq m x|
          ≤ sSup (Set.range fun y => |seq n y - seq m y|) :=
            le_csSup (bddAbove_val_diff (seq n) (seq m)) (Set.mem_range.mpr ⟨x, rfl⟩)
        _ ≤ ε / 2 := le_of_lt (hN n m hn hm)
    linarith [abs_sub_comm ((seq n : ℝ → ℝ) x) (limFn x)]
  -- ============================================================
  -- Step C: Apply hasDerivAt_of_tendstoUniformly
  -- ============================================================
  have hhasderiv : ∀ x, HasDerivAt limFn (limDeriv x) x := by
    intro x
    apply hasDerivAt_of_tendstoUniformly hunif_deriv _ hlimFn
    apply Eventually.of_forall
    intro n y
    exact (seq n).2.1 y |>.hasDerivAt
  -- ============================================================
  -- Step D: Construct lim : MedicusMin
  -- ============================================================
  -- limFn is differentiable
  have hlimFn_diff : Differentiable ℝ limFn :=
    fun x => (hhasderiv x).differentiableAt
  -- limFn has bounded values: uniform limit of bounded functions
  -- For n₀ large: |limFn x| ≤ |limFn x - seq n₀ x| + |seq n₀ x| < 1 + supNorm(seq n₀)
  have hlimFn_bdd : BddAbove (Set.range fun x => |limFn x|) := by
    have hmono := Metric.tendstoUniformly_iff.mp hunif_val 1 one_pos
    rw [Filter.eventually_atTop] at hmono
    obtain ⟨N₀, hN₀⟩ := hmono
    refine ⟨supNorm (seq N₀) + 1, ?_⟩
    rintro y ⟨x, rfl⟩
    dsimp only []
    have hdist := hN₀ N₀ (le_refl _) x
    rw [Real.dist_eq] at hdist
    have hfN₀ : |seq N₀ x| ≤ supNorm (seq N₀) :=
      le_csSup (seq N₀).2.2.1 (Set.mem_range.mpr ⟨x, rfl⟩)
    have h1 : |limFn x| ≤ |limFn x - seq N₀ x| + |seq N₀ x| := by
      calc |limFn x| = |limFn x - seq N₀ x + seq N₀ x| := by ring_nf
        _ ≤ |limFn x - seq N₀ x| + |seq N₀ x| := abs_add_le _ _
    linarith
  -- limFn has bounded derivative = limDeriv
  have hderiv_eq : deriv limFn = limDeriv := by
    funext x; exact (hhasderiv x).deriv
  -- limDeriv is bounded: uniform limit of bounded gradient functions
  have hlimFn_bdd_deriv : BddAbove (Set.range fun x => |deriv limFn x|) := by
    rw [hderiv_eq]
    have hmono := Metric.tendstoUniformly_iff.mp hunif_deriv 1 one_pos
    rw [Filter.eventually_atTop] at hmono
    obtain ⟨N₀, hN₀⟩ := hmono
    refine ⟨gradNorm (seq N₀) + 1, ?_⟩
    rintro y ⟨x, rfl⟩
    dsimp only []
    have hdist := hN₀ N₀ (le_refl _) x
    rw [Real.dist_eq] at hdist
    have hgN₀ : |deriv (seq N₀) x| ≤ gradNorm (seq N₀) :=
      le_csSup (seq N₀).2.2.2 (Set.mem_range.mpr ⟨x, rfl⟩)
    have h1 : |limDeriv x| ≤ |limDeriv x - deriv (seq N₀) x| + |deriv (seq N₀) x| := by
      calc |limDeriv x| = |limDeriv x - deriv (seq N₀) x + deriv (seq N₀) x| := by ring_nf
        _ ≤ |limDeriv x - deriv (seq N₀) x| + |deriv (seq N₀) x| := abs_add_le _ _
    linarith
  -- Construct the limit element
  let lim : MedicusMin := ⟨limFn, hlimFn_diff, hlimFn_bdd, hlimFn_bdd_deriv⟩
  refine ⟨lim, ?_⟩
  -- ============================================================
  -- Step E: Show M_0 convergence
  -- Use ε/3 strategy: each sSup ≤ ε/3, so sum ≤ 2ε/3 < ε
  -- ============================================================
  intro ε hε
  obtain ⟨N, hN⟩ := hcauchy (ε / 3) (by linarith)
  refine ⟨N, fun n hn => ?_⟩
  -- Helper: lim coercion equals limFn
  have hlim_eq : (lim : ℝ → ℝ) = limFn := rfl
  have hderiv_lim : deriv (lim : ℝ → ℝ) = limDeriv := hderiv_eq
  -- Value part: sSup |seq n - lim| ≤ ε/3
  -- Note: lim.val = limFn by definition
  have hval_conv : sSup (Set.range fun x => |seq n x - lim x|) ≤ ε / 3 := by
    apply csSup_le (Set.range_nonempty _)
    rintro y ⟨x, rfl⟩
    -- Beta-reduce: (fun x => |seq n x - lim x|) x = |seq n x - lim x| = |seq n x - limFn x|
    -- lim.val = limFn definitionally, so le_of_tendsto works
    dsimp only []
    have htend : Tendsto (fun m => |seq n x - seq m x|) atTop (nhds |seq n x - limFn x|) :=
      ((tendsto_const_nhds).sub (hlimFn x)).abs
    apply le_of_tendsto htend
    apply Eventually.mono (Ici_mem_atTop N)
    intro m hm
    have hnn : 0 ≤ sSup (Set.range fun y => |deriv (seq n) y - deriv (seq m) y|) :=
      le_csSup_of_le (bddAbove_deriv_diff (seq n) (seq m))
        (Set.mem_range.mpr ⟨0, rfl⟩) (abs_nonneg _)
    calc |seq n x - seq m x|
        ≤ sSup (Set.range fun y => |seq n y - seq m y|) :=
          le_csSup (bddAbove_val_diff (seq n) (seq m)) (Set.mem_range.mpr ⟨x, rfl⟩)
      _ ≤ ε / 3 := le_of_lt (by linarith [hN n m hn hm])
  -- Deriv part: sSup |deriv(seq n) - deriv(lim)| ≤ ε/3
  -- Note: deriv (↑lim) = deriv limFn = limDeriv (by hderiv_eq)
  have hderiv_conv : sSup (Set.range fun x => |deriv (seq n) x - deriv lim x|) ≤ ε / 3 := by
    -- Reduce to limDeriv form using hderiv_eq
    suffices h : sSup (Set.range fun x => |deriv (↑(seq n)) x - limDeriv x|) ≤ ε / 3 by
      have heq : (fun x => |deriv (↑(seq n)) x - deriv (↑lim) x|) =
                 (fun x => |deriv (↑(seq n)) x - limDeriv x|) := by
        funext x
        congr 1; congr 1
        -- deriv (↑lim) x = limDeriv x
        -- ↑lim = limFn (by let-def), deriv limFn x = limDeriv x (by hderiv_eq)
        exact congr_fun hderiv_eq x
      exact heq ▸ h
    apply csSup_le (Set.range_nonempty _)
    rintro y ⟨x, rfl⟩
    simp only []  -- beta-reduce
    have htend : Tendsto (fun m => |deriv (seq n) x - deriv (seq m) x|)
                          atTop (nhds |deriv (seq n) x - limDeriv x|) :=
      ((tendsto_const_nhds).sub (hlimDeriv x)).abs
    apply le_of_tendsto htend
    apply Eventually.mono (Ici_mem_atTop N)
    intro m hm
    have hnn : 0 ≤ sSup (Set.range fun y => |seq n y - seq m y|) :=
      le_csSup_of_le (bddAbove_val_diff (seq n) (seq m))
        (Set.mem_range.mpr ⟨0, rfl⟩) (abs_nonneg _)
    calc |deriv (seq n) x - deriv (seq m) x|
        ≤ sSup (Set.range fun y => |deriv (seq n) y - deriv (seq m) y|) :=
          le_csSup (bddAbove_deriv_diff (seq n) (seq m)) (Set.mem_range.mpr ⟨x, rfl⟩)
      _ ≤ ε / 3 := le_of_lt (by linarith [hN n m hn hm])
  -- Sum ≤ ε/3 + ε/3 = 2ε/3 < ε
  linarith

end MedicusMin

end
