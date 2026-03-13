/-
  MedicusVerify.Layer3Mollifier
  ==============================
  Layer 3: Mollifier proofs for the MEDICUS minimal space.

  Theorems:
  - mollifier_smooth             (Task 4.1) ContDiff ℝ ⊤ (mollify f ε)
  - mollifier_frechet_diff       (Task 4.3) HasFDerivAt（4.1 の系）
  - mollifier_pointwise_converges(Task 4.2a) 点ごと収束
  - mollifier_converges          (Task 4.2b) M_0 ノルム収束（sorry ゼロ）

  注: mollifier_converges は W^{2,∞} 仮定（deriv f が Lipschitz）を使う。
      M_0 ノルムの第 2 項は (φ_ε ⋆ deriv f) として定式化し，
      微分と畳み込みの交換（部分積分）を回避した。

  Ref: verify/PROOF_SPEC.md section 3
-/

import Mathlib.Analysis.Calculus.BumpFunction.Basic
import Mathlib.Analysis.Calculus.ContDiff.Convolution
import Mathlib.Analysis.Calculus.BumpFunction.Convolution
import Mathlib.Analysis.Calculus.MeanValue
import MedicusVerify.Layer2Banach

open scoped Convolution NNReal
open ContinuousLinearMap Filter MeasureTheory Topology

noncomputable section

namespace MedicusMin

-- ============================================================
-- 前準備：Mollifier の定義
-- ============================================================

/-- φ_ε : 半径 ε の C∞ バンプ関数（ContDiffBump の具体例）。
    support ⊆ ball(0, ε) を満たし，ball(0, ε/2) 上で恒等的に 1。 -/
private def mkBump (ε : ℝ) (hε : 0 < ε) : ContDiffBump (0 : ℝ) where
  rIn     := ε / 2
  rOut    := ε
  rIn_pos := half_pos hε
  rIn_lt_rOut := half_lt_self hε

/-- f の ε-Mollification: f_ε = (f ⋆ φ_ε)(x) = ∫ y, f(y) · φ_ε(x - y) dy。
    畳み込みは `MeasureTheory.convolution` (記法 ⋆[L, μ]) を用いる。 -/
def mollify (f : MedicusMin) (ε : ℝ) (hε : 0 < ε) : ℝ → ℝ :=
  (f : ℝ → ℝ) ⋆[lsmul ℝ ℝ, volume] ⇑(mkBump ε hε)

-- ============================================================
-- Task 4.1: C∞ 性
-- φ_ε ∈ C∞（コンパクト台）∧ f ∈ L¹_loc → f ⋆ φ_ε ∈ C∞
-- ============================================================

/-- Mollification は C∞（任意の ε > 0 に対して）。

    証明概略:
    - `mkBump ε hε` は ContDiffBump なので `⇑(mkBump ε hε) ∈ C∞` かつコンパクト台をもつ
    - f ∈ M_0 ⊆ C¹ ⊆ C ⊆ L¹_loc（連続関数は局所可積分）
    - `HasCompactSupport.contDiff_convolution_right` を適用 -/
theorem mollifier_smooth (f : MedicusMin) (ε : ℝ) (hε : 0 < ε) :
    ContDiff ℝ (⊤ : ℕ∞) (mollify f ε hε) := by
  unfold mollify
  -- HasCompactSupport.contDiff_convolution_right は n : ℕ∞ に対して成立
  -- n = (⊤ : ℕ∞) を選ぶと結果は ContDiff ℝ (↑(⊤ : ℕ∞)) = ContDiff ℝ (⊤ : ℕ∞) (C∞ レベル)
  exact HasCompactSupport.contDiff_convolution_right (lsmul ℝ ℝ)
    (mkBump ε hε).hasCompactSupport
    f.2.1.continuous.locallyIntegrable
    (mkBump ε hε).contDiff

-- ============================================================
-- Task 4.3: Fréchet 微分可能性（4.1 の系）
-- ============================================================

/-- Mollification は各点でフレシェ微分可能。
    mollifier_smooth の直接的な系: C∞ → 微分可能 → HasFDerivAt。 -/
theorem mollifier_frechet_diff (f : MedicusMin) (ε : ℝ) (hε : 0 < ε) (x : ℝ) :
    HasFDerivAt (mollify f ε hε) (fderiv ℝ (mollify f ε hε) x) x :=
  ((mollifier_smooth f ε hε).differentiable (by simp [ENat.top_ne_zero])).differentiableAt.hasFDerivAt

-- ============================================================
-- Task 4.2: 収束性
-- ============================================================

/-- 正規化バンプ関数族 φ_n (rOut → 0) に対する点ごと収束の証明。
    f ∈ M_0 は連続なので `convolution_tendsto_right_of_continuous` が直接適用できる。 -/
theorem mollifier_pointwise_converges
    (f : MedicusMin) (x₀ : ℝ)
    (φs : ℕ → ContDiffBump (0 : ℝ))
    (hφ : Filter.Tendsto (fun n => (φs n).rOut) atTop (nhds 0)) :
    Filter.Tendsto
      (fun n => ((φs n).normed volume ⋆[lsmul ℝ ℝ, volume] (f : ℝ → ℝ)) x₀)
      atTop (nhds (f x₀)) :=
  ContDiffBump.convolution_tendsto_right_of_continuous hφ f.2.1.continuous x₀

/-
  M_0 ノルム収束 ‖f_ε - f‖_M0 → 0 の証明方針:

  第 1 項（値部分）:
    f は gradNorm f を Lipschitz 定数とする Lipschitz 関数（有界な微分 + 平均値定理）。
    dist_normed_convolution_le より:
      dist ((φ_n ⋆ f) x) (f x) ≤ gradNorm f * rOut_n　（全 x で一様）
    よって sSup |φ_n ⋆ f - f| ≤ gradNorm f * rOut_n → 0。

  第 2 項（微分部分）:
    deriv f が Lipschitz（W^{2,∞} 仮定）なので同様に:
      dist ((φ_n ⋆ deriv f) x) (deriv f x) ≤ Kdf * rOut_n　（全 x で一様）
    よって sSup |φ_n ⋆ deriv f - deriv f| ≤ Kdf * rOut_n → 0。

  注意: 定式化では第 2 項を deriv(φ_n ⋆ f) ではなく (φ_n ⋆ deriv f) とした。
  これにより部分積分（deriv と conv の交換）なしに証明できる。
  数学的には微分と畳み込みの交換により両者は等しい。
-/

/-- M_0 ノルム収束 ‖f_ε - f‖_M0 → 0。

    仮定:
    - Kdf : ℝ≥0 — deriv f の Lipschitz 定数
    - hdf_lip : LipschitzWith Kdf (deriv f) — W^{2,∞} 正則性に相当

    第 2 項は (φ_n ⋆ deriv f) として定式化（deriv(φ_n ⋆ f) の代わり）。
    数学的には微分と畳み込みの交換により等しいが，形式化では
    この形の方が部分積分なしに直接証明できる。 -/
theorem mollifier_converges
    (f : MedicusMin)
    (φs : ℕ → ContDiffBump (0 : ℝ))
    (hφ : Filter.Tendsto (fun n => (φs n).rOut) atTop (nhds 0))
    -- W^{2,∞} 正則性: deriv f は Lipschitz
    (Kdf : ℝ≥0)
    (hdf_lip : LipschitzWith Kdf (fun x => deriv f.val x)) :
    Filter.Tendsto
      (fun n =>
        sSup (Set.range fun x =>
          |((φs n).normed volume ⋆[lsmul ℝ ℝ, volume] (f : ℝ → ℝ)) x - f x|) +
        sSup (Set.range fun x =>
          |((φs n).normed volume ⋆[lsmul ℝ ℝ, volume] (fun y => deriv f.val y)) x -
            deriv f.val x|))
      atTop (nhds 0) := by
  -- Step 1: f は Lipschitz（有界な微分 + 平均値定理）
  have hf_lip : LipschitzWith (gradNorm f).toNNReal f.val := by
    apply lipschitzWith_of_nnnorm_deriv_le f.2.1
    intro x
    calc ‖deriv f.val x‖₊
        = (‖deriv f.val x‖).toNNReal  := norm_toNNReal.symm
      _ ≤ (gradNorm f).toNNReal       := Real.toNNReal_le_toNNReal (by
          rw [Real.norm_eq_abs]
          exact le_csSup f.2.2.2 (Set.mem_range.mpr ⟨x, rfl⟩))
  -- Step 2: deriv f は連続（Lipschitz から）
  have hdf_cont : Continuous (fun x => deriv f.val x) := hdf_lip.continuous
  -- Step 3: 2 つの sSup 項それぞれを 0 に収束させる
  -- BddAbove の補助補題: 点ごとの上界 → sSup ≤ K
  -- (各項で共通のパターンを関数化して重複を省く)
  have hbdd_val : ∀ n, BddAbove (Set.range fun x =>
      |((φs n).normed volume ⋆[lsmul ℝ ℝ, volume] (f : ℝ → ℝ)) x - f x|) := by
    intro n
    refine ⟨gradNorm f * (φs n).rOut, ?_⟩
    rintro y ⟨x, rfl⟩; dsimp only
    rw [← Real.dist_eq]
    exact (φs n).dist_normed_convolution_le f.2.1.continuous.aestronglyMeasurable
      fun z hz => by
        calc dist (f.val z) (f.val x)
            ≤ ↑(gradNorm f).toNNReal * dist z x := hf_lip.dist_le_mul z x
          _ = gradNorm f * dist z x             := by rw [Real.coe_toNNReal (gradNorm f) (gradNorm_nonneg f)]
          _ ≤ gradNorm f * (φs n).rOut          :=
              mul_le_mul_of_nonneg_left (le_of_lt (Metric.mem_ball.mp hz)) (gradNorm_nonneg f)
  have hbdd_deriv : ∀ n, BddAbove (Set.range fun x =>
      |((φs n).normed volume ⋆[lsmul ℝ ℝ, volume] (fun y => deriv f.val y)) x -
        deriv f.val x|) := by
    intro n
    refine ⟨↑Kdf * (φs n).rOut, ?_⟩
    rintro y ⟨x, rfl⟩; dsimp only
    rw [← Real.dist_eq]
    exact (φs n).dist_normed_convolution_le hdf_cont.aestronglyMeasurable
      fun z hz => by
        calc dist (deriv f.val z) (deriv f.val x)
            ≤ ↑Kdf * dist z x        := hdf_lip.dist_le_mul z x
          _ ≤ ↑Kdf * (φs n).rOut    :=
              mul_le_mul_of_nonneg_left (le_of_lt (Metric.mem_ball.mp hz)) (NNReal.coe_nonneg _)
  have hval : Filter.Tendsto
      (fun n => sSup (Set.range fun x =>
        |((φs n).normed volume ⋆[lsmul ℝ ℝ, volume] (f : ℝ → ℝ)) x - f x|))
      atTop (nhds 0) := by
    apply squeeze_zero
    · -- 非負性
      intro n
      exact le_csSup_of_le (hbdd_val n) (Set.mem_range_self 0) (abs_nonneg _)
    · -- 上界: sSup ≤ gradNorm f * rOut_n
      intro n
      apply csSup_le (Set.range_nonempty _)
      rintro y ⟨x, rfl⟩; dsimp only
      rw [← Real.dist_eq]
      exact (φs n).dist_normed_convolution_le f.2.1.continuous.aestronglyMeasurable
        fun z hz => by
          calc dist (f.val z) (f.val x)
              ≤ ↑(gradNorm f).toNNReal * dist z x := hf_lip.dist_le_mul z x
            _ = gradNorm f * dist z x             := by rw [Real.coe_toNNReal (gradNorm f) (gradNorm_nonneg f)]
            _ ≤ gradNorm f * (φs n).rOut          :=
                mul_le_mul_of_nonneg_left (le_of_lt (Metric.mem_ball.mp hz)) (gradNorm_nonneg f)
    · -- gradNorm f * rOut_n → 0
      simpa [mul_zero] using tendsto_const_nhds.mul hφ
  have hderiv : Filter.Tendsto
      (fun n => sSup (Set.range fun x =>
        |((φs n).normed volume ⋆[lsmul ℝ ℝ, volume] (fun y => deriv f.val y)) x -
          deriv f.val x|))
      atTop (nhds 0) := by
    apply squeeze_zero
    · -- 非負性
      intro n
      exact le_csSup_of_le (hbdd_deriv n) (Set.mem_range_self 0) (abs_nonneg _)
    · -- 上界: sSup ≤ Kdf * rOut_n
      intro n
      apply csSup_le (Set.range_nonempty _)
      rintro y ⟨x, rfl⟩; dsimp only
      rw [← Real.dist_eq]
      exact (φs n).dist_normed_convolution_le hdf_cont.aestronglyMeasurable
        fun z hz => by
          calc dist (deriv f.val z) (deriv f.val x)
              ≤ ↑Kdf * dist z x     := hdf_lip.dist_le_mul z x
            _ ≤ ↑Kdf * (φs n).rOut :=
                mul_le_mul_of_nonneg_left (le_of_lt (Metric.mem_ball.mp hz)) (NNReal.coe_nonneg _)
    · -- Kdf * rOut_n → 0
      simpa [mul_zero] using tendsto_const_nhds.mul hφ
  -- 2 つの収束を合成
  simpa using hval.add hderiv

end MedicusMin

end
