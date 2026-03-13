# MEDICUS 証明仕様書

> **方針（2026-03-10 確定）**
> - `PatientState` は抽象的な状態空間 𝒫 のまま（ℝ³ に固定しない）
> - `noncomm_exists` は抽象的な存在証明（具体的なデータセット・パラメータを使わない）
> - 理由：データの質・量・施設数の論争を回避し、純粋な数学論文として成立させる
>
> 対応論文：`report/SUBMISSION_CANDIDATE_report_v3_math.md`

---

## 0. 前提：型と基本定義

Lean 4 での実装に入る前に確定させる型・定義の一覧。

### 0.1 基本型

```lean
-- 患者状態空間（抽象型、内部構造を指定しない）
variable (𝒫 : Type*) [TopologicalSpace 𝒫]

-- 医療介入の型（𝒫 上の自己写像）
def MedicalIntervention (𝒫 : Type*) := 𝒫 → 𝒫

-- 介入の全体集合
variable (ℰ : Set (MedicalIntervention 𝒫))
```

### 0.2 合成演算

```lean
-- 介入の合成（写像の合成）
def compose (a b : MedicalIntervention 𝒫) : MedicalIntervention 𝒫 :=
  fun p => a (b p)

-- 記法
instance : Mul (MedicalIntervention 𝒫) := ⟨compose⟩
```

### 0.3 単位元

```lean
-- 恒等介入（「何もしない」）
def idIntervention : MedicalIntervention 𝒫 := id
```

---

## 1. Layer 1：非可換モノイドの証明仕様

### 1.1 モノイド公理（閉性・結合法則・単位元）

**証明すること：**
$(MedicalIntervention\,\mathcal{P},\, \circ)$ がモノイドを形成する。

**証明戦略：**
写像の合成は一般に結合法則を満たす——これは Lean 4 / Mathlib の
`Function.comp_assoc` として既に存在する。
モノイドインスタンスはこれを利用して機械的に構成できる。

```lean
instance : Monoid (MedicalIntervention 𝒫) where
  mul       := compose
  one       := idIntervention
  mul_assoc := fun a b c => funext (fun p => rfl)  -- 結合法則
  one_mul   := fun a => funext (fun p => rfl)       -- 左単位元
  mul_one   := fun a => funext (fun p => rfl)       -- 右単位元
```

**Mathlib 依存：**
- `Function.comp_assoc`（結合法則）
- `Function.id_comp`, `Function.comp_id`（単位元）

**難易度：低**——写像の合成の性質から自動的に従う。

---

### 1.2 非可換性の存在（noncomm_exists）

**証明すること：**
ある介入 $E_a, E_b$ が存在して $E_a \circ E_b \neq E_b \circ E_a$。

**証明戦略（抽象的存在証明）：**
具体的なパラメータを使わず、「非可換になる性質」を公理として仮定し存在を示す。

```lean
-- 公理：状態依存的な介入効果の存在
-- 「介入 a の効果が、事前の状態に依存する介入 b が存在する」
axiom state_dependent_intervention :
  ∃ (a b : MedicalIntervention 𝒫) (p : 𝒫),
    a (b p) ≠ b (a p)

-- これから noncomm_exists を導く
theorem noncomm_exists :
  ∃ (a b : MedicalIntervention 𝒫), a * b ≠ b * a := by
  obtain ⟨a, b, p, h⟩ := state_dependent_intervention
  exact ⟨a, b, fun heq => h (congr_fun heq p)⟩
```

**注意（公理について）：**
`state_dependent_intervention` は公理（axiom）として置く。
これは「医療介入には状態依存的なものが存在する」という臨床的事実の
数学的な表現であり、論文の Introduction で生物学的根拠とともに動機づける。
（化療後の免疫抑制が放射線感受性を変える、等）

**Mathlib 依存：**
- `funext` の逆（関数の外延的等号から点ごとの等号）

**難易度：低〜中**——公理の設計が論理的に整合している必要がある。

---

### 1.3 群でないこと（逆元の不在）

**証明すること：**
ある介入 $E$ に対して $E \circ E^{-1} = \mathrm{id}$ を満たす $E^{-1}$ が存在しない。

**証明戦略（抽象的）：**

```lean
-- 公理：不可逆介入の存在
-- 「適用後に元の状態に戻せない介入が存在する」
axiom irreversible_intervention :
  ∃ (a : MedicalIntervention 𝒫),
    ∀ (b : MedicalIntervention 𝒫), ∃ (p : 𝒫), b (a p) ≠ p

-- 逆元が存在しないことの証明
theorem no_inverse :
  ∃ (a : MedicalIntervention 𝒫),
    ¬∃ (b : MedicalIntervention 𝒫), ∀ p, b (a p) = p := by
  obtain ⟨a, ha⟩ := irreversible_intervention
  exact ⟨a, fun ⟨b, hb⟩ => by
    obtain ⟨p, hp⟩ := ha b
    exact hp (hb p)⟩
```

**難易度：低**——公理から直接従う。

---

## 2. Layer 2：MEDICUS 最小空間（Banach 空間）の証明仕様

### 2.0 前提設定

```lean
-- パラメータ領域（有界閉集合）
variable (Ω : Set ℝⁿ) [BoundedSpace Ω] [CompactSpace Ω]

-- MEDICUS 最小空間の台となる関数空間
-- C¹(Ω̄) = 閉包上の C¹ 関数
def MedicusMin (Ω : Set ℝⁿ) := {f : Ω → ℝ | Differentiable ℝ f}
```

### 2.1〜2.3 ノルム公理

**証明すること：**
$\|f\|_{\mathcal{M}_0} = \|f\|_\infty + \|\nabla f\|_\infty$ がノルムの 3 公理を満たす。

```lean
-- ノルムの定義
noncomputable def medicusNorm (f : MedicusMin Ω) : ℝ :=
  ‖f‖_∞ + ‖gradient f‖_∞

-- 正定値性
lemma medicus_pos_def (f : MedicusMin Ω) :
    medicusNorm f = 0 ↔ f = 0 := by
  constructor
  · intro h
    -- ‖f‖∞ ≥ 0 かつ ‖∇f‖∞ ≥ 0 なので両方 0
    -- ‖f‖∞ = 0 → f ≡ 0
    sorry
  · intro h; simp [h, medicusNorm]

-- 斉次性
lemma medicus_smul (λ : ℝ) (f : MedicusMin Ω) :
    medicusNorm (λ • f) = |λ| * medicusNorm f := by
  simp [medicusNorm, norm_smul]
  ring

-- 三角不等式
lemma medicus_triangle (f g : MedicusMin Ω) :
    medicusNorm (f + g) ≤ medicusNorm f + medicusNorm g := by
  simp [medicusNorm]
  -- ‖f+g‖∞ ≤ ‖f‖∞ + ‖g‖∞ は norm_add_le
  -- ‖∇(f+g)‖∞ ≤ ‖∇f‖∞ + ‖∇g‖∞ は gradient の線形性から
  gcongr
  · exact norm_add_le f g
  · rw [gradient_add]; exact norm_add_le _ _
```

**Mathlib 依存：**
- `norm_add_le`（三角不等式の基本形）
- `norm_smul`（スカラー倍のノルム）
- `ContinuousLinearMap` の線形性（勾配の加法性）

**難易度：中**——`gradient` の Mathlib での定義と整合させる必要がある。

---

### 2.4 完備性（Banach 空間）

**証明すること：**
$\mathcal{M}_0$ の任意の Cauchy 列が $\mathcal{M}_0$ 内に収束する。

**証明戦略：**
1. $\|f\|_\infty \leq \|f\|_{\mathcal{M}_0}$ より Cauchy 列は $\|\cdot\|_\infty$ でも Cauchy
2. $(C(\bar\Omega), \|\cdot\|_\infty)$ の完備性（Mathlib 既知）から極限 $f$ が存在
3. $\|\nabla f_n - \nabla f_m\|_\infty \to 0$ より $\{\nabla f_n\}$ も一様収束
4. 一様収束と微分の交換（Mathlib: `hasFDerivAt_of_tendsto_uniformly`）から $\nabla f$ が存在
5. よって $f \in C^1 = \mathcal{M}_0$

```lean
theorem medicus_complete :
    CompleteSpace (MedicusMin Ω) := by
  apply Metric.complete_of_cauchySeq_tendsto
  intro f hf
  -- Step 1: C(Ω̄) での完備性を使う
  have hf_inf : CauchySeq (fun n => (f n : C(Ω, ℝ))) := ...
  obtain ⟨g, hg⟩ := cauchySeq_tendsto_of_complete hf_inf
  -- Step 2: 勾配列の一様収束
  have hdf : CauchySeq (fun n => gradient (f n)) := ...
  -- Step 3: 微分と極限の交換
  exact ⟨g, by exact hasFDerivAt_of_tendsto_uniformly ...⟩
```

**Mathlib 依存：**
- `ContinuousMap.completespace`（$C(\bar\Omega)$ の完備性）
- `hasFDerivAt_of_tendsto_uniformly`（一様収束と微分の交換）
- `cauchySeq_tendsto_of_complete`

**難易度：高**——`hasFDerivAt_of_tendsto_uniformly` の適用条件の整合が最難関。

---

## 3. Layer 3：Mollifier の証明仕様

### 3.1 C∞ 性

**証明すること：**
$f \in L^1,\ \varepsilon > 0$ ならば $f_\varepsilon = f * \phi_\varepsilon \in C^\infty$。

**証明戦略：**
Mathlib に `ContDiff` + `BumpFunction` の定理が存在する。
医療制約付き Mollifier $\phi_\varepsilon^{\mathcal{C}}$ は標準 Mollifier を
制約領域 $\Omega_{\mathcal{C}}$ に制限したものなので、
Mathlib の定理を制限付き領域に適用する形で証明できる。

```lean
-- Mathlib の既存定理を流用
-- Mathlib.Analysis.Calculus.BumpFunction.Basic
#check ContDiffBump.contDiff  -- φ ∈ C∞

theorem mollifier_smooth (f : MedicusMin Ω) (ε : ℝ) (hε : 0 < ε) :
    ContDiff ℝ ⊤ (mollify f ε) := by
  apply ContDiff.comp
  · exact ContDiffBump.contDiff ...
  · exact contDiff_const
```

**Mathlib 依存：**
- `Mathlib.Analysis.Calculus.BumpFunction.Basic`
- `ContDiff.comp`（合成の滑らかさ）
- `MeasureTheory.convolution_contDiff`（畳み込みの滑らかさ）

**難易度：中**——Mathlib の畳み込み定理の型合わせが必要。

---

### 3.2 収束性

**証明すること：**
$\|f_\varepsilon - f\|_{\mathcal{M}_0} \to 0$（$\varepsilon \to 0$）。

**証明戦略：**
- $\|f_\varepsilon - f\|_\infty \to 0$：Mathlib `MeasureTheory.tendsto_convolution`
- $\|\nabla f_\varepsilon - \nabla f\|_\infty \to 0$：$\nabla f_\varepsilon = (\nabla f) * \phi_\varepsilon$ より同様

```lean
theorem mollifier_converges (f : MedicusMin Ω) :
    Filter.Tendsto (fun ε => medicusNorm (mollify f ε - f))
                   (nhds 0) (nhds 0) := by
  apply tendsto_add
  · exact MeasureTheory.tendsto_convolution_left ...
  · -- ∇(f*φ) = (∇f)*φ の交換と同様の収束
    exact gradient_mollifier_converges f
```

**Mathlib 依存：**
- `MeasureTheory.tendsto_convolution`
- `MeasureTheory.hasFDerivAt_convolution`（勾配と畳み込みの交換）

**難易度：中**——勾配と畳み込みの交換定理の探索が必要。

---

### 3.3 フレシェ微分可能性（系）

**証明すること：**
$f_\varepsilon \in C^\infty$ ならばフレシェ微分可能であり連鎖律が成立する。

```lean
-- C∞ → フレシェ微分可能は Mathlib の自動帰結
theorem mollifier_frechet_diff (f : MedicusMin Ω) (ε : ℝ) (hε : 0 < ε) :
    HasFDerivAt (mollify f ε)
                (fderiv ℝ (mollify f ε) ·)
                · := by
  exact (mollifier_smooth f ε hε).differentiable le_top |>.hasFDerivAt
```

**Mathlib 依存：**
- `ContDiff.differentiable`（C∞ → 微分可能）
- `Differentiable.hasFDerivAt`

**難易度：低**——3.1 の系として自動的に従う。

---

## 4. 公理リスト（論文でモチベートが必要なもの）

数学的には公理として置くが、論文本文で生物学的・臨床的に動機づける必要がある。

| 公理 | 数学的内容 | 論文での動機 |
|---|---|---|
| `state_dependent_intervention` | 状態依存的な介入が存在する | 化療後の免疫抑制が放射線感受性を変える |
| `irreversible_intervention` | 不可逆な介入が存在する | 外科的切除・放射線の組織変化は元に戻らない |

---

## 5. 未解決・要検討事項

| 項目 | 内容 | 対応 |
|---|---|---|
| `MedicusMin` の Lean 型定義 | `{f : Ω → ℝ \| Differentiable ℝ f}` の部分型をどう扱うか | `Subtype` か `Structure` か |
| `gradient` の定義 | Mathlib では `fderiv` が標準。`‖fderiv ℝ f x‖` として扱う | 型合わせが必要 |
| 完備性証明の `hasFDerivAt_of_tendsto_uniformly` | 適用条件（一様収束 + 各点微分存在）の整合 | 最難関箇所 |
| Ω の具体化 | 論文では有界開集合としか言っていない。Lean では `[IsOpen Ω] [Bounded Ω]` が必要 | 定義で解決 |

---

## 6. 実装順序（再掲）

```
Step 1  Lean 4 プロジェクト初期化（mathlib4 依存追加）
Step 2  基本型定義（PatientState 抽象型、MedicalIntervention）
Step 3  公理 2 つの定式化（state_dependent / irreversible）
Step 4  1.1〜1.3：Monoid instance（簡単・自信をつける）
Step 5  1.4〜1.5：noncomm_exists, no_inverse
Step 6  2.1〜2.3：ノルム公理（medicusNorm）
Step 7  3.1〜3.3：Mollifier（Mathlib 流用中心）
Step 8  2.4：完備性（最難関、最後に回す）
Step 9  Appendix A としてまとめ、arXiv 投稿
```

> Step 4〜5 が完了した時点で「部分的に形式検証済み」として投稿可能。
> 完備性（Step 8）は査読後の revision で追加しても許容範囲。
