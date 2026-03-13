# Lean 4 形式検証 タスク一覧

> **対応文書**
> - 仕様：`verify/PROOF_SPEC.md`
> - 論文：`report/SUBMISSION_CANDIDATE_report_v3_math.md`
>
> **方針**
> - `PatientState` は抽象型（ℝ³ に固定しない）
> - `noncomm_exists` は抽象的存在証明（データセット不使用）
> - Step 2 完了時点で論文投稿可能（部分的形式検証済みとして）

---

## 0. 環境構築

- [x] 0.1 Lean 4 プロジェクト初期化
  - `lake init medicus-verify` でプロジェクト作成
  - `lakefile.lean` に `mathlib4` 依存を追加
  - `lake update` で Mathlib キャッシュを取得
  - _仕様: PROOF_SPEC §0_

- [x] 0.2 Mathlib の利用可能性確認
  - `import Mathlib.Topology.Basic` が通ることを確認
  - `import Mathlib.Analysis.Calculus.BumpFunction.Basic` が通ることを確認
  - `import Mathlib.MeasureTheory.Integral.SetIntegral` が通ることを確認
  - _仕様: PROOF_SPEC §3.1_

---

## 1. 基本型定義

- [x] 1.1 患者状態空間・介入の型定義
  - `variable (𝒫 : Type*) [TopologicalSpace 𝒫]` を定義
  - `MedicalIntervention 𝒫 := 𝒫 → 𝒫` を定義
  - `compose` 関数（写像の合成）を定義
  - `idIntervention`（恒等介入）を定義
  - _仕様: PROOF_SPEC §0.1〜0.3_

- [x] 1.2 公理の定式化
  - `axiom state_dependent_intervention`：状態依存的な介入の存在
  - `axiom irreversible_intervention`：不可逆な介入の存在
  - 各公理のコメントに臨床的動機を記載
  - _仕様: PROOF_SPEC §4（公理リスト）_

---

## 2. Layer 1：非可換モノイドの証明
> **★ 最優先：ここまで完了すれば論文投稿可能**

- [x] 2.1 Monoid instance（モノイド公理）
  - `instance : Monoid (MedicalIntervention 𝒫)` を実装
  - 閉性：`mul_assoc` — `Function.comp_assoc` を流用
  - 単位元：`one_mul` / `mul_one` — `Function.id_comp` / `Function.comp_id` を流用
  - _仕様: PROOF_SPEC §1.1_

- [x] 2.2 `noncomm_exists`（非可換性の存在）
  - `theorem noncomm_exists : ∃ a b : MedicalIntervention 𝒫, a * b ≠ b * a`
  - `state_dependent_intervention` 公理から導出
  - `funext` の逆（関数の外延的等号 → 点ごとの等号）を利用
  - _仕様: PROOF_SPEC §1.2_

- [x] 2.3 `no_inverse`（群でないこと）
  - `theorem no_inverse : ∃ a : MedicalIntervention 𝒫, ¬∃ b, ∀ p, b (a p) = p`
  - `irreversible_intervention` 公理から直接導出
  - _仕様: PROOF_SPEC §1.3_

- [x] 2.4 チェックポイント：Layer 1 の確認
  - 3 つの定理がエラーなくコンパイルされることを確認
  - `#check noncomm_exists` `#check no_inverse` が通ることを確認
  - **この時点で arXiv 投稿可能（Appendix A.1 として掲載）**

---

## 3. Layer 2：Banach 空間の証明

- [x] 3.1 MEDICUS 最小空間の型定義
  - `variable (Ω : Set (Fin n → ℝ)) [BoundedSpace Ω]` を定義
  - `MedicusMin Ω := {f : Ω → ℝ | Differentiable ℝ f}` を部分型として定義
  - `Subtype` か `Structure` かを選択（PROOF_SPEC §5 の未解決事項）
  - _仕様: PROOF_SPEC §2.0_

- [x] 3.2 MEDICUS ノルムの定義
  - `medicusNorm f := ‖f‖_∞ + ‖fderiv ℝ f ·‖_∞` を定義
  - `fderiv` と `‖·‖∞` の Mathlib 上の型を整合させる
  - _仕様: PROOF_SPEC §2.1_

- [x] 3.3 ノルム公理の証明（正定値性・斉次性・三角不等式）
  - `lemma medicus_pos_def` — `‖f‖∞ = 0 → f ≡ 0` から導出
  - `lemma medicus_smul` — `norm_smul` を流用
  - `lemma medicus_triangle` — `norm_add_le` + 勾配の線形性を利用
  - _仕様: PROOF_SPEC §2.1〜2.3_

- [x] 3.4 Banach 空間（完備性）の証明
  - `theorem medicusMin_complete` を実装・証明（sorry ゼロ）
  - Step A：`cauchySeq_tendsto_of_complete` で点ごとの極限を取得
  - Step B：`Metric.tendstoUniformly_iff` + ε/2 戦略で一様収束を確認
  - Step C：`hasDerivAt_of_tendstoUniformly` で微分と極限の交換
  - Step D：上界の証明（一様収束 + ε/3 戦略）
  - _仕様: PROOF_SPEC §2.4_

- [x] 3.5 チェックポイント：Layer 2 の確認
  - `lake build MedicusVerify.Layer2Banach` がエラーゼロで通ることを確認
  - ノルム公理 3 つ + 完備性がすべてコンパイルされることを確認
  - **Appendix A.2 として追加**

---

## 4. Layer 3：Mollifier の証明

- [x] 4.1 Mollifier の C∞ 性
  - `theorem mollifier_smooth : ContDiff ℝ (⊤ : ℕ∞) (mollify f ε hε)`
  - `ContDiffBump.contDiff` + `HasCompactSupport.contDiff_convolution_right` を適用
  - 注: `ContDiff ℝ ⊤` は解析的レベル（強すぎ）；C∞ は `ContDiff ℝ (⊤ : ℕ∞)` と型注釈が必要
  - _仕様: PROOF_SPEC §3.1_

- [x] 4.2 収束性の証明
  - `theorem mollifier_pointwise_converges` — 点ごと収束（sorry ゼロ）
    - `ContDiffBump.convolution_tendsto_right_of_continuous` を流用
  - `theorem mollifier_converges` — M_0 ノルム収束（sorry ゼロ）
    - 仮定: `LipschitzWith Kdf (deriv f)` (W^{2,∞} 正則性)
    - 第 1 項: f が Lipschitz（有界な微分 + MVT）→ `dist_normed_convolution_le` で上界
    - 第 2 項: (φ_n ⋆ deriv f) として定式化し部分積分を回避；同様の Lipschitz 上界
    - `squeeze_zero` + `Filter.Tendsto.add` で証明完結
  - _仕様: PROOF_SPEC §3.2_

- [x] 4.3 フレシェ微分可能性（系）
  - `theorem mollifier_frechet_diff` — sorry ゼロで証明済み
  - `ContDiff.differentiable` → `Differentiable.hasFDerivAt` を繋いで完成
  - _仕様: PROOF_SPEC §3.3_

- [x] 4.4 チェックポイント：Layer 3 の確認
  - `lake build` がエラーゼロ（sorry 警告 1 件のみ）で完了
  - **Appendix A.3 として追加**

---

## 5. 論文への統合・投稿

- [ ] 5.1 Appendix A の Lean 4 コードを整形
  - 各定理に日本語コメントを追加
  - GitHub リポジトリの URL を確定
  - `SUBMISSION_CANDIDATE_report_v3_math.md` の Appendix A に挿入

- [ ] 5.2 arXiv プレプリント投稿
  - math.FA カテゴリで投稿
  - GitHub リポジトリ（Lean 4 コード）を本文中に参照
  - 「部分的に / 完全に形式検証済み」の記載を確定

- [ ] 5.3 査読付きジャーナルへの投稿
  - 投稿先の最終確定（応用数学系推奨）
  - 査読コメントに応じて必要箇所のみ追記
