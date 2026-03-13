# Lean 4 形式証明タスク一覧

> **方針**
> - Layer 1–3 形式証明はすべて sorry ゼロで完了（旧タスクは `_tasks.md` に凍結済み）
> - 本ファイルは **Lean 4 の証明作業のみ** を管理
> - 実験実装・論文記述タスクは `experience/tasks.md` で管理
> - 投稿スケジュールは `master_schedule.md` で管理
>
> B/D シリーズは旧 MEDICUS 論文（`SUBMISSION_CANDIDATE_report_v3_math.md`）向け。
> 本研究（`report/manuscript_v_20260313.md`）向けは A2 シリーズ。

---

## 完了済み（参照用）

```
✅ Layer 1: 非可換モノイド証明（noncomm_exists, no_inverse）
✅ Layer 2: Banach 空間証明（ノルム公理 + 完備性、sorry ゼロ）
✅ Layer 3: Mollifier C∞性・Fréchet 微分可能性・収束（sorry ゼロ）
⚠️  mollifier_converges: W^{2,∞} 仮定を明示（→ A1 で解消 or A1' で論文に明記）
```

---

## A1. 部分積分（convolution と deriv の交換）

> **目的:** `mollifier_converges` の W^{2,∞} 仮定を除去し、M₀ = W^{1,∞} のまま収束を証明する。

- [ ] A1.1 `HasDerivAt_convolution_right_integral` — 積分記号下微分
  - `hasFDerivAt_integral_of_dominated_loc_of_lip` を適用
  - 前提: `φ_ε` コンパクト台 + `f` 微分可能 + domination 条件
  - ファイル: `MedicusVerify/Layer3Mollifier.lean`

- [ ] A1.2 `convolution_deriv_comm` — 交換公式の定理化
  ```lean
  theorem convolution_deriv_comm (f : MedicusMin) (φ : ContDiffBump (0 : ℝ)) (x : ℝ) :
      HasDerivAt (f.val ⋆[lsmul ℝ ℝ, volume] ⇑φ)
                 ((fun y => deriv f.val y) ⋆[lsmul ℝ ℝ, volume] ⇑φ) x x
  ```
  - A1.1 の系として導出

- [ ] A1.3 `mollifier_converges` の改訂
  - `Kdf` と `hdf_lip` 仮定を削除
  - 第 2 項を `(φ_n ⋆ deriv f)` → `deriv (φ_n ⋆ f)` に戻す
  - A1.2 を内部で使って証明

  > **代替案 A1':** W^{2,∞} 仮定を明示したまま論文 Corollary 1 に追記し、Lean コードはそのまま維持。

---

## A2. 現論文 Appendix A の形式証明（`manuscript_v_20260313.md`）

> **対象:** Appendix A.0〜A.3 および §4 の定理・系。
> ファイル: 新規 `MedicusVerify/Layer4Regularity.lean`

- [ ] A2.1 **Lemma A.0** — Lipschitz 合成安定性 (`lipschitz_comp_finite`)
  - 命題: $h \circ g$ の Lipschitz 定数 ≤ $L_h \cdot L_g$、有限合成へ帰納的に拡張
  - Mathlib: `LipschitzWith.comp` を活用
  - 論文への対応: Appendix A.0

- [ ] A2.2 **Lemma A.1** — 観測関数の W^{1,∞} 正則性 (`observable_in_W1inf`)
  - 命題: $E_{t_i}$, $U_{t_j}$, $O_t$ が Lipschitz かつ $P$ が有界ならば $f_{E,t} \in W^{1,\infty}$
  - 証明戦略: A2.1 で合成の Lipschitz 性 → 有界性 → Rademacher
  - 論文への対応: Appendix A.1

- [ ] A2.3 **Theorem 2** — 平滑化目的関数の最小解の存在 (`J_eps_minimizer_exists`)
  - 命題: $\Theta$ コンパクト ⟹ $J_\varepsilon$ は最小解をもつ
  - 証明戦略: $J_\varepsilon$ の連続性 + Weierstrass（Mathlib: `IsCompact.exists_isMinOn`）
  - 論文への対応: §4 Theorem 2, Appendix A.3

- [ ] A2.4 **Corollary 2** — 平滑化最小解の収束 (`minimizers_converge`)
  - 命題: $\sup_\theta |J_\varepsilon(\theta) - J(\theta)| \le C'\varepsilon$ から $\varepsilon \to 0$ での最小解の収束
  - 論文への対応: §4 Corollary 2, Appendix A.3

---

## B. 旧 MEDICUS 論文向け証明・記述タスク

### B1–B3. 論文への Lean4 verified バッジ付与

- [ ] B1.1 Layer 1 の各定理に "*(Lean 4 verified)*" を追記
- [ ] B2.1 Layer 2（Banach 空間）に "*(Lean 4 verified)*" を追記
- [ ] B2.2 §3.3（拡張ノルム問題）を削除
- [ ] B3.1 Layer 3（Mollifier）に "*(Lean 4 verified)*" を追記（A1/A1' に依存）
- [ ] B3.2 §4.2 に Lean コード断片を掲載

### B4–B5. 論文構成の整理

- [ ] B4.1〜B4.5 §6 を削除し論文を完結した構成に
- [ ] B5.1〜B5.3 数値例セクションを追加（Haskell による ‖A∘B(p) - B∘A(p)‖ の計算）

### B6. Appendix A（Lean4 コード）の追加

- [ ] B6.1 Appendix A（Layer 1–3 のコア部分）を論文末尾に追加
- [ ] B6.2 `verify/lean4/` を公開 GitHub リポジトリに push
- [ ] B6.3 `lean-toolchain` バージョンを論文に記載・`lake-manifest.json` を commit

---

## D. 第二論文準備（不確定性原理）

- [ ] D1. 交換子の Lean 4 定義（型のみ）
  ```lean
  def commutator (A B : MedicalIntervention P) : P → P :=
    fun p => A (B p) - B (A p)
  ```
  ファイル: `MedicusVerify/SecondPaper/Commutator.lean`（新規）

- [ ] D2. Robertson 不等式との対応関係メモ
  ファイル: `discussion/SECOND_PAPER_UNCERTAINTY.md`（新規）

---

## 優先順と依存関係

```
【本研究 manuscript_v_20260313.md】
A1 or A1' ──→ B3.1（Corollary 1 verified バッジ）
A2.1 → A2.2 → A2.3 → A2.4

【旧 MEDICUS SUBMISSION_CANDIDATE_report_v3_math.md】
B1.1, B2.1 ──→ B6.1 → B6.2
B5.1〜B5.3 ──→ B4.3 → B4.5

D1, D2 ─── 独立（いつでも並行可）
```

---

## 完了の定義

| タスク群 | 完了条件 |
|---|---|
| A1 | `lake build MedicusVerify` が sorry ゼロで通る（W^{2,∞} 仮定除去） |
| A2 | `Layer4Regularity.lean` が sorry ゼロでビルドを通る |
| B | 旧 MEDICUS 論文の §6 が削除され、全主要定理に "(Lean 4 verified)" がある |
| B6 | GitHub に push 済み、Appendix A が論文に存在 |
| D | `SECOND_PAPER_UNCERTAINTY.md` が存在し、Lean 4 ファイルがビルドを通る |
