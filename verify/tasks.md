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

## A1. ✅ A1' 選択済み（W^{2,∞} 仮定を明示して維持）

> **調査結果（2026-03-13）:** A1（W^{2,∞} 除去）は数学的に不可能。
> M₀ ノルム収束の微分項 `sSup |φ_n ⋆ (deriv f) - deriv f| → 0` は
> `deriv f` が Lipschitz（= W^{2,∞}）でなければ一様収束しない。
> IBP ステップ `(f ⋆ deriv φ)(x) = ((deriv f) ⋆ φ)(x)` に必要な `deriv f` の
> 連続性は W^{1,∞} から導けない。
>
> Mathlib には `HasCompactSupport.hasDerivAt_convolution_right` が存在し、
> `HasDerivAt (f ⋆ φ) ((f ⋆ deriv φ)(x)) x` は証明できる（A1.1 相当）。
> しかし A1.2 の IBP ステップには W^{2,∞} が本質的に必要。
>
> **A1' 採用**: W^{2,∞} 仮定は論文 §3.2 Corollary 1 に明示済み。Lean コードはそのまま維持。

**対応済み:** `mollifier_converges` の `hdf_lip` 仮定は正当であり、sorry ゼロのまま。

---

## A2. ✅ 完了（2026-03-13）— Appendix A の形式証明

> **ファイル:** `MedicusVerify/Layer4Regularity.lean`（新規作成）
> **ビルド:** `lake build MedicusVerify` — sorry ゼロ・warning ゼロで通過

- [x] A2.1 **Lemma A.0** — `lipschitz_comp_of_lipschitz` / `lipschitz_comp_cons`
  - `LipschitzWith (Kh * Kg) (h ∘ g)` を `LipschitzWith.comp` の直接系として証明
  - 論文への対応: Appendix A.0

- [x] A2.2 **Lemma A.1** — `observable_in_W1inf`
  - 命題: Differentiable + LipschitzWith K + BddAbove |f| ⟹ f ∈ MedicusMin
  - `norm_deriv_le_of_lipschitz` で |deriv f x| ≤ K を証明し、MedicusMin の条件を構成
  - 注: 論文の Rademacher 経由の議論はここでは可微分性を仮定として直接受け取る
  - 論文への対応: Appendix A.1

- [x] A2.3 **Theorem 2** — `J_eps_minimizer_exists`
  - 命題: IsCompact S → Nonempty S → ContinuousOn J_ε S → ∃ 最小解
  - `IsCompact.exists_isMinOn`（Weierstrass の定理）の直接適用
  - 論文への対応: §4 Theorem 2, Appendix A.3

- [x] A2.4 **Corollary 2** — `minimizers_approximate` / `minimum_values_converge`
  - 命題: sup |J_ε - J| ≤ C'ε ⟹ J(θ_ε) ≤ J(θ*) + 2C'ε
  - `minimum_values_converge` でさらに |J(θ_ε) - J(θ*)| ≤ 2C'ε を証明
  - 論文への対応: §4 Corollary 2, Appendix A.3

---

## B. 旧 MEDICUS 論文向け証明・記述タスク

### B1–B3. 論文への Lean4 verified バッジ付与

- [x] B1.1 Layer 1 の各定理に "*(Lean 4 verified)*" を追記
  - 定理 M（非可換モノイド）← Monoid instance + `noncomm_exists`
  - 命題 1（群ではないこと）← `no_inverse`
- [x] B2.1 Layer 2（Banach 空間）に "*(Lean 4 verified)*" を追記
  - 補題 1（ノルム公理）← `medicusNorm_pos_def` / `medicusNorm_smul` / `medicusNorm_triangle`
  - 定理 1（Banach 空間）← `medicusMin_complete`
- [x] B2.2 §3.3（拡張ノルム問題）を削除 — Shannon エントロピー項の未解決問題ごと削除
- [ ] B3.1 Layer 3（Mollifier）に "*(Lean 4 verified)*" を追記（A1/A1' に依存）
- [ ] B3.2 §4.2 に Lean コード断片を掲載

### B4–B5. 論文構成の整理

- [ ] B4.1〜B4.5 §6 を削除し論文を完結した構成に
- [ ] B5.1〜B5.3 数値例セクションを追加（Haskell による ‖A∘B(p) - B∘A(p)‖ の計算）

### B6. Appendix A（Lean4 コード）の追加

- [ ] B6.1 Appendix A（Layer 1–3 のコア部分）を論文末尾に追加
- [x] B6.2 `verify/lean4/` を公開 GitHub リポジトリに push
  - URL: https://github.com/kazuma0606/don_theory.git
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

> **現在の方針（2026-03-13）:** P1（本研究論文）を優先。
> P1 の arXiv 投稿完了後に P2（B シリーズ）・P3（D シリーズ）を再開する。
> P1 の残タスクは `experience/tasks.md` §0〜6（実験実装・論文統合）のみ。

```
【P1: 本研究 manuscript_v_20260313.md】← 現在優先
✅ A1' 選択済み
✅ A2.1 → A2.2 → A2.3 → A2.4 完了
→ experience/tasks.md §0〜6（実験実装）が残課題

【P2: 旧 MEDICUS — P1 完了後に再開】
✅ B1.1, B2.1, B2.2 完了
B3.1, B3.2 → B4 → B5 → B6.1 → B6.2

【P3: 第二論文 — P1 or P2 完了後】
D1, D2 ─── 独立
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
