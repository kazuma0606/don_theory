# Lean 4 形式証明タスク一覧

> **方針（2026-03-13 改訂）**
> - Layer 1–3 形式証明はすべて sorry ゼロで完了（旧タスクは `_tasks.md` に凍結済み）
> - 本ファイルは **Lean 4 の証明作業** と **論文本文の技術的記述** のみを管理
> - 投稿スケジュール・提出先・タイミングは `../master_schedule.md` で管理
>
> B/C/D シリーズは旧 MEDICUS 論文（`SUBMISSION_CANDIDATE_report_v3_math.md`）向け。
> 本研究（`report/manuscript_v_20260313.md`）向けは A2・E シリーズ。

---

## 完了済み（参照用）

```
✅ Layer 1: 非可換モノイド証明（noncomm_exists, no_inverse）
✅ Layer 2: Banach 空間証明（ノルム公理 + 完備性、sorry ゼロ）
✅ Layer 3: Mollifier C∞性・Fréchet 微分可能性・収束（sorry ゼロ）
⚠️  mollifier_converges: W^{2,∞} 仮定を明示（→ A1 で解消 or A1' で論文に明記）
```

---

## A. Lean 4：残証明タスク

### A1. 部分積分（convolution と deriv の交換）の形式証明

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

  > A1 が難しい場合の代替案（A1': 仮定の明示化）
  > - W^{2,∞} 仮定を明示したまま論文 Corollary 1 に「f ∈ W^{2,∞}」と追記
  > - Lean 4 コードはそのまま。論文で「この条件は臨床的パラメータで自然に成立」と動機づけ

---

### A2. 現論文 Appendix A の形式証明（`manuscript_v_20260313.md`）

> **対象命題:** Appendix A.0〜A.3 および §4 の定理・系。
> Layer 1–3 は完了済みだが、観測関数の正則性と最適化理論が未形式化。
> ファイル: 新規 `MedicusVerify/Layer4Regularity.lean`

- [ ] A2.1 **Lemma A.0** — Lipschitz 合成安定性 (`lipschitz_comp_finite`)
  - 命題: $g$ が $L_g$-Lipschitz、$h$ が $L_h$-Lipschitz ならば $h \circ g$ は $L_h L_g$-Lipschitz。有限合成へ帰納的に拡張。
  - Mathlib: `LipschitzWith.comp` が存在するため、有限列への帰納部分のみ実装
  - 論文への対応: Appendix A.0

- [ ] A2.2 **Lemma A.1** — 観測関数の W^{1,∞} 正則性 (`observable_in_W1inf`)
  - 命題: $E_{t_i}$, $U_{t_j}$, $O_t$ が Lipschitz かつ $P$ が有界ならば
    $f_{E,t} = O_t \circ U_{t-t_k} \circ E_{t_k} \circ \cdots \circ E_{t_1} \in W^{1,\infty}$
  - 証明戦略:
    1. A2.1 を繰り返し適用して合成が Lipschitz であることを示す
    2. $P$ 有界 + 各演算子が有界集合を保存 → $f_{E,t}$ の有界性
    3. Lipschitz + 有界 → W^{1,∞}（Rademacher: `LipschitzWith.hasDerivAt` 等）
  - 論文への対応: Appendix A.1（Layer 2 の `MedicusMin` 型の正当性の根拠）

- [ ] A2.3 **Theorem 2** — 平滑化目的関数の最小解の存在 (`J_eps_minimizer_exists`)
  - 命題: $\Theta$ がコンパクト、$J_\varepsilon : \Theta \to \mathbb{R}$ が連続ならば最小解が存在する
  - 証明戦略:
    1. $\theta \mapsto f_{E_\theta}$ の $L^\infty$ 連続性（Assumption 2–4 から）
    2. 畳み込みは $L^\infty$ 上の連続線形作用素 → $J_\varepsilon$ が連続
    3. Weierstrass 定理（Mathlib: `IsCompact.exists_isMinOn`）で最小解の存在
  - 論文への対応: §4 Theorem 2, Appendix A.3 item 1–2

- [ ] A2.4 **Corollary 2** — 平滑化最小解の収束 (`minimizers_converge`)
  - 命題: $\sup_{\theta \in \Theta} |J_\varepsilon(\theta) - J(\theta)| \le C'\varepsilon$（Lemma A.3）から
    $\varepsilon \to 0$ での最小解の収束を導く
  - 証明戦略: A2.3 の一様近似 + $\Theta$ のコンパクト性 + $\varepsilon \to 0$ の極限
  - 論文への対応: §4 Corollary 2, Appendix A.3 item 3

---

## B. 旧 MEDICUS 論文（`SUBMISSION_CANDIDATE_report_v3_math.md`）の改訂タスク

### B1. §2（介入代数）の更新

- [ ] B1.1 Lean 4 検証済みバッジを定理に付与
  - 定理 (Monoid Instance), 定理 (noncomm_exists), 定理 (no_inverse) に
    "*(Lean 4 verified)*" を追記
  - 公理 `state_dependent_intervention`・`irreversible_intervention` の
    数学的動機（臨床的根拠）を1段落追加

- [ ] B1.2 §2.2 の非可換性定理を明確化
  - 現行の証明スケッチを、Lean 4 コードへの参照（Appendix A）に置き換え

### B2. §3（MEDICUS 関数空間）の更新

- [ ] B2.1 定理 1（Banach 空間）に "*(Lean 4 verified)*" を追記
  - 完備性の証明概略（Cauchy 列 → 一様収束 → 微分交換）を整理

- [ ] B2.2 §3.3（拡張ノルムの問題）を削除または縮小
  - エントロピーを目的関数に移した設計判断を1文で説明し、§3.3 は削除

### B3. §4（Mollifier）の更新

- [ ] B3.1 定理 2（C∞性・収束）に "*(Lean 4 verified)*" を追記
  - A1 完了時: W^{1,∞} のまま収束定理を記述
  - A1' の場合: 「f ∈ W^{2,∞} のとき」と仮定を明示。臨床的動機を付記

- [ ] B3.2 §4.2 の「MEDICUS 空間への適用」を Lean 4 実装と整合させる
  - `mollify` 定義・`mollifier_smooth`・`mollifier_frechet_diff` の Lean コード断片を掲載

### B4. §6（今後の課題）の廃止・再構成

- [ ] B4.1 §6.1（Lean 4 形式検証）→ 削除（完了済みのため）
- [ ] B4.2 §6.2（拡張ノルムの問題）→ 削除（B2.2 で解決）
- [ ] B4.3 §6.3（非可換性の定量的評価）→ B5 の数値例セクションに移動
- [ ] B4.4 §6.4（不確定性原理）→ 削除（第二論文として切り離し）
- [ ] B4.5 §6 全体を削除し、§5 を最終セクションとして論文を締める

### B5. 数値例セクションの追加（§5 更新または新 §5.3）

- [ ] B5.1 Haskell による数値実証の記述
  - 具体例: 「化療(A)→手術(B)」 vs 「手術(B)→化療(A)」の患者状態差 ‖A∘B(p) - B∘A(p)‖

- [ ] B5.2 数値の計算・表示
  - Haskell スクリプトで具体的なパラメータと数値を出力し、論文に表または図として掲載

- [ ] B5.3 数値例が "公理の動機" を裏付けることを文章で説明

### B6. Appendix A（Lean 4 コード）の追加

- [ ] B6.1 Appendix A の節を論文末尾に追加
  - A.1: Layer 1, A.2: Layer 2, A.3: Layer 3

- [ ] B6.2 GitHub リポジトリの準備
  - `verify/lean4/` を公開リポジトリに push
  - README に `lake build` 手順を追記

- [ ] B6.3 Mathlib バージョンの固定（`lean-toolchain` のバージョンを論文に記載）

---

## D. 第二論文準備（不確定性原理）——最小限の準備のみ

- [ ] D1. 交換子の Lean 4 定義（型のみ）
  ```lean
  def commutator (A B : MedicalIntervention P) : P → P :=
    fun p => A (B p) - B (A p)
  ```
  ファイル: `MedicusVerify/SecondPaper/Commutator.lean`（新規）

- [ ] D2. 不確定性原理の数学的定式化メモ
  - Robertson 不等式との対応関係
  - ファイル: `discussion/SECOND_PAPER_UNCERTAINTY.md`（新規）

---

## E. 本研究論文（`manuscript_v_20260313.md`）の技術的記述タスク

### E1. Lean4 形式証明の論文への統合

- [ ] E1.1 Appendix A に Lean4 コードを追記
  - A.1（Layer 1）— `noncomm_exists`, `no_inverse` のコア
  - A.2（Layer 2）— `medicusNorm` の公理 + `medicusMin_complete`
  - A.3（Layer 3）— `mollifier_smooth`, `mollifier_converges`
  - A.4（Layer 4）— A2.1〜A2.4 完了後に追加
  - 各定理に "*(Lean 4 verified)*" バッジを付与

- [ ] E1.2 Layer 1–3 の名前空間について注記を追加
  - `MedicalIntervention` / `MedicusMin` は旧プロジェクト由来の名称だが、
    本論文の一般的な介入演算子 $E: P \to P$ および $W^{1,\infty}$ 空間に対応することを明記

### E2. 論文本体の残タスク

- [ ] E2.1 §5.3 実験結果の実装と記述（`futere_work.md` の実験仕様に従う）
  - Experiment 1: 平滑化による勾配安定化
  - Experiment 2: ダイナミクス生成器の違いへの頑健性
  - Experiment 3: 次元数 d ∈ {32, 64, 128} での頑健性
  - Experiment 4: 損失景観の可視化

- [ ] E2.2 §7 Discussion の完成（"To be expanded" を削除）

- [ ] E2.3 References の追加

- [ ] E2.4 §7.x Future Work を §8 の前に適切なセクションとして整理

---

## 優先順と依存関係

```
【本研究 manuscript_v_20260313.md】
A1 or A1' ──→ E1.1（Corollary 1 verified バッジ）
A2.1 → A2.2 → A2.3 → A2.4 ──→ E1.1（Appendix A.4）
E2.1（実験）──→ E2.2（Discussion）──→ E2.3（References）

【旧 MEDICUS SUBMISSION_CANDIDATE_report_v3_math.md】
B1.1, B2.1 ──→ B6.1 → B6.2
B5.1, B5.2 ──→ B4.3 → B4.5
D1, D2 ─── 独立（いつでも並行可）
```

---

## 完了の定義

| タスク群 | 完了条件 |
|---|---|
| A1 | `lake build MedicusVerify` が sorry ゼロで通る（W^{2,∞} 仮定除去） |
| A2 | `Layer4Regularity.lean` が sorry ゼロでビルドを通る |
| E1 | `manuscript_v_20260313.md` の全主要定理に "(Lean 4 verified)" がある |
| E2 | §5.3 に実験結果が記述され、§7 Discussion が完成している |
| B | 旧 MEDICUS 論文の §6 が削除され、全主要定理に "(Lean 4 verified)" がある |
| B5 | 数値が出力され、論文に表/図として掲載 |
| B6 | GitHub に push 済み、Appendix A が論文に存在 |
| D | `SECOND_PAPER_UNCERTAINTY.md` が存在し、Lean 4 ファイルがビルドを通る |
