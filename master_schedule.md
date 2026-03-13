# Master Schedule

全論文・リポジトリの投稿・公開スケジュールを管理するファイル。
個別の証明作業・論文改訂タスクは `verify/tasks.md` で管理。

---

## 論文一覧

| ID | 論文 | ファイル | ステータス |
|---|---|---|---|
| **P1** | Differentiable Optimization of Non-Commutative Intervention Sequences via Banach-Space Smoothing | `report/manuscript_v_20260313.md` | 執筆中（実験未実装） |
| **P2** | MEDICUS: A Formally Verified Non-Commutative Monoid Framework for Medical Intervention Optimization | `SUBMISSION_CANDIDATE_report_v3_math.md`（旧リポジトリ） | 形式証明完了・論文改訂中 |
| **P3** | 不確定性原理論文（仮題） | 未着手 | 準備段階（`verify/tasks.md` D シリーズ） |

---

## P1 — 本研究論文

**タイトル:** Differentiable Optimization of Non-Commutative Intervention Sequences via Banach-Space Smoothing

**投稿先候補:**
- 第一候補: arXiv `math.FA` + クロスリスト `cs.LG`
- 会議: NeurIPS / ICML / ICLR（実験結果の完成度次第）

**投稿前の必須条件:**
1. `verify/tasks.md` A2（Layer4 正則性・最適化理論）完了 or A1'（W^{2,∞} 仮定明示）選択
2. `verify/tasks.md` E2.1 — 実験 4 本の実装と結果記述
3. `verify/tasks.md` E2.2〜E2.4 — Discussion・References 完成
4. `verify/tasks.md` E1.1 — Appendix A に Lean4 コード統合

**投稿可能な最短ルート（A1' 選択時）:**
```
A1' 選択 → A2.1〜A2.4 → E2.1（実験）→ E2.2〜E2.4 → E1.1 → arXiv 投稿
```

---

## P2 — 旧 MEDICUS 論文

**タイトル候補:** MEDICUS: A Formally Verified Non-Commutative Monoid Framework for Medical Intervention Optimization

**投稿先候補:**
- arXiv: `math.FA` + クロスリスト `math.LO`
- ジャーナル: 応用数学系（査読付き）

**投稿前の必須条件（`verify/tasks.md` B シリーズ）:**
1. B1〜B4: 論文本文の改訂完了（§6 削除・Lean4 verified バッジ付与）
2. B5: Haskell 数値例の掲載
3. B6: GitHub リポジトリ公開・Appendix A 追加

**P1 との依存関係:** P1 の arXiv 投稿後、または並行して進めることができる。
形式証明コード（`verify/lean4/`）は P1・P2 で共有。

---

## P3 — 第二論文（不確定性原理）

**状態:** 最小限の準備のみ（`verify/tasks.md` D シリーズ）

**着手条件:** P1 または P2 の arXiv 投稿完了後

---

## GitHub 公開

**対象:** `verify/lean4/`（Lean4 形式証明コード）

**公開タイミング:** P1 または P2 の arXiv 投稿と同時

**必要作業:**
- `verify/lean4/README.md` に `lake build` 手順を整備（既存 README を更新）
- `lake-manifest.json` を commit して Mathlib バージョンを固定
- 論文本文中に GitHub URL を記載
