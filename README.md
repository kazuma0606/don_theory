# Differentiable Optimization of Non-Commutative Intervention Sequences via Banach-Space Smoothing

研究リポジトリ。時間発展する系に対する非可換介入列のパラメータを、Banach 空間（W^{1,∞}）埋め込みとモリファイア平滑化によって微分可能最適化する枠組みの理論・実験・形式証明を管理する。

## 論文

`report/manuscript_v_20260313.md`

**主な貢献：**
- 非可換介入演算子の関数解析的モデル化
- W^{1,∞} への埋め込みによる一様制御
- モリファイア平滑化による微分可能近似（Theorem 1: ‖f_ε − f‖ ≤ Cε、Corollary 2: ε→0 での最小解収束）
- 高次元合成環境での実証実験（PyTorch）

投稿先候補・スケジュール → `master_schedule.md`

## リポジトリ構成

```
report/          論文原稿・付録素材
experience/      実験仕様（exp02.md）・実装タスク（tasks.md）
verify/lean4/    Lean 4 形式証明（Layer 1–3 は sorry ゼロ完了）
visualization/   可視化スクリプト（Python）
master_schedule.md  投稿スケジュール・論文間依存関係
CLAUDE.md        Claude Code 向けガイド
```

## 形式証明（Lean 4）

```bash
cd verify/lean4
lake build
```

Mathlib v4.28.0 が必要（`lean-toolchain` および `lake-manifest.json` で固定）。

| 層 | 内容 | 状態 |
|---|---|---|
| Layer 1 | 非可換モノイド（`noncomm_exists`, `no_inverse`） | ✅ sorry ゼロ |
| Layer 2 | W^{1,∞} Banach 空間（`MedicusMin`、ノルム公理・完備性） | ✅ sorry ゼロ |
| Layer 3 | モリファイア（C∞ 平滑性・収束）| ✅ sorry ゼロ（W^{2,∞} 仮定明示）|
| Layer 4 | Lipschitz 合成・最小解存在・収束（Appendix A） | 🔲 未着手 |

残タスク → `verify/tasks.md`

## 実験（PyTorch / CUDA）

CUDA 前提。仕様書 → `experience/exp02.md`

| # | 実験 | 主な検証対象 |
|---|---|---|
| 1 | 平滑化の最適化安定性 | 非平滑 vs 平滑の収束率・勾配安定性 |
| 2 | ダイナミクスへの頑健性 | 10 通りのランダムダイナミクス |
| 3 | 状態次元への頑健性 | d ∈ {32, 64, 128} |
| 4 | 損失 landscape 可視化 | 2D ヒートマップ（J vs J_ε） |
| 5 | ε スケーリング検証 | Theorem 1 O(ε) 則・Corollary 2 最小解収束 |

実装タスク → `experience/tasks.md`

## 可視化

```bash
cd visualization
uv run python run_all.py   # img/ に PNG を出力
```
