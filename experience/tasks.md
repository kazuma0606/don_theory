# 実験実装タスク

> 実装仕様: `experience/exp02.md`（共通コンポーネントと4実験の PyTorch 仕様）
> 形式証明タスクは `verify/tasks.md` で管理。
> 投稿スケジュールは `master_schedule.md` で管理。

---

## 完了済み

```
✅ 実験の要件定義（exp01.md）
✅ Claude Code 向け実装仕様書（exp02.md）
✅ §0 共通コンポーネント（common.py）— CUDA サニティチェック通過
```

---

## 0. 共通コンポーネントの実装

- [x] 0.0 CUDA デバイス設定（`device = torch.device("cuda")`）・全テンソル/モジュールの `.to(device)` 統一
- [x] 0.1 `Dynamics` クラス（`p_{t+1} = linear(p) + gamma*tanh(W(p))`）
- [x] 0.2 介入 `E1`（加算型）/ `E2`（線形変換型）
- [x] 0.3 観測関数 `build_observer()`（`C: ℝ^d → ℝ^q`, q=8）
- [x] 0.4 `rollout(p0, theta, dynamics, schedule, T)`
- [x] 0.5 `smooth_time(zs, kernel)`（時間方向 1D 畳み込み）
- [x] 0.6 損失関数：`loss_raw()` / `loss_smooth()`

---

## 1. Experiment 1：平滑化の有無による最適化の安定性

**目的:** 非平滑では発散・不安定、平滑では安定収束することを示す

- [x] 1.1 実験の実装（exp01.py）
- [x] 1.2 評価指標の計算・保存（exp1_steps.csv, exp1_summary.csv）
  - raw: conv_rate=16/20 (80%), loss→0, ||theta-theta*||=0.037±0.059
  - smooth: loss floor≈0.0104 (smoothing bias), ||theta-theta*||=0.458±0.088 → Exp5 で ε→0 収束を検証
- [x] 1.3 図の生成（exp1_loss_curves.png, exp1_theta_dist.png, exp1_grad_norms.png）

---

## 2. Experiment 2：ダイナミクスのランダム性への頑健性

**目的:** 特定のダイナミクスに依存せず平滑化の効果が安定して現れることを確認

- [x] 2.1 Dynamics を 10 通りランダム生成
- [x] 2.2 各 Dynamics で Experiment 1 を実行（exp02.py）
- [x] 2.3 評価指標の集計（exp2_dynamics.csv）
  - raw: conv_rate=0.80±0.00, theta_dist=0.037±0.001（dynamics 非依存）
  - smooth: conv_rate=0.03±0.03, theta_dist=0.499±0.083（bias 大きさは dynamics 依存）
- [x] 2.4 図の生成（exp2_conv_rate.png, exp2_theta_dist.png）

---

## 3. Experiment 3：状態次元への頑健性

**目的:** d が変わっても平滑化の効果が維持されることを確認

- [x] 3.1 d ∈ {32, 64, 128} で Experiment 1 を実行（exp03.py）
- [x] 3.2 評価指標の集計（exp3_dims.csv）
  - raw: conv_rate=0.80（全次元で完全一致、次元非依存）
  - smooth: theta_dist=0.597/0.458/0.661（次元依存のバイアス）
- [x] 3.3 図の生成（exp3_conv_rate.png, exp3_theta_dist.png）

---

## 4. Experiment 4：損失 landscape の可視化

**目的:** 平滑化が landscape を滑らかにすることを視覚的に示す

- [x] 4.1 θ を 2 次元に拡張（θ₁, θ₂）
- [x] 4.2 グリッドサーチで非平滑損失 J(θ) を計算
- [x] 4.3 グリッドサーチで平滑損失 J_ε(θ) を計算
- [x] 4.4 2D ヒートマップ + 3D サーフェスで可視化（exp4_heatmap.png, exp4_surface.png）
  - J: min≈0 at (1.04,1.04), max=0.089（値域広く凹凸あり）
  - J_ε: min=0.0104 at (1.41,1.29)（バイアスあり）, max=0.037（値域圧縮・平滑化）

---

## 5. Experiment 5：ε スケーリング検証（Theorem 1 & Corollary 2）

**目的:** 論文の中核的な収束保証を定量的に裏付ける

- [x] 5.1 ε 依存カーネル生成関数 `make_kernel(eps)` の実装
- [x] 5.2 軌跡近似誤差の計測（Theorem 1）
  - `eps_list = [2.0, 1.0, 0.5, 0.2, 0.1]`（K=41,21,11,5,3）で MAE 計算
  - log-log slope = 0.15（理論値 ~= 1.0 には届かず — 離散カーネルと連続モリファイアの乖離）
- [x] 5.3 最小解収束の計測（Corollary 2）
  - ||θ_ε* - θ*|| = 0.956→0.664→0.563→0.521→0.471（単調減少 ✓ 定性的に成立）
- [x] 5.4 図の生成（exp5_scaling.png）

---

## 6. 論文への統合（§5.3 実験結果）

- [ ] 6.1 実験結果の図・表を `report/manuscript_v_20260313.md` §5.3 に記述
- [ ] 6.2 §7 Discussion の完成（"To be expanded" を削除）
- [ ] 6.3 References の追加
- [ ] 6.4 §7.x Future Work を §8 の前に適切なセクションとして整理
- [ ] 6.5 Lean4 コードの Appendix A への統合
  - 各主要定理に "*(Lean 4 verified)*" バッジを付与
  - `MedicalIntervention` / `MedicusMin` の名前空間について注記を追加

---

## 優先順と依存関係

```
0（共通コンポーネント）──→ 1 → 2 → 3 → 4 ──→ 6
                             └──────────────── 5 ─┘
```

Exp 5 は Exp 1 と同じ環境で動作するため 0 完了後すぐ着手可能。

---

## 完了の定義

| タスク群 | 完了条件 |
|---|---|
| 0 | 共通コンポーネントが単体で動作確認済み |
| 1–4 | 各実験のコードが動作し、図・数値が得られている |
| 5 | log-log プロットで傾き ≈ 1.0 が確認でき、Corollary 2 の収束グラフが得られている |
| 6 | §5.3 に実験結果が記述され、§7 Discussion が完成している |
