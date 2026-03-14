# figures.md — 図番号とリポジトリファイルの対応表

論文に使用する図の正式番号（Fig. XX）とリポジトリ内のファイルパスの対応。
`result.md` 執筆時はこのテーブルを参照してファイル名を統一する。

---

## 本文用図（Main Figures）

| 論文番号 | ファイルパス | 内容 | 対応実験 |
|---|---|---|---|
| Fig. 01 | `visualization/img/01_state_trajectory.png` | 状態軌跡の可視化（非可換介入の例） | 導入・動機 |
| Fig. 02 | `visualization/img/02_outcome_comparison.png` | 介入順序による outcome の差（非可換性の直観） | 導入・動機 |
| Fig. 03 | `visualization/img/03_noncommutativity_heatmap.png` | ‖E_a∘E_b − E_b∘E_a‖ のヒートマップ | 非可換性の定量化 |
| Fig. 04 | `visualization/img/04_mollifier_kernel.png` | mollifier カーネル φ_ε の形状（複数 ε） | 理論（Section 3） |
| Fig. 05 | `visualization/img/05_mollifier_smoothing.png` | ステップ関数への mollifier 適用（平滑化効果） | 理論（Section 3） |
| Fig. 06 | `visualization/img/06_mollifier_convergence.png` | ‖f_ε − f‖_∞ → 0（Theorem 1 の数値確認） | Exp5 補完 |
| Fig. 07 | `experience/results/exp04/exp4_heatmap.png` | J と J_ε の loss landscape（ヒートマップ） | Exp4 |
| Fig. 08 | `experience/results/exp04/exp4_surface.png` | J と J_ε の loss landscape（3D サーフェス） | Exp4 |
| Fig. 09 | `experience/results/exp01_modify/exp1m_grad_smoothness.png` | \|Δ‖∇J‖\| 比較（勾配場の Lipschitz 性） | Exp1-modify |
| Fig. 10 | `experience/results/exp05/exp5_scaling.png` | ε → 0 での ‖θ_ε* − θ*‖ 収束（Corollary 2） | Exp5 |

---

## Appendix 用図

| 論文番号 | ファイルパス | 内容 | 対応実験 |
|---|---|---|---|
| Fig. A1 | `experience/results/exp01_modify/exp1m_loss_curves.png` | 損失曲線（raw vs smooth_sym, Adam） | Exp1-modify |
| Fig. A2 | `experience/results/exp01_modify/exp1m_theta_dist.png` | ‖θ−θ*‖ 推移（raw vs smooth_sym） | Exp1-modify |
| Fig. A3 | `experience/results/exp01_modify/exp1m_grad_norms.png` | 勾配ノルム推移（raw vs smooth_sym） | Exp1-modify |
| Fig. A4 | `experience/results/exp02/exp2_conv_rate.png` | Dynamics 依存性：収束率（10 seeds） | Exp2 |
| Fig. A5 | `experience/results/exp02/exp2_theta_dist.png` | Dynamics 依存性：‖θ_ε*‖ のばらつき | Exp2 |
| Fig. A6 | `experience/results/exp03/exp3_conv_rate.png` | 次元依存性：収束率（d = 32/64/128） | Exp3 |
| Fig. A7 | `experience/results/exp03/exp3_theta_dist.png` | 次元依存性：‖θ_ε*‖ の変化 | Exp3 |
| Fig. A8 | `experience/results/exp_extra_01/extra1_convergence_speed.png` | 収束速度分布（4 条件、boxplot） | Extra Exp1 |
| Fig. A9 | `experience/results/exp_extra_01/extra1_loss_curves.png` | 損失曲線（4 条件、Adam vs L-BFGS） | Extra Exp1 |
| Fig. A10 | `experience/results/exp_extra_01/extra1_grad_norms.png` | 勾配ノルム（4 条件比較） | Extra Exp1 |

---

## 不使用（参考保管）

| ファイルパス | 内容 | 不使用理由 |
|---|---|---|
| `experience/results/exp01/exp1_loss_curves.png` | 損失曲線（raw vs smooth, 片側平滑化） | exp01_modify に置換済み |
| `experience/results/exp01/exp1_theta_dist.png` | ‖θ−θ*‖（raw vs smooth, 片側平滑化） | 同上 |
| `experience/results/exp01/exp1_grad_norms.png` | 勾配ノルム（raw vs smooth, 片側平滑化） | 同上 |
| `experience/results/exp_extra_01/extra1_theta_dist.png` | ‖θ−θ*‖ 推移（4 条件） | A8/A9 で代替 |

---

## 採番方針

- **Fig. 01–06**: 理論的動機・数学的構造（visualization/ スクリプト生成）
- **Fig. 07–10**: 主要実験結果（本文 Results/Discussion に直接引用）
- **Fig. A1–A10**: 追加詳細（Appendix）
- exp01（片側平滑化）の図は exp01_modify に置換されたため本文・Appendix ともに不使用
