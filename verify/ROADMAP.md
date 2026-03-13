# MEDICUS 検証ロードマップ（2026-03-10 改訂）

> **改訂の背景**
> 旧版は「非可換群」「セキュリティ×効率の不確定性」「Protobuf スキーマ」等を含んでいたが、
> 現在の理論的議論と整合していなかった。本版は `SUBMISSION_CANDIDATE_report_v3_math.md`
> の Layer 1・2 を唯一の基準として整理し直したもの。

---

## 証明が必要なもの（論文の主張に直結）

これらは Lean 4 で形式検証し、論文 Appendix A に掲載する。

---

### Layer 1：介入代数（非可換モノイド）

| # | 命題 | 内容 | 優先度 |
|---|---|---|---|
| 1.1 | **モノイド公理（閉性）** | $E_a \circ E_b \in \mathcal{E}$ | ★ 最高 |
| 1.2 | **モノイド公理（結合法則）** | $(E_a \circ E_b) \circ E_c = E_a \circ (E_b \circ E_c)$ | ★ 最高 |
| 1.3 | **モノイド公理（単位元）** | $\exists\,\mathrm{id}$ s.t. $\mathrm{id} \circ E = E \circ \mathrm{id} = E$ | ★ 最高 |
| 1.4 | **非可換性の存在** | $\exists\,E_a, E_b$ s.t. $E_a \circ E_b \neq E_b \circ E_a$ | ★ 最高 |
| 1.5 | **群でないこと（逆元の不在）** | 不可逆介入に対し $E \circ E^{-1} = \mathrm{id}$ を満たす $E^{-1}$ は存在しない | ★ 最高 |

```lean
-- Lean 4 マイルストーン（この順序で実装）
instance : Monoid MedicalIntervention := { ... }       -- 1.1〜1.3
theorem noncomm_exists :
  ∃ a b : MedicalIntervention, a * b ≠ b * a := ...   -- 1.4
theorem no_inverse :
  ∃ a : MedicalIntervention, ¬∃ b, a * b = 1 := ...   -- 1.5
```

---

### Layer 2：MEDICUS 最小空間（Banach 空間）

| # | 命題 | 内容 | 優先度 |
|---|---|---|---|
| 2.1 | **ノルム公理（正定値性）** | $\|f\|_{\mathcal{M}_0} = 0 \Rightarrow f \equiv 0$ | ★ 最高 |
| 2.2 | **ノルム公理（斉次性）** | $\|\lambda f\|_{\mathcal{M}_0} = \|\lambda\|\|f\|_{\mathcal{M}_0}$ | ★ 最高 |
| 2.3 | **ノルム公理（三角不等式）** | $\|f+g\|_{\mathcal{M}_0} \leq \|f\|_{\mathcal{M}_0} + \|g\|_{\mathcal{M}_0}$ | ★ 最高 |
| 2.4 | **完備性（Banach 空間）** | Cauchy 列が $\mathcal{M}_0$ 内に収束する | ★ 最高 |

```lean
theorem medicus_norm_is_norm : IsNorm (medicusNorm) := ...    -- 2.1〜2.3
theorem medicus_space_is_banach : IsBanachSpace MedicusMin := ... -- 2.4
```

---

### Layer 3：Mollifier と C∞ 近似

| # | 命題 | 内容 | 優先度 |
|---|---|---|---|
| 3.1 | **Mollifier の C∞ 性** | $f_\varepsilon = f * \phi_\varepsilon \in C^\infty$ | ★ 最高 |
| 3.2 | **収束性** | $\|f_\varepsilon - f\|_{\mathcal{M}_0} \to 0$（$\varepsilon \to 0$） | ★ 最高 |
| 3.3 | **フレシェ微分可能性** | $f_\varepsilon$ はフレシェ微分可能であり連鎖律が成立する | ★ 最高 |

```lean
-- Mathlib の BumpFunction 定理を活用
theorem mollifier_smooth : ∀ ε > 0, Smooth (mollify f ε) := ...  -- 3.1
theorem mollifier_converges : Tendsto (mollify f) (𝓝 0) (𝓝 f) := ... -- 3.2
```

> **注：** 3.1・3.2 は Friedrichs (1944) の既知結果であり、
> Lean 4 では Mathlib の `Mathlib.Analysis.Calculus.BumpFunction` を流用できる。
> 3.3 は $C^\infty$ から直接従う系（corollary）として処理可能。

---

## 証明が不要なもの（理由付き）

| 項目 | 不要な理由 |
|---|---|
| セキュリティ×効率の不確定性（旧 §3） | 演算子 $\hat{S}, \hat{E}$ の定義が未確定。論文スコープ外に切り離した |
| Shannon エントロピーのノルム性 | 凹関数で三角不等式を満たさない。目的関数 $J(\theta)$ に移動済み |
| ヘッセ行列の正定値性 | 最適化の収束条件だが論文の主張（空間の定義）には不要。将来課題 |
| Protobuf スキーマ整合性 | API 実装は論文化の後で検討。現段階では過剰 |
| Green の関係による同値類 | 将来の拡張（report_v4_extended.md に記載）。本論文には不要 |
| 最適輸送・変分法の弱解 | 同上。拡張論文候補 |

---

## Haskell による数値検証（補助的・Appendix 不要）

Lean 4 での形式証明の補助として、数値シミュレーションで「それらしい」ことを確認する。
論文には載せないが、開発中の健全性チェックとして使う。

| # | 内容 | スクリプト |
|---|---|---|
| H.1 | $E_\mathrm{chemo} \circ E_\mathrm{radio} \neq E_\mathrm{radio} \circ E_\mathrm{chemo}$ の数値確認 | `visualization/noncommutativity.py` で既に実施済み |
| H.2 | $\varepsilon \to 0$ での $\|f_\varepsilon - f\|_\infty$ の収束確認 | `visualization/mollifier.py` で既に実施済み |
| H.3 | ノルム三角不等式のランダムテスト | 未実装（必要になれば追加） |

---

## 実施順序

```
Step 1 ── Lean 4: 1.1〜1.5（モノイド＋逆元不在）
              ↓ ここまでで Layer 1 完了
Step 2 ── Lean 4: 2.1〜2.4（Banach 空間）
              ↓ ここまでで Layer 2 完了
Step 3 ── Lean 4: 3.1〜3.3（Mollifier、Mathlib 流用）
              ↓ Appendix A が揃う
Step 4 ── arXiv math.FA にプレプリント投稿
              ↓
Step 5 ── 査読コメントを受けて必要な箇所のみ深める
```

Step 1 の `noncomm_exists` だけでも完成していれば「部分的に形式検証済み」として投稿可能。
