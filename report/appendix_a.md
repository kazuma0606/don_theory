# appendix_a.md — Appendix A: Proofs and Formal Verification 草稿

---

## A.5 Lean 4 Proof Summary

All core theoretical claims in this paper have been formally verified
using Lean 4 with Mathlib. The complete proof development is publicly available at:

> **https://github.com/kazuma0606/don_theory**
> Directory: `verify/lean4/MedicusVerify/`

### Build instructions

```bash
cd verify/lean4
lake build MedicusVerify
```

Verified with:
- Lean 4 `v4.28.0` (`lean-toolchain`: `leanprover/lean4:v4.28.0`)
- Mathlib `v4.28.0`

All proofs compile with **zero `sorry`** and **zero warnings**.

### Proof file index

| ファイル | 内容 | 対応する定理 |
|---|---|---|
| `Basic.lean` | 抽象公理（`state_dependent`, `irreversible`） | §2 前提 |
| `Layer1Monoid.lean` | 非可換モノイド（`noncomm_exists`, `no_inverse`） | §2.2–2.3 |
| `Layer2Banach.lean` | $W^{1,\infty}$ Banach 空間（ノルム公理・完備性） | §3.1, Lemma 1 |
| `Layer3Mollifier.lean` | Mollifier の $C^\infty$ 性・Fréchet 微分可能性・収束 | §3.2, Theorem 1, Corollary 1 |
| `Layer4Regularity.lean` | Lipschitz 合成補題・可観測量の $W^{1,\infty}$ 所属・最小解の存在と収束 | Appendix A.0–A.1, Theorem 2, Corollary 2 |

### Note on W^{2,∞} assumption

`Layer3Mollifier.lean` の `mollifier_converges` は `hdf_lip`（$\|f'\|_{L^\infty} < \infty$、
すなわち $f \in W^{2,\infty}$）を仮定として明示的に受け取る。
この仮定は数学的に本質的であり（IBP ステップに $(\text{deriv}\, f)$ の連続性が必要）、
論文 §3.2 Corollary 1 に明記している。
$W^{1,\infty}$ のみからは導けないことも `verify/tasks.md` task A1' に記録済み。
