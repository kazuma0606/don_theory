# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a pure academic research repository, not a software project. It contains a mathematical manuscript and supporting documents for a novel framework on optimizing non-commutative intervention sequences using functional-analytic methods (Banach-space smoothing via mollifiers).

There are no build systems, package managers, test runners, or linters. All content is Markdown with embedded LaTeX mathematics.

## Key Files

- `report/manuscript_v_20260313.md` — Primary deliverable. Complete draft of the paper including theorems, proofs, and planned experiments.
- `report/seq01.md`–`seq04.md` — Alternative paper structures and appendix materials.
- `experience/exp02.md` — Authoritative PyTorch implementation specification for all four experiments (function signatures, tensor shapes, algorithm steps).
- `experience/exp01.md` — Working notes and context from earlier research sessions.

## Mathematical Framework

The core theory models:

- **State space** `P ⊂ ℝ^d` (default d=64) — high-dimensional belief states
- **Intervention operators** `E: P → P` — non-commutative self-maps
- **Time evolution operator** `U_t: P → P` — satisfies semigroup property
- **Fundamental non-commutativity**: `U_{Δt} ∘ E_t ≠ E_{t+Δt} ∘ U_{Δt}`

The solution embeds the problem into `W^{1,∞}` (Sobolev space) and applies mollifier-based smoothing (`f_{E,t,ε} = f_{E,t} * ρ_ε`) to produce a differentiable surrogate objective `J_ε(θ)`.

Key results:
- **Theorem 1**: Uniform approximation by mollifiers
- **Theorem 2**: Existence of minimizers for smoothed objective
- **Corollary 2**: Smoothed minimizers converge to original minimizers as `ε → 0`

## Planned Experiments (Not Yet Implemented)

Four computational experiments are specified in `experience/exp02.md` (PyTorch):

1. **Smoothing Stabilization** — 64D state, 20 timesteps, Adam lr=1e-2, 20 random inits; shows smoothing prevents gradient divergence
2. **Generator Robustness** — Repeats Exp1 across 10 random dynamics generators
3. **Dimensionality Robustness** — Runs Exp1 at d ∈ {32, 64, 128}
4. **Loss Landscape Visualization** — 2D parameter grid, heatmap/3D surface comparison of smoothed vs. non-smoothed objectives

When implementing these experiments, use `experience/exp02.md` as the authoritative specification for function signatures, tensor shapes, and algorithm steps.

## Project Management

- `master_schedule.md` — 投稿先・投稿タイミング・論文間の依存関係など、全体スケジュールを管理
- `verify/tasks.md` — Lean4 形式証明タスクのみを管理
- `experience/tasks.md` — 実験実装タスクおよび論文への統合タスクを管理

## Formal Verification (verify/)

Lean 4 proofs ported from a prior related project. The math maps directly to this research:

- `verify/lean4/MedicusVerify/Layer1Monoid.lean` — Non-commutativity monoid: `noncomm_exists`, `no_inverse`
- `verify/lean4/MedicusVerify/Layer2Banach.lean` — W^{1,∞} Banach space (`MedicusMin`): norm axioms + completeness (sorry-free)
- `verify/lean4/MedicusVerify/Layer3Mollifier.lean` — Mollifier: C∞ smoothness, pointwise convergence, M₀-norm convergence (W^{2,∞} assumption on `deriv f`)
- `verify/lean4/MedicusVerify/Basic.lean` — Abstract axioms (`state_dependent`, `irreversible`)

The namespaces use `MedicalIntervention` / `MedicusMin` from the prior clinical project, but the math is identical to the current framework's `E: P → P` and `W^{1,∞}`.

**Build:** `cd verify/lean4 && lake build` (requires Lean 4 + Mathlib v4.28.0)

Remaining open task: removing the W^{2,∞} assumption in `mollifier_converges` via integration-by-parts (convolution/deriv commutativity). See `verify/tasks.md` task A1.

## Visualization (visualization/)

Python scripts (run with `uv run python`):

- `noncommutativity.py` — Visualizes `E_a ∘ E_b ≠ E_b ∘ E_a` using a 3D clinical state model (tumor/immune/tissue); generates figures 01–03
- `mollifier.py` — Mollifier kernel shape, C∞ smoothing of step functions, and `‖f_ε - f‖_∞ → 0` convergence; generates figures 04–06

Run all: `uv run python run_all.py` → outputs PNG files to `img/`.

## Paper Target

The manuscript targets top-tier ML venues (NeurIPS/ICML/ICLR). The theoretical contribution is domain-agnostic but motivated by clinical intervention sequencing.
