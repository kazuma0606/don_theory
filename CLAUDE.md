# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a pure academic research repository, not a software project. It contains a mathematical manuscript and supporting documents for a novel framework on optimizing non-commutative intervention sequences using functional-analytic methods (Banach-space smoothing via mollifiers).

There are no build systems, package managers, test runners, or linters. All content is Markdown with embedded LaTeX mathematics.

## Key Files

- `report/manuscript_v_20260313.md` — Primary deliverable. Complete draft of the paper including theorems, proofs, and planned experiments.
- `report/futere_work.md` — Detailed implementation roadmap for four computational experiments (in Japanese, with PyTorch code specifications).
- `report/seq01.md`–`seq04.md` — Alternative paper structures and appendix materials.
- `experience/exp01.md`, `exp02.md` — Working notes and context from research sessions.

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

Four computational experiments are specified in `futere_work.md` (PyTorch):

1. **Smoothing Stabilization** — 64D state, 20 timesteps, Adam lr=1e-2, 20 random inits; shows smoothing prevents gradient divergence
2. **Generator Robustness** — Repeats Exp1 across 10 random dynamics generators
3. **Dimensionality Robustness** — Runs Exp1 at d ∈ {32, 64, 128}
4. **Loss Landscape Visualization** — 2D parameter grid, heatmap/3D surface comparison of smoothed vs. non-smoothed objectives

When implementing these experiments, use `futere_work.md` as the authoritative specification for function signatures, tensor shapes, and algorithm steps.

## Paper Target

The manuscript targets top-tier ML venues (NeurIPS/ICML/ICLR). The theoretical contribution is domain-agnostic but motivated by clinical intervention sequencing.
