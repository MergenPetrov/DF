# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A reference / review workspace for validating a C++ implementation of the **Drift Flux slip model** used in wellbore multiphase flow solvers. It is **not** a buildable project — `my_code_12*.cpp|txt` is an out-of-context fragment extracted from a larger proprietary wellbore simulator. Symbols like `wsncs`, `fully_implicit_element_status`, `model_parameters`, `tnm::min_compare`, `tnav_pow`, `run_flash`, `pipe_gas_liq_interfacial_tension_holdup_weightening`, etc. are defined in the parent project — don't try to compile in isolation.

## Files

- `EclipseTechnicalDescription.pdf` / `Eclipse.txt` — ECLIPSE manual; the drift-flux section is **Eq. 8.66–8.91** (search `"8.66"` or `"drift flux slip model"` in `Eclipse.txt`).
- `shi2005 (1).pdf` / `shi2005.txt` — SPE paper (Shi, Holmes et al.). Eq. 14–15 disambiguate the sqrt position in the drift-velocity denominator.
- `Chen01.pdf` / `Chen01.txt` — Stanford PhD thesis (background theory).
- `my_code_12.txt` — user's C++ implementation (1472 lines). First ~673 lines are a finite-difference Jacobian wrapper; `test_function` body (starting ~line 675) is the actual drift-flux math.
- `my_code_12_fixed.cpp` — corrected version, 1519 lines. Each fix carries a `// FIX #N` comment matching numbering in `df_code_review.md`.
- `df_code_review.md` — full review (in Russian) listing 12 findings.
- The `.txt` companions to PDFs are `pdftotext` extractions; regenerate with `pdftotext file.pdf file.txt` (requires `poppler-utils`).

## Four critical bugs already identified (in `my_code_12.txt`)

When asked to touch the code, check these first — they're the most likely source of regressions vs. Eclipse:

1. **β\* clamp** (line 1176): `fabs(β-B)/(1-B)` must be clamped to `[0,1]`, not taken by absolute value.
2. **sqrt in drift velocity** (lines 1195–1198): the sqrt must cover the **entire** denominator `αg·C0·ρg/ρl + 1 − αg·C0`.
3. **Iteration structure** (lines 1095–1412): `vsg`, `vso`, `vsw` must be computed once from `reconstruct_phase_rates_df` before the loop and held constant. The loop should iterate only on holdups `αg`, `βo`, not on all four `prev_vels` × `prev_holdups` simultaneously.
4. **σow formula** (lines 1326–1328): Eclipse specifies `σow = |σgo − σwg|` — **no** holdup weighting.

All four are already applied in `my_code_12_fixed.cpp`.

## Working conventions

- **Primary language with user: Russian.** All reviews, commit messages, and explanations in the existing artifacts are in Russian; stay consistent.
- **Development branch: `claude/complex-calculation-lFj0L`.** Don't commit to `main`. Create the branch from main if missing.
- **Don't push to remote without explicit request.** Git push in this environment returns `403 Permission to MergenPetrov/DF.git denied`; MCP `create_or_update_file` returns `403 Resource not accessible by integration`. This is expected — commit locally and report the push failure to the user.
- **Tests live on the user's side.** There is no test harness in the repo; the user runs the code against their simulator and reports back regime + divergence from Eclipse.

## When asked to modify the drift-flux code

- Default parameters (ORIGINAL set): `A=1.2, B=0.3, Fv=1.0, a1=0.2, a2=0.4` (gas-liquid); `A'=1.2, B1=0.4, B2=0.7, n'=2` (oil-water).
- 3-phase handling: compute gas-vs-liquid slip first (combining oil+water into a pseudo-liquid), then oil-water slip inside the liquid fraction.
- SHI-03 / SHI-04 parameter sets are **not** supported yet — the user explicitly said default model only.
- Before changing iteration logic, re-read the four-bug summary above; structural bug #3 (#4 in the review) is subtle and easy to reintroduce.
