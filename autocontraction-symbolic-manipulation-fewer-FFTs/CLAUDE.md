# Autocontraction Symbolic Manipulation — Fewer-FFTs Variant

This directory is a variant of `../autocontraction-symbolic-manipulation/` with one
design change aimed at reducing the number of position-space FFTs needed at the
C++/CUDA stage. See the parent directory's `CLAUDE.md` for the shared algebra,
momentum conventions, diagram types, and Python concepts — only the differences
are documented here.

## Goal

Reduce the FFT count in the EM-corrections C++/CUDA stage by ensuring every Phase 3
term contains **exactly one standalone factor of `<w|γ_μ|v>`**, so that a single
FFT of `<w|γ_μ|v>(x)` can be reused across all terms instead of FFTing each
chained `prod_vec` separately.

## What's the same

- `transform_wick.py` is unchanged — produces `*_pos.txt` and `*_mom.txt` exactly
  as before.
- The two-phase product optimization structure (Phase 1 = matrix-matrix, Phase 2
  = vector-matrix, Phase 3 = term assembly) is preserved.
- Position locality, no vector-vector outer products, and `_mu` canonical naming
  with gamma/position re-attachment in Phase 3 — all preserved.
- Position-swap dedup (`x_1 ↔ x_2`) — preserved.

## What's different

### Universal `<w|γ_μ|v>` factor

In the original optimizer, the current vertex `<w|γ_μ|v>` is factored out as a
Level 5 product (`prod_vec57_mu`) but only some terms contain it explicitly —
specifically Types 1–4 already have it standalone, while Types 5–12 absorb both
`<w|γ|v>` insertions into longer chained products like
`<w|γ_μ|v>(x_1) · prod_Pi · ...`.

In this variant, **every Phase 3 term must contain exactly one standalone
`<w|γ_μ|v>` factor.** The natural place for this is the bottom of the Phase 2
hierarchy:

```
prod_vec1_mu = <w| . gamma_mu |v>     # universal Level 1 building block
```

Other vector products are renumbered from `prod_vec2` onward.

### Term structure after optimization

Every EM term has two current-vertex insertions, one at `x_1` and one at `x_2`.
A generic term is:

```
Tr[V_μ(x_1) · M_1 · V_ν(x_2) · M_2]
```

where `V_x ≡ <w|γ|v>(x)` and `M_i` are matrix products (Pi-chains, position-free).
Trace cyclicity lets us choose to absorb everything except *one* `V` into a single
chained product at one position. Two options:

- **(a) Keep `V_ν(x_2)` standalone** → chained product at x_1:
  `prod_vec_X_μ(x_1) = <w|γ_μ|v>(x_1) · M_1 · M_2`
- **(b) Keep `V_μ(x_1)` standalone** → chained product at x_2:
  `prod_vec_X_ν(x_2) = <w|γ_ν|v>(x_2) · M_2 · M_1`

The two options yield *different* chained products (cyclic rotations of the
matrix string). The optimizer chooses between (a) and (b) per term to maximize
chained-product reuse across all 40 unique terms.

## Optimization rule

For each term, pick the option whose resulting chained product appears most often
globally (highest reuse → fewest unique chained products to compute).

Implementation strategy:

1. **Single-pass greedy (default):** For each term, enumerate both candidate
   chained products in canonical form (ignoring `mu/nu` and `x_1/x_2` labels —
   same canonicalization as the existing dedup). Compute global occurrence
   counts across all terms (treating each candidate as if it were chosen).
   For each term, pick the option whose candidate has the higher initial global
   count. Ties broken by lexicographic order of the chained product name.

2. **Iterative fix-point (optional, `iterate=True`):** Same as above, but after
   each pass recompute global counts using the *current* set of choices, and
   re-pick if any term's optimal choice has changed. Repeat until stable. On
   the EM input, iterative produces the same 32 unique chained products as
   single-pass, so single-pass is the default.

The choice is independent of the position-swap dedup pass — that runs on top of
whichever choice was made.

## Implementation overview (`optimize_products_fewer_FFT.py`)

Top-level entry point: `optimize(terms, iterate=False)` → returns
`(prod_Pi_lines, prod_vec_lines, term_lines)`. CLI is `main()`, invoked the same
way as the parent optimizer.

Pipeline:

1. **Phase 1 (verbatim from parent):** Build `prod_Pi` matrix-matrix products
   from the disconnected meson chains.
2. **Term decomposition (`decompose_term`):** For each term, separate the trace
   containing the two current vertices (`<w|γ_μ|v>(x_1)` and `<w|γ_ν|v>(x_2)`)
   into two segments around them. Disconnected terms keep `other_traces`
   verbatim; connected terms have a single trace whose vertex segments are the
   two cyclic options.
3. **Candidate enumeration (`enumerate_candidates`):** For each term produce
   `option_a` (keep V_ν standalone, absorb V_μ + segments into a chain at x_1)
   and `option_b` (keep V_μ standalone, absorb V_ν + segments into a chain at
   x_2). Each candidate carries a canonical chain key with `mu/nu` and
   `x_1/x_2` stripped, used for global counting.
4. **Global counting + greedy choice (`count_candidate_chains`,
   `greedy_choose`):** Sum candidate occurrences across all terms; for each
   term pick the option with the higher count (lex-tie-break). With
   `iterate=True`, recount using only chosen candidates and repeat until
   stable.
5. **Naming (`assign_names_and_rewrite`):** `prod_vec1_mu = <w|.γ_μ|v>` is
   universal. Remaining unique canonical chains get `prod_vec2_mu`,
   `prod_vec3_mu`, … in order of descending global count (lex tie-break).
   Every term is rewritten as
   `Tr[prod_vec1_X(x_N) · prod_vecK_X'(x_M)] · <other traces>`.
6. **Position-swap dedup (`_dedup_position_swap`, copied verbatim):** Terms
   equivalent under `x_1 ↔ x_2` (and `μ ↔ ν`) are emitted as redirects.

## Output (`I2_pipi_EM_cexpr_mom_optimized.txt`)

For the EM input:

- 24 `prod_Pi` products (Phase 1, unchanged from parent).
- 1 `prod_vec1_mu = <w| . gamma_mu |v>` (used in **every** term, 80 occurrences).
- 62 chained `prod_vec` products across 6 further levels:
  - L2 (14): `Pi.<w|`, `gamma|v>.Pi`, `gamma|v>.prod_Pi` (Phase 1 L1)
  - L3 (10): `gamma|v>.prod_Pi` (Phase 1 L2)
  - L4 (10): closure — `<w|.prod_vec_X`, `prod_vec_X.gamma|v>` (closing L2)
  - L5 (8):  closure — `<w|.prod_vec_X` (closing L3)
  - L6 (18): canonical-pair closures (the new pass — see below)
  - L7 (2):  canonical-pair closures (iteration 2)
  Includes bra-side dressed products (`Pi . <w|`) which the parent optimizer
  forbids via its orphan-bra rule — needed here because absorbed-vertex chains
  in connected types (Type5/9) have Pi matrices on both sides of the split
  bra/ket.
- 80 terms → 40 unique definitions + 40 redirects via position-swap dedup.
- **Every** term has a 2-element main trace `Tr[prod_vec1_X(pos1).prod_vecK_Y(pos2)]`
  (plus optional `Tr[prod_PiN]` factors for disconnected diagrams). The
  previously-suboptimal Type5_2 and Type9_1 (4-element traces) are now
  cleanly 2-element via L7 closures.

### Canonical-pair closure pass (`_close_canonical_pairs`)

Why needed: Phase 2's pair counting uses position-attached names, so
`prod_vec4(x_1) . prod_vec7_mu(x_1)` and `prod_vec4(x_2) . prod_vec7_nu(x_2)`
are two different keys with count 1 each. They fall below the count≥2
threshold so the closure isn't created — even though they are *canonically*
the same pair (equivalent under x_1↔x_2/μ↔ν).

The canonical-pair pass runs after `_absorb_bare_matrices`. It re-collects
pairs using canonical pair keys (positions and gamma labels stripped) and
substitutes per-position instances of any canonical pair occurring ≥2
times. Iterates until no canonical pair has count ≥2. The unsplit standalone
`<w|γ_X|v>(pos)` is excluded (still in trace form at this point), and raw
`<w|(pos).γ|v>(pos)` pairs are skipped so `_create_current_vertex_product`
can still factor them into the universal vertex.

Six representative terms (Type1_0, Type5_2, Type6_0, Type9_1, Type10_0,
Type11_0) hand-expanded and verified against the original
`I2_pipi_EM_cexpr_mom.txt` up to cyclic permutation of traces.

## Files

- `transform_wick.py` — copied verbatim from parent directory; unchanged.
- `luchang-qlat-AC-output/` — copied verbatim; input cexpr files.
- `optimize_products_fewer_FFT.py` — implements the rule above. Run as
  `python3 optimize_products_fewer_FFT.py I2_pipi_EM_cexpr_mom.txt` to produce
  `I2_pipi_EM_cexpr_mom_optimized.txt`.
- `I2_pipi_EM_cexpr_pos.txt`, `I2_pipi_EM_cexpr_mom.txt` — outputs from
  `transform_wick.py` (input to the optimizer).
- `I2_pipi_EM_cexpr_mom_optimized.txt` — final optimizer output (the
  fewer-FFTs hierarchy).
- (Future) `parse_contractions.py`, `create_output_cu.py` — adapt parent
  versions to the new product hierarchy.

## Status

- 2026-04-30: Directory created. `transform_wick.py` and input data copied
  from parent. Design agreed.
- 2026-05-01: `optimize_products_fewer_FFT.py` implemented (single-pass
  greedy, with optional iterative fix-point). Verified on EM input: 80 terms
  → 40 unique with universal `prod_vec1_mu` factor and 42 chained products.
  Six representative terms (Type1_0, Type5_2, Type6_0, Type9_1, Type10_0,
  Type11_0) hand-checked against original `_mom.txt`. Two terms (Type5_2,
  Type9_1) retain 4-element traces but are mathematically correct.
- 2026-05-04: Added `_close_canonical_pairs` pass to absorb same-position
  pairs whose canonical (position/gamma-stripped) signature occurs ≥2 times
  even when full-name counts are 1 each. Now every term has the optimal
  2-element main trace form, including the previously suboptimal Type5_2
  and Type9_1. Total: 62 chained prod_vec across 6 further levels.
