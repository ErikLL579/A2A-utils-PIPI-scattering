# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lattice QCD research code for **pion-pion (pipi) scattering** using **All-to-All (A2A) factorization** with electromagnetic (EM) corrections. The project has two main components:

1. **C++/CUDA (root directory)**: GPU-accelerated lattice computations using the Grid framework — photon propagators, meson field contractions, FFT convolutions
2. **Python (autocontraction-symbolic-manipulation/)**: Symbolic transformation of Wick contractions from quark propagator form to pion meson field form

## Build Commands

### C++ photon propagator test (CPU, MPI):
```bash
mpicxx prop-test.cc $(grid-config --cxxflags) $(grid-config --ldflags) $(grid-config --libs) -O3 -o test
```

### CUDA GPU tests:
```bash
GRID=/path/to/grid-config
nvcc FFT-test.cu $($GRID --cxxflags) $($GRID --ldflags) $($GRID --libs) -O3 -o test
nvcc matrix-vector-test.cu $($GRID --cxxflags) $($GRID --ldflags) $($GRID --libs) -O3 -o test
```

### Run with MPI:
```bash
srun -n 2 -N 1 --ntasks-per-node=2 --gpus-per-task=1 --gpu-bind=none ./test --mpi 1.1.1.2
```

### Python symbolic manipulation:
```bash
# Zeroth order pipi scattering
python3 autocontraction-symbolic-manipulation/transform_wick.py luchang-qlat-AC-output/I2_pipi_cexpr_original.txt

# EM corrections (keeps only terms with exactly 2 gamma_x)
python3 autocontraction-symbolic-manipulation/transform_wick.py luchang-qlat-AC-output/I2_pipi_EM_cexpr_original.txt --filter-gamma-x

# Optimize momentum-space output
python3 autocontraction-symbolic-manipulation/optimize_products.py <name>_mom.txt
```

## Architecture

### C++ Components

- **`A2A_Mat_Vec_util.h`** — Core header defining `PipiA2Autils<FImpl>` class (templated on fermion implementation). Contains:
  - `ContractMesonFieldAndVector`: Matrix-vector contraction producing fermion field with one outstanding A2A index
  - `MesonField_MesonField_connected/disconnected`: Batched BLAS trace operations Tr(Pi·Pi) for connected/disconnected diagrams with two-level contraction support. Level 2 uses per-operand `buffer_flag_A`/`buffer_flag_B` vectors (0 = source buffer `A`, 1 = Level 1 result buffer `C`) to handle mixed and C×C products.
  - `FFT_type1_prod/convolve`: FFT-based contraction strategies (see Appendix B/C of notes)
  - `FFT_type2_contract_convolve`: Combined contraction and convolution with photon propagator
  - Custom lattice types: `LatticeVecSpinMatrix`, `LatticeVecComplex` using `A2Ablocking=8`

- **`IV-photon-props.h`** — `IVPhotonPropagator<FImpl>` class for **infinite-volume** (continuum `1/k²`) QED photon propagators. This is distinct from Grid's built-in `Photon` class (`Grid/qcd/action/gauge/Photon.h`) which uses finite-volume lattice momentum `k_hat = 2 sin(πn/L)`. Contains:
  - Feynman gauge: diagonal `δ_μν/k²` (momentum) + FFT (position)
  - Coulomb gauge: `D_tt = 1/|vec{p}|²`, `D_ij = (δ_ij - p_i p_j/|vec{p}|²)/p²` (momentum) + FFT (position)
  - IR regulated via `min_momenta` cutoff
  - Grid convention: 4th index is time (`Tp`), momenta in `(-π, π]`

- **`prop-test.cc`** — Validation tests for photon propagators (both gauges, both spaces)

- **`load_data.h`** — Standalone header (Grid-only, no Hadrons) with two loaders:
  - `loadMesonFields(filename, datasetName, nt)` — Reads A2A meson fields from HDF5 (`.h5`) files. Returns `std::vector<MesonFieldMatrix>` (one `Eigen::Matrix<ComplexF, Dynamic, Dynamic, RowMajor>` per timeslice). On-disk format: HDF5 dataset `<datasetName>/a2aMatrix` with shape `[nt, ni, nj]` in `ComplexF`.
  - `loadBinnedA2AVecs<binSize>(vec, filestem, trajectory, grid)` — Reads binned A2A eigenvectors from SCIDAC/LIME multiFile format (`<filestem>.<traj>/elem0.bin`, `elem1.bin`, ...). Unpacks into a pre-sized `std::vector<FermionField>`. Template parameter `binSize` must match the data (e.g. 173 for light quarks, 196 for strange). Uses Grid's `ScidacReader` and `peekLorentz` for unpacking.

### Python Pipeline (autocontraction-symbolic-manipulation/)

Transforms Wick contraction output (from Luchang's qlat autocontraction) into A2A meson field form. See `autocontraction-symbolic-manipulation/CLAUDE.md` for detailed documentation of:
- The A2A factorization algebra: `S(x,y) = Σᵢ |vⁱ(x)⟩⟨wⁱ(y)|`
- Data classes: `Propagator`, `Trace`, `Term`, `MesonField`, `CurrentVertex`, `MesonChain`
- Two-phase product optimization (matrix-matrix then vector-matrix, with position locality constraints)
- Position-swap deduplication: terms equivalent under x_1 ↔ x_2 are identified and written as redirects (e.g., `term_Type12_2 = term_Type12_1 (x1 <-> x2)`), halving the number of unique evaluations (80 → 40 for EM)
- Momentum space conventions and diagram types

### Contraction Index Pipeline (autocontraction-symbolic-manipulation/)

`parse_contractions.py` (and `.ipynb` notebook) bridges the symbolic output from the Python pipeline to the C++ GPU code. It:
1. **Phase 1** (`parse_phase1`): Parses `prod_Pi` lines into Level 1 and Level 2 dicts, where each operand is either:
   - `'type': 'source'` — an original meson field `Pi(momentum, time)` (lives in GPU `A` buffer)
   - `'type': 'product'` — a reference to a Level 1 result (lives in GPU `C` buffer)
2. **Phase 2** (`parse_phase2`): Parses all Phase 2 vector-matrix products (position/gamma-independent format). Returns `{name: (vector_flag, matrix_label)}` where:
   - `vector_flag`: 0 = `<w|` (bra), 1 = `gamma |v>` (gamma-ket), 2 = current vertex (`<w| . gamma |v>`)
   - `matrix_label`: tuple `('momentum', 'time')` for source Pi, string `'prod_PiN'` or `'prod_vecN'` for product ref, or `None` for current vertex (flag=2)
   - Handles all levels (L1: source Pi, L2: absorbed bare matrix, L3: chained bra, L4: bra+L2, L5: current vertex)
3. **Phase 3** (`parse_phase3`): Parses term assembly lines into dicts of `{'coef': str, 'traces': [[operand_dicts]]}`. Operand types: `'source'` (Pi fields), `'product'` (prod_Pi refs), `'vec_product'` (prod_vec refs with gamma/position). Disconnected diagrams have multiple traces; connected have one trace with multiple operands.
4. **Index resolution** (`level1_to_contractions`, `level2_to_contractions` in `parse_contractions.py`): Maps symbolic labels to flat buffer indices:
   - `mom_map`: `{'p': p_idx, '-p': neg_p_idx, 'k': k_idx, '-k': neg_k_idx}`
   - `time_map`: `{'t_src': t_src, 't_src + Delta': (t_src+Delta)%Nt, ...}`
   - `flat_index = p * Nt * Nmodes^2 + t * Nmodes^2`
   - Level 1: Returns `[Ac, Bc, Cc]` dicts with tuple keys `(prod_name, momentum, time)` for A/B and `(prod_name, left_mom, left_time, right_mom, right_time)` for C
   - Level 2: Returns `[Aadd, Badd, flag_A, flag_B, Cadd]` — handles both explicit Level 2 products (EM) and connected traces from Phase 3 (zeroth order). Uses per-operand buffer flags (0 = source A buffer, 1 = Level 1 result C buffer). All three dicts (Aadd, Badd, Cadd) use `operand_labels()` to resolve product refs back through Level 1 into `(momentum, time, ...)` tuples, so keys are consistent across A/B/C.
5. Resolved indices fill `vector<int> A_vec, B_vec, C_vec` and `buffer_flag_A/B` for `MesonField_MesonField_connected`

**Resolved:** `MesonField_MesonField_connected` Level 2 now uses `buffer_flag_A`/`buffer_flag_B` vectors to select between source (`A`) and Level 1 result (`C`) buffers per operand, handling both mixed (A×C) and C×C products.

## Key Physics Conventions

- Meson field: `Πⁱʲ(x) = ⟨wⁱ(x)|γ₅|vʲ(x)⟩`
- Current vertex (EM): `⟨wⁱ(z)|γ_μ|vʲ(z)⟩` — stays in position space (summed over)
- Momentum assignments: `src → (p, t_src)`, `snk → (k, t_snk)`, with partner pions at `(-p, t_src+Δ)`, `(-k, t_snk+Δ)`
- Reference: arXiv 2301.09286 (RBC/UKQCD), Eqs. A5, A9

## Grid Framework Essentials

For full Grid documentation see `~/Documents/Physics/lattice/Grid_tests/Grid/CLAUDE.md`. Key patterns for this project:

### Critical Pitfalls

- **Never use `auto` for lattice arithmetic results.** Grid uses expression templates — `auto` captures the unevaluated AST, not a `Lattice`. Always assign to an explicitly-typed variable: `ComplexField tmp = toComplex(prop);` not `auto tmp = toComplex(prop);`.

- **`where` evaluates both branches everywhere.** `where(k_sqr < min, zero, one / k_sqr)` computes `1/k_sqr` at all sites including where `k_sqr == 0`, producing `inf`/`NaN`. Safe pattern:
  ```cpp
  LatticeRealD k_sqr_safe = where(k_sqr < min_momenta, one, k_sqr);
  LatticeRealD prop = where(k_sqr < min_momenta, zero, one / k_sqr_safe);
  ```

- **No scalar division by lattice:** `1.0 / lattice_field` doesn't work. Must use `one_field / lattice_field` where `one_field` is a lattice initialized to `RealD(1.0)`.

- **`LatticeComplexD` has no ordering comparisons** (`<`, `>`) — use `LatticeRealD` for comparisons, then `toComplex()` to convert.

### Type Mixing Rules

- `LatticeRealD` and `LatticeComplexD` cannot be mixed in arithmetic directly
- `toComplex(LatticeRealD)` converts real → complex
- `where(LatticeInteger_pred, complex_true, complex_false)` — predicate and value types can differ
- `LatticeCoordinate(field, mu)` fills with integer coordinate along dimension `mu` (0-indexed, `mu < Nd`)

### View Protocol (GPU code)

Lattice data must be accessed through views inside `accelerator_for`:
```cpp
autoView(v, lattice, AcceleratorRead);  // or AcceleratorWrite, CpuRead, CpuWrite
```
Views are trivially-copyable handles safe for GPU lambda capture. Use `coalescedRead`/`coalescedWrite` for SIMT access.

### FFT

`FFT` constructor requires `GridCartesian*` (not `GridBase*`): `FFT theFFT(dynamic_cast<GridCartesian*>(grid))`. Forward transform is unnormalized; backward divides by lattice extent (forward then backward = identity).

### Coordinate Access

- `grid->FullDimensions()` returns `Coordinate` of global lattice extents (0-indexed, valid indices `0..Nd-1`)
- `peekSite(scalar_obj, lattice, coor)` / `pokeSite(scalar_obj, lattice, coor)` for single-site access
- `Coordinate` assignment needs explicit constructor: `coor = Coordinate({1,0,0,0})`
- Time dimension is the last index (`Tp`, i.e., `Nd-1`)

### All code is wrapped in Grid namespace

```cpp
NAMESPACE_BEGIN(Grid);
// ... all project code ...
NAMESPACE_END(Grid);
```

## MPI and Parallelization Strategy

### Decision: Use Grid's MPI volume decomposition (`--mpi`)

The workload has ~2700 A2A vectors (lattice fermion fields) and meson field matrices Pi. The main computational patterns are:

1. **Zeroth-order contractions** (Tr[Pi·Pi]): Matrix-matrix products over A2A indices, no FFTs. Embarrassingly parallel.
2. **FFT type 1** (Appendix B of notes, unmixed A2A indices): FFTs are *outside* the mode loop — only ~16 FFTs total. Volume decomposition is fine.
3. **FFT type 2** (Appendix C of notes, mixed A2A indices): FFTs are *inside* the `(i_2, i_3)` loop — ~2700² × 4 ≈ 29 million FFTs. However, what gets FFTed is the **scalar result** of `⟨w|γ|v⟩` (a `LatticeComplex`, ~14 MB on 24³×64), not the fermion fields themselves (which would be 12× larger). The all-to-all transpose cost per FFT is small relative to the local computation.

**Use `--mpi` with volume decomposition** (e.g., `--mpi 2.5.5.1` for 50 ranks). This is simpler than manual MPI mode-index distribution and the FFT communication overhead is acceptable:
- Grid's distributed `FFT` class handles split volumes correctly (transposes as needed per dimension).
- With `--mpi 1.1.1.N` (time-only split), spatial FFTs are fully local — only the time-direction FFT requires communication.
- Each rank holds 1/N of each lattice field's volume, so memory is automatically distributed.
- Grid's I/O routines (`BinaryIO`, ILDG) are MPI-aware — each rank reads only its local portion.
- Lattice dimensions must be divisible by the corresponding MPI dimensions.

### GPU binding on Perlmutter

**Must use `--gpu-bind=none`, NOT `--gpu-bind=single:1`.** Grid's FFT uses `Cshift` → `MPI_Sendrecv` with GPU pointers for inter-rank communication. Cray MPICH's GPU Transport Layer (GTL) uses CUDA IPC (`cuIpcOpenMemHandle`) for same-node transfers, which requires all GPUs to be visible to all ranks. `--gpu-bind=single:1` restricts `CUDA_VISIBLE_DEVICES` so each rank only sees its own GPU, causing `cuIpcOpenMemHandle: CUDA_ERROR_INVALID_VALUE`. Grid's `--enable-setdevice` (used in our build) handles GPU assignment via `cudaSetDevice`, so Slurm only needs to *allocate* GPUs (`--gpus-per-task=1`), not restrict visibility.

**Alternative considered and rejected for now:** Mode-index distribution (`--enable-comms=none` + manual `MPI_Init`/`MPI_Comm_rank`). Each rank holds the full volume but only a subset of A2A vectors. Eliminates FFT communication entirely but requires manual MPI code and custom I/O. Would only be worth it if FFT communication proves to be a bottleneck after profiling.

### FFT convolution optimization (Appendix A of notes)

The FFT convolution `C = Σ_{x,y} φ(x) ψ(y) Δ(x-y)` can be computed entirely in momentum space: `C = Σ_k φ̃(-k) Δ̃(k) ψ̃(k)`. No inverse FFT is needed — Parseval's theorem applies since C is a scalar. This saves one FFT per convolution compared to the IFFT+position-space-sum approach.

**Important:** The bilinear `⟨w(x)|γ|v(x)⟩` is a site-local pointwise product, which becomes a convolution in momentum space. You cannot FFT v and w individually and combine in momentum space — that gives only the diagonal term, missing all cross-terms. The bilinear must be computed in position space first, then the scalar result FFTed.

## Dependencies

- **Grid**: Lattice QCD framework (`#include <Grid/Grid.h>`, `Grid_Eigen_Tensor.h`, `A2Autils.h`). Configured via `grid-config`.
- **CUDA/NVCC**: For GPU-accelerated `.cu` files
- **Python 3**: Standard library only (dataclasses, typing, re, collections)
