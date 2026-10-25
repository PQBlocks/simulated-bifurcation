# Testing: pqblocks-bifurcation

**Status at release 0.1.0:**
- 38 of 38 tests pass on Python 3.11, 3.12, 3.13 and 3.14.
- Each run used the built wheel, installed in a fresh virtual environment with dependencies from PyPI.
- Statement coverage is 90 %.

## How to run

```bash
pip install "pqblocks-bifurcation[test]"
pytest -q                      # from a checkout: pip install -e ".[test]" first
```

Coverage of the Numba kernels can only be traced with the JIT disabled:

```bash
NUMBA_DISABLE_JIT=1 pytest -q --cov=pqblocks_bifurcation --cov-report=term-missing
```

## Results (measured 2026-10-24 UTC)

| Python | Install source | Tests | Examples |
|---|---|---|---|
| 3.11 | wheel + PyPI dependencies | 38 passed | 3 / 3 passed |
| 3.12 | wheel + PyPI dependencies | 38 passed | 3 / 3 passed |
| 3.13 | wheel + PyPI dependencies | 38 passed | 3 / 3 passed |
| 3.14 | wheel + PyPI dependencies | 38 passed | 3 / 3 passed |

The dependencies resolved at release were numba 0.68.0, llvmlite 0.50.0, numpy 2.5.4 and scipy 1.18.1.

### Coverage (`NUMBA_DISABLE_JIT=1`)

| Module | Statements | Missed | Coverage |
|---|---|---|---|
| `__init__.py` | 5 | 0 | 100 % |
| `kernel.py` | 92 | 9 | 90 % |
| `reference.py` | 41 | 11 | 73 % |
| `solver.py` | 32 | 0 | 100 % |
| `validation.py` | 22 | 0 | 100 % |
| **Total** | **192** | **20** | **90 %** |

## What the tests check

### Correctness against independent oracles: `test_reference.py` (5)

| Test | Oracle |
|---|---|
| `test_4spin_worked_example_bsb`, `test_4spin_worked_example_dsb` | Hand-derived first integration step of a 4-spin model (c0 = 0.25, Δt = 0.5), asserted value by value |
| `test_sign_convention` | The J → −J, h → −h mapping onto oscillator wells |
| `test_energy_calculation_independent` | Energy re-derived from the formula in the test itself, not by calling the solver's own energy function |
| `test_determinism` | Same seed gives an identical trajectory |

### Two implementations must agree: `test_kernel.py` (6)

| Test | What it proves |
|---|---|
| `test_bsb_bit_identity`, `test_dsb_bit_identity`, `test_multiple_sweeps_bit_identity` | The Numba kernel and the pure NumPy reference produce **bit-identical** results for the same seed (bSB, dSB, many sweeps) |
| `test_compute_c0_function` | The c0 scaling, including the zero-coupling fallback |
| `test_initialize_xy_function` | The initial state range and its seeding |
| `test_edge_case_single_spin` | N = 1 |

### Public API behaviour: `test_solver.py` (17)

| Area | Tests |
|---|---|
| Valid solves | `sample_ising_valid`, `bsb_vs_dsb`, `with_initial_conditions`, `single_spin`, `decoupled_model`, `multiple_decoupled_spins` |
| Input normalisation | `non_symmetric_j`, `non_zero_diagonal`, `asymmetric_matrix`, `nonzero_diagonal`, `asymmetric_and_diagonal`: J is symmetrised and the diagonal ignored, as documented |
| Rejection of bad input | `invalid_mode`, `empty_model`, `nan_inf_inputs`, `invalid_initial_conditions`, `dt_validation`: each must raise `ValueError` |
| Construction | `initialization` |

### Input validation: `test_validation.py` (10)

Empty model, dimension mismatch, NaN/Inf, non-symmetric J, non-zero diagonal, valid inputs, negative and valid sweep counts, valid and invalid initial conditions.

### Coverage by problem dimension

| Dimension | Covered |
|---|---|
| Problem size | N = 1 to 8 spins in the unit tests; up to N = 100 exercised before release |
| Topology | dense, sparse, 1-D chains, fully decoupled |
| Couplings | antiferromagnetic, ferromagnetic, random |
| Degenerate cases | multiple optima, symmetric basins |
| Determinism | fixed-seed replay |

## Examples run as tests

Each example checks its own answer and runs in every release check above:

| Example | Check |
|---|---|
| `examples/quickstart.py` | prints `Spins: [1 1]` and `Energy: -1.0` |
| `examples/maxcut.py` | the best cut over 10 reads equals the brute-force optimum (44.0) |
| `examples/qubo.py` | the QUBO→Ising mapping finds the brute-force optimum (objective −7.0) |

## What is not covered

- **GPU execution:** not implemented.
- **Large problems:** problems above N = 1000 are not tested.
- **Benchmarks are separate:** solution quality on large instances is a benchmarking question, covered in [BENCHMARKS.md](BENCHMARKS.md).
- **Not in this suite:** the pre-release cross-check of benchmark optima against an exact solver is not part of the shipped tests. Its results are in BENCHMARKS.md.
