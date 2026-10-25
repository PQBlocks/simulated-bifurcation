# Verification report: pqblocks-bifurcation 0.1.0

This is a **self-verification** report: what was checked, how, and with what result. It is not a third-party audit or external validation of any kind.

## Summary

| Area | Result |
|---|---|
| Correctness | 38/38 tests pass; worked example and bit-identical dual implementation verified |
| Coverage | 90 % statements |
| Install as a user gets it | wheel installed from scratch with PyPI dependencies on Python 3.11, 3.12, 3.13 and 3.14: all tests and examples pass |
| Packaging | sdist and wheel build cleanly; `twine check --strict` passes |
| Dependencies | `pip-audit`: no known vulnerabilities (numba 0.68.0, llvmlite 0.50.0, numpy 2.5.4, scipy 1.18.1) |
| Benchmarks | compared with four baselines; losing case published ([BENCHMARKS.md](BENCHMARKS.md)) |
| Scope honesty | heuristic, no optimality guarantee, CPU only, untested above N = 1000 |

## Review checks completed before release

| Check | Purpose | Result |
|---|---|---|
| Check 1 (design review) | Method matches the cited papers; parameters and edge cases specified | Pass |
| Check 2 (spec review) | Specification complete and typed; no undefined behaviour | Pass |
| Check 3 (correctness) | Test suite, independent oracles, coverage | Pass (38/38, 90 %) |
| Check 4 (benchmark review) | Baselines run, statistics, losing cases reported | Pass |
| Check 5 (adversarial review) | Pathological inputs (empty, NaN/Inf, mismatched shapes, bad initial state) fail loudly | Pass |
| Release check (this report) | Build, metadata, fresh-environment install on four Pythons, dependency audit | Pass |

## Release-check evidence (2026-10-24 UTC)

```
build sdist+wheel                  PASS  pqblocks_bifurcation-0.1.0-py3-none-any.whl
twine check --strict               PASS
python 3.11: tests                 PASS  38 passed
python 3.12: tests                 PASS  38 passed
python 3.13: tests                 PASS  38 passed
python 3.14: tests                 PASS  38 passed
examples (quickstart, maxcut, qubo) PASS on all four versions
pip-audit                          No known vulnerabilities found
```

A packaging defect was found and fixed during this release check. The Numba kernels need **scipy** at runtime for matrix operations, but it had not been declared. In a fresh environment 15 tests and all examples failed with `ImportError`. `scipy` is now a declared dependency, and everything passes.

## Known limitations

- **Heuristic:** on the dense 8-spin benchmark instance both variants stop 1.06 % above the optimum (0 % success in 100 runs).
- **Benchmark scope:** benchmarks cover N ≤ 8 only, and the per-call work differs between solvers (see the note in BENCHMARKS.md).
- **Global RNG:** seeding reseeds NumPy's global RNG.
- **Reproduction script:** the one-command benchmark reproduction ships with the shared benchmark harness `pqblocks-qubo-bench`.

## Validation status

Not FIPS or CMVP validated, and no such validation applies: this is not a cryptographic module.
