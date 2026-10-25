# Security notes: pqblocks-bifurcation

## What this package is

A classical optimisation heuristic. It is **not** a cryptographic primitive. It is not "quantum-safe" or "post-quantum secure" in any sense, and it must not be used to protect data. It handles no keys and no secrets.

## Randomness

- The random initial state comes from NumPy's PRNG. It is used for reproducibility, never for anything secret.
- Seeding calls `np.random.seed(seed)`, which **reseeds NumPy's global RNG** on every call, including when `seed=None`. Applications that depend on the global NumPy RNG state should use their own `np.random.Generator`.

## Untrusted input

- `sample_ising` rejects empty models, shape mismatches, NaN/Inf values, invalid sweep counts and invalid initial states with `ValueError`.
- **Resource use** grows as O(N²) memory and O(num_sweeps · N²) time. A caller handling untrusted problem sizes should cap `N` and `num_sweeps` to avoid excessive CPU or memory use.
- The package does no file, network or subprocess access.

## Dependencies

- Runtime: `numpy` and `numba`. The first call compiles kernels with Numba's JIT.
- `pip-audit` reported no known vulnerabilities in the dependency set at release time (see [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md)).

## Audit status

The package has been checked by its own test suite and an internal adversarial input review. It has had no external third-party audit.
