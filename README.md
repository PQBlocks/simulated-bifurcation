# pqblocks-bifurcation

Simulated Bifurcation solver for Ising and QUBO minimisation, in ballistic (bSB) and discrete (dSB) form.
Use it to get fast approximate solutions to small and medium combinatorial problems (Max-Cut, Ising spin glasses, QUBO) on a CPU.
Implements the algorithms of Goto et al., *Science Advances* 2019 and 2021. Quantum-inspired classical optimisation: **not cryptography, no security claim**.

```bash
pip install pqblocks-bifurcation
```

```python
import numpy as np
from pqblocks_bifurcation.solver import SimulatedBifurcationSampler

sampler = SimulatedBifurcationSampler(seed=42, mode="dSB")
J = np.array([[0.0, -1.0], [-1.0, 0.0]])   # couplings
h = np.array([0.5, -0.5])                   # local fields
spins, energy = sampler.sample_ising(J, h)
print(spins, energy)                        # [1 1] -1.0
```

## Why PQBlocks

- **Tested:** 38 tests. They cover a hand-derived worked-example trace, bit-identical agreement between the NumPy reference and the Numba kernel, and loud rejection of bad input.
- **Honest benchmarks:** the losing cases are published. See [BENCHMARKS.md](BENCHMARKS.md).
- **Licence:** Apache-2.0, clean-room implementation from the published papers.

## Usage

**Energy.** The solver minimises

`H(s) = 1/2 · Σ_{i≠j} J_ij s_i s_j + Σ_i h_i s_i`, with `s_i ∈ {-1, +1}`.

`J` is symmetrised and its diagonal zeroed before solving.

**Modes.**
- `mode="dSB"` (discrete, the default) uses `sign(x_j)` in the coupling force.
- `mode="bSB"` (ballistic) uses `x_j` directly.

**Reads.** Each `sample_ising` call is one read, and the solver is a heuristic. Run several seeds and keep the best result; see `examples/maxcut.py`.

**QUBO.** Convert with `J_ij = Q_ij/2` and `h_i = Q_ii/2 + Σ_{j≠i} Q_ij/2`, where `x = (1+s)/2`. `examples/qubo.py` shows this and checks it against brute force.

**Reproducibility.** The same `seed` gives bit-identical results.

## API reference

`SimulatedBifurcationSampler(dt=0.5, num_sweeps=1000, mode="dSB", seed=None)`

| Argument | Meaning |
|---|---|
| `dt` | Integration step. Must be positive and finite. 0.5 is the stability limit used in the papers. |
| `num_sweeps` | Number of integration steps; the pump ramps linearly from 0 to 1 over them. |
| `mode` | `"dSB"` or `"bSB"`. |
| `seed` | Seed for the random initial positions and momenta (uniform in [-0.1, 0.1]). |

`sample_ising(J, h, x_init=None, y_init=None) -> (spins, energy)`

- `J`: (N, N) array of couplings.
- `h`: (N,) array of local fields.
- Optional `x_init` / `y_init`: initial oscillator state.
- Returns the ±1 spin vector and its energy.
- Raises `ValueError` on an empty model, mismatched shapes, NaN/Inf values, invalid `num_sweeps`, invalid initial state or an unknown mode.

Lower-level modules: `kernel` (Numba kernels), `reference` (pure NumPy reference) and `validation` (input checks).

## Limitations

- **Heuristic:** there is no optimality guarantee. On the dense 8-spin benchmark instance it stops about 1 % above the optimum.
- **Problem size:** tested up to N = 100. Behaviour above N = 1000 is unverified.
- **Hardware:** CPU only. The first call includes Numba compile time.
- **Global RNG side effect:** seeding uses NumPy's global RNG (`np.random.seed`), so each call reseeds it. Code that relies on the global NumPy RNG state should use its own `np.random.Generator`.

## Security notes

This package does optimisation only. It handles no keys or secrets and provides no protection. See [SECURITY_NOTES.md](SECURITY_NOTES.md).

## Verification

[TESTING.md](TESTING.md) · [BENCHMARKS.md](BENCHMARKS.md) · [EVIDENCE.md](EVIDENCE.md) · [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) · [USE_CASES.md](USE_CASES.md)

## Part of PQBlocks

[PQBlocks](https://github.com/PQBlocks) builds open, tested building blocks for the post-quantum transition. This package is one of its classical optimisation tools.
