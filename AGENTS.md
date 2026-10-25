# AGENTS.md: pqblocks-bifurcation

Guidance for AI coding agents using this package.

- **Use it for:** approximate minimisation of Ising or QUBO objectives (Max-Cut, spin glasses, small combinatorial models) on a CPU.
- **Install:** `pip install pqblocks-bifurcation`
- **Correct import:** `from pqblocks_bifurcation.solver import SimulatedBifurcationSampler`
- **Call:** `spins, energy = SimulatedBifurcationSampler(seed=0, mode="dSB").sample_ising(J, h)`
  - `J` is (N, N) and `h` is (N,). Both are NumPy float arrays.
  - `J` is symmetrised and its diagonal ignored.
  - `spins` are ±1.
- **QUBO:** convert with `J_ij = Q_ij/2` and `h_i = Q_ii/2 + Σ_{j≠i} Q_ij/2` (see `examples/qubo.py`).
- **It is a heuristic.** Run several seeds and keep the lowest energy. Do not present its output as a proven optimum.
- **Do not use it for security or cryptography.** It provides none.
- **Global RNG side effect:** seeding reseeds NumPy's global RNG. Use a local `np.random.Generator` elsewhere in your code.
