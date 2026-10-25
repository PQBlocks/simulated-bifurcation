# Evidence: pqblocks-bifurcation

This file shows where each part of the implementation comes from, and how the code maps to the published method.

## Primary sources

1. H. Goto, K. Tatsumura, A. R. Dixon. *Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems.* Science Advances 5(4), eaav2372 (2019). doi:[10.1126/sciadv.aav2372](https://doi.org/10.1126/sciadv.aav2372)
   - Maps combinatorial optimisation to coupled parametric oscillators.
2. H. Goto, K. Endo, M. Suzuki, Y. Sakai, T. Kanao, Y. Hamakawa, R. Hidaka, M. Yamasaki, K. Tatsumura. *High-performance combinatorial optimization based on classical mechanics.* Science Advances 7(6), eabe7953 (2021). doi:[10.1126/sciadv.abe7953](https://doi.org/10.1126/sciadv.abe7953)
   - Defines ballistic (bSB) and discrete (dSB) Simulated Bifurcation with inelastic walls, and the c0 scaling.
3. E. S. Tiunov, A. E. Ulanov, A. I. Lvovsky. *Annealing by simulating the coherent Ising machine.* Optics Express 27(7), 10288 (2019). doi:[10.1364/OE.27.010288](https://doi.org/10.1364/OE.27.010288)
   - Background on the related SimCIM mean-field model.

## Method to code

| Element | Published method | Implementation |
|---|---|---|
| Objective | Ising energy over s ∈ {−1,+1}^N | `solver.SimulatedBifurcationSampler.sample_ising`; energy `H(s) = Σ_{i<j} J_ij s_i s_j + Σ_i h_i s_i` |
| Sign convention | Minimisation mapped onto oscillator potential wells | `J_SB = −J`, `h_SB = −h` before integration (`solver.py`) |
| Pump schedule | a(t) ramps from 0 to 1 | `a = (step + 1) / num_sweeps`, linear (`kernel.py`) |
| Coupling force | bSB uses x_j; dSB uses sign(x_j) | `f = J_SB @ x + h_SB` (bSB); `f = J_SB @ sign(x) + h_SB` (dSB) |
| Symplectic Euler | y ← y + [−(1−a)x + c0·f]Δt, then x ← x + yΔt | the same two updates in `_simulate_bsb_kernel` / `_simulate_dsb_kernel` |
| Inelastic walls | \|x_i\| > 1 ⇒ x_i ← sign(x_i), y_i ← 0 | the same, applied every step |
| Read-out | s_i = sign(x_i) | the same, with x_i = 0 mapped to +1 |
| c0 scaling | c0 = 0.5 / (σ_J √N) | `kernel._compute_c0`: σ_J is the std. dev. of the off-diagonal J_SB. Implementation choice: when σ_J ≈ 0 (no couplings), c0 = 0.5/√N so local fields still drive the dynamics |
| Initial state | small random perturbation | x, y ~ U[−0.1, 0.1], seeded |
| Step size | Δt = 0.5 | default `dt=0.5`, validated positive and finite |

## Independent checks of the mapping

- **Worked example:** a hand-derived trace of the first integration step for a 4-spin antiferromagnetic model (c0 = 0.25, Δt = 0.5) is asserted bit-for-bit in `tests/test_reference.py`.
- **Two implementations:** a pure NumPy reference (`reference.py`) and the Numba kernel (`kernel.py`) are required to give bit-identical trajectories for the same seed (`tests/test_kernel.py`).
- **Exact optima:** the brute-force-proven optima of the benchmark instances are listed in [BENCHMARKS.md](BENCHMARKS.md).

## Licence and provenance

- Written from the equations in the papers above.
- No third-party solver code is included or linked.
- Baseline solvers (dwave-neal, OpenJij, MQLib, OR-Tools) were used only as separate comparison tools.
- Distributed under Apache-2.0.
