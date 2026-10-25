"""Max-Cut on a small weighted graph with Simulated Bifurcation, checked against brute force.

Max-Cut maps to an Ising model: with J_ij = w_ij the solver minimises sum_{i<j} w_ij s_i s_j, and the cut
value of a spin assignment is sum_{i<j} w_ij (1 - s_i s_j) / 2. For 8 nodes brute force is instant, so the
example prints both and says whether the heuristic found the optimum (it is a heuristic: it may not).
"""
import itertools

import numpy as np

from pqblocks_bifurcation.solver import SimulatedBifurcationSampler

rng = np.random.default_rng(7)
n = 8
W = np.triu(rng.integers(0, 5, size=(n, n)).astype(float), k=1)
W = W + W.T                                   # symmetric weights, zero diagonal


def cut_value(spins):
    return float(sum(W[i, j] * (1 - spins[i] * spins[j]) / 2 for i in range(n) for j in range(i + 1, n)))


best_cut = 0.0
for bits in itertools.product((-1, 1), repeat=n):
    best_cut = max(best_cut, cut_value(np.array(bits)))

results = []
for seed in range(10):                         # several independent reads; keep the best
    spins, energy = SimulatedBifurcationSampler(seed=seed, mode="dSB").sample_ising(W, np.zeros(n))
    results.append(cut_value(spins))

print("best cut found by SB over 10 reads:", max(results))
print("optimal cut (brute force):        ", best_cut)
print("optimum reached:", max(results) == best_cut)
