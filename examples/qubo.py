"""Solve a small QUBO (minimise x^T Q x over x in {0,1}^n) by mapping it to the solver's Ising form.

With x = (1 + s) / 2 and a symmetric Q, the Ising problem the sampler minimises,
H(s) = 1/2 sum_{i!=j} J_ij s_i s_j + sum_i h_i s_i, is obtained with
    J_ij = Q_ij / 2            (i != j)
    h_i  = Q_ii / 2 + sum_{j!=i} Q_ij / 2
(the constant offset is dropped). The example checks the answer against brute force.
"""
import itertools

import numpy as np

from pqblocks_bifurcation.solver import SimulatedBifurcationSampler


def qubo_to_ising(Q):
    Q = (Q + Q.T) / 2
    J = Q / 2
    np.fill_diagonal(J, 0.0)
    h = np.diag(Q) / 2 + (Q.sum(axis=1) - np.diag(Q)) / 2
    return J, h


Q = np.array([[-3.0, 2.0, 0.0, 1.0],
              [2.0, -2.0, 1.0, 0.0],
              [0.0, 1.0, -4.0, 2.0],
              [1.0, 0.0, 2.0, -1.0]])
J, h = qubo_to_ising(Q)

best = min((float(np.array(x) @ Q @ np.array(x)), x) for x in itertools.product((0, 1), repeat=4))
reads = [SimulatedBifurcationSampler(seed=s).sample_ising(J, h)[0] for s in range(5)]
found = min((float(((1 + sp) // 2) @ Q @ ((1 + sp) // 2)), tuple(int(v) for v in (1 + sp) // 2)) for sp in reads)

print("best x found by SB:", found[1], "objective", found[0])
print("brute-force optimum:", best[1], "objective", best[0])
print("optimum reached:", found[0] == best[0])
