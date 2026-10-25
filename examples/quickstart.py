import numpy as np
from pqblocks_bifurcation.solver import SimulatedBifurcationSampler

# 1. Initialize solver with fixed seed and discrete SB (dSB) mode
sampler = SimulatedBifurcationSampler(seed=42, mode='dSB')

# 2. Define a small antiferromagnetic problem
J = np.array([[0.0, -1.0], [-1.0, 0.0]])
h = np.array([0.5, -0.5])

# 3. Solve the problem
spins, energy = sampler.sample_ising(J, h)
print(f"Spins: {spins}")
print(f"Energy: {energy}")
