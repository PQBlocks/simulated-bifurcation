import numpy as np
from typing import Optional, Tuple
from .kernel import simulate_sb_numba, _compute_c0
from .validation import validate_j_h, validate_num_sweeps, validate_initial_conditions


class SimulatedBifurcationSampler:
    """
    A sampler for the Simulated Bifurcation (SB) algorithm.
    
    This class provides a clean API for solving Ising problems using the
    Simulated Bifurcation algorithm with both ballistic (bSB) and discrete (dSB)
    variants.
    """
    
    def __init__(self, dt: float = 0.5, num_sweeps: int = 1000, 
                 mode: str = 'dSB', seed: Optional[int] = None):
        """
        Initialize the Simulated Bifurcation sampler.
        
        Parameters:
        - dt: Time step (fixed at 0.5 per parameter policy)
        - num_sweeps: Number of simulation steps (defaults to 1000)
        - mode: 'bSB' for ballistic SB, 'dSB' for discrete SB
        - seed: Random seed for initialization
        """
        # Validate dt parameter
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        
        self.dt = dt
        self.num_sweeps = num_sweeps
        self.mode = mode
        self.seed = seed
    
    def sample_ising(self, J: np.ndarray, h: np.ndarray,
                     x_init: Optional[np.ndarray] = None,
                     y_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Sample from an Ising model using the Simulated Bifurcation algorithm.
        
        Solves the Ising minimization problem: 
        H(s) = 1/2 * sum_{i!=j} J_{ij} * s_i * s_j + sum_i h_i * s_i
        
        Parameters:
        - J: Coupling matrix (NxN), will be symmetrized and diagonal zeroed
        - h: Local field vector (N,)
        - x_init: Initial x values (N,), if None will be randomly initialized
        - y_init: Initial y values (N,), if None will be randomly initialized
        
        Returns:
        - spins: Final spin states (N,) in {-1, +1}
        - energy: Energy of the final state
        
        Raises:
        - ValueError: If input validation fails
        """
        # Validate inputs
        validate_num_sweeps(self.num_sweeps)
        validate_initial_conditions(x_init, y_init, len(h))
        
        # Validate mode
        if self.mode not in ['bSB', 'dSB']:
            raise ValueError("mode must be 'bSB' or 'dSB'")
        
        # Automatic matrix normalization: symmetrize and zero diagonal
        J_sym = (J + J.T) / 2  # Symmetrize
        np.fill_diagonal(J_sym, 0)  # Zero out diagonal
        
        # Validate J and h after normalization
        validate_j_h(J_sym, h)
        
        # Compute c0 according to parameter policy
        J_SB = -J_sym  # Apply sign convention
        c0 = _compute_c0(J_SB)
        
        # Run simulation
        x_final, y_final, spins = simulate_sb_numba(
            J_sym, h, self.dt, c0, self.num_sweeps,
            x_init, y_init, self.mode, self.seed
        )
        
        # Calculate energy of the final state
        energy = self._calculate_ising_energy(J_sym, h, spins)
        
        return spins, energy
    
    def _calculate_ising_energy(self, J: np.ndarray, h: np.ndarray, 
                               spins: np.ndarray) -> float:
        """
        Calculate the Ising energy for a given spin configuration.
        
        H(s) = sum_{i<j} J_{ij} * s_i * s_j + sum_i h_i * s_i
        
        Parameters:
        - J: Coupling matrix (NxN)
        - h: Local field vector (N,)
        - spins: Spin configuration (N,) in {-1, +1}
        
        Returns:
        - energy: Energy of the spin configuration
        """
        # Calculate the quadratic term
        quadratic_term = 0.0
        for i in range(len(spins)):
            for j in range(i+1, len(spins)):
                quadratic_term += J[i, j] * spins[i] * spins[j]
        
        # Calculate the linear term
        linear_term = np.dot(h, spins)
        
        return quadratic_term + linear_term