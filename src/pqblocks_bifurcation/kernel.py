import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(fastmath=False)
def _compute_c0(J_SB: np.ndarray) -> float:
    """
    Compute the scaling constant c0 based on the standard deviation of J^SB.
    
    Parameters:
    - J_SB: Coupling matrix (NxN), symmetric, zeros on diagonal
    
    Returns:
    - c0: Scaling constant
    """
    N = J_SB.shape[0]
    if N == 0:
        return 0.0
    
    # Compute standard deviation of J^SB (off-diagonal elements only)
    # Since J_SB is symmetric and zeros on diagonal, we can just take all elements
    # and compute std, but we should exclude diagonal elements
    # Flatten the matrix and exclude diagonal elements
    flat_J = J_SB.flatten()
    # Create a mask to exclude diagonal elements
    # For an NxN matrix, diagonal elements are at indices [0, N+1, 2*(N+1), ..., N*(N+1)]
    N_total = N * N
    diag_indices = np.arange(0, N_total, N+1)
    mask = np.ones(N_total, dtype=np.bool_)
    mask[diag_indices] = False
    off_diag_elements = flat_J[mask]
    
    if len(off_diag_elements) == 0:
        sigma_J = 0.0
    else:
        sigma_J = np.std(off_diag_elements)
    
    # If sigma_J is zero (e.g., all elements are the same), set c0 = 0
    # But we still need to ensure local fields drive the dynamics
    if sigma_J < 1e-12:
        # When sigma_J is zero, we still need to maintain proper scaling
        # We'll use a default scaling that ensures local fields are effective
        # For decoupled models, we want c0 to be non-zero to allow h to drive dynamics
        return 0.5 / np.sqrt(N)  # Default scaling when coupling matrix is zero
    
    c0 = 0.5 / (sigma_J * np.sqrt(N))
    return c0


@njit(fastmath=False)
def _initialize_xy(N: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize x and y arrays with random values in [-0.1, 0.1].
    
    Parameters:
    - N: Number of spins
    - seed: Random seed
    
    Returns:
    - x: Initial x values (N,)
    - y: Initial y values (N,)
    """
    # Use SplitMix64-like approach for seeding
    # This is a simple approach for generating deterministic sequences
    # In a real implementation, we would use a proper SplitMix64 implementation
    # For now, we'll use a simple linear congruential generator with the seed
    # to generate the initial values
    
    # Initialize with seed
    np.random.seed(seed)
    x = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
    y = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
    return x, y


@njit(fastmath=False)
def _simulate_bsb_kernel(J_SB: np.ndarray, h_SB: np.ndarray, dt: float, c0: float, 
                         num_sweeps: int, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the ballistic SB (bSB) kernel.
    
    Parameters:
    - J_SB: Coupling matrix (NxN), symmetric, zeros on diagonal
    - h_SB: Local field vector (N,)
    - dt: Time step
    - c0: Scaling constant
    - num_sweeps: Number of simulation steps
    - x: Initial x values (N,)
    - y: Initial y values (N,)
    
    Returns:
    - x: Final x values (N,)
    - y: Final y values (N,)
    """
    N = len(h_SB)
    
    # Main simulation loop
    for step in range(num_sweeps):
        # Pump parameter
        a = (step + 1) / num_sweeps
        
        # For ballistic SB, op(x) = x
        op_x = x.copy()  # Copy to avoid modifying x during computation
        
        # f_i(x) = sum_j J^SB_ij * op(x_j) + h^SB_i
        # Using matrix multiplication for efficiency
        f = J_SB @ op_x + h_SB
        
        # Update y: y += (-(1 - a) * x + c0 * f) * dt
        # Compute in steps to ensure precision
        term1 = -(1 - a) * x
        term2 = c0 * f
        y += (term1 + term2) * dt
        
        # Update x: x += y * dt
        x += y * dt
        
        # Apply inelastic boundary walls
        for i in range(N):
            if np.abs(x[i]) > 1.0:
                x[i] = np.sign(x[i])
                y[i] = 0.0
    
    return x, y


@njit(fastmath=False)
def _simulate_dsb_kernel(J_SB: np.ndarray, h_SB: np.ndarray, dt: float, c0: float, 
                         num_sweeps: int, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the discrete SB (dSB) kernel.
    
    Parameters:
    - J_SB: Coupling matrix (NxN), symmetric, zeros on diagonal
    - h_SB: Local field vector (N,)
    - dt: Time step
    - c0: Scaling constant
    - num_sweeps: Number of simulation steps
    - x: Initial x values (N,)
    - y: Initial y values (N,)
    
    Returns:
    - x: Final x values (N,)
    - y: Final y values (N,)
    """
    N = len(h_SB)
    
    # Main simulation loop
    for step in range(num_sweeps):
        # Pump parameter
        a = (step + 1) / num_sweeps
        
        # For discrete SB, op(x) = sign(x)
        op_x = np.sign(x)
        # Handle case where x[i] = 0 -> sign(0) = 0, but we want +1
        for i in range(N):
            if op_x[i] == 0:
                op_x[i] = 1
        
        # f_i(x) = sum_j J^SB_ij * op(x_j) + h^SB_i
        # Using matrix multiplication for efficiency
        f = J_SB @ op_x + h_SB
        
        # Update y: y += (-(1 - a) * x + c0 * f) * dt
        # Compute in steps to ensure precision
        term1 = -(1 - a) * x
        term2 = c0 * f
        y += (term1 + term2) * dt
        
        # Update x: x += y * dt
        x += y * dt
        
        # Apply inelastic boundary walls
        for i in range(N):
            if np.abs(x[i]) > 1.0:
                x[i] = np.sign(x[i])
                y[i] = 0.0
    
    return x, y


def simulate_sb_numba(J, h, dt, c0, num_sweeps, x_init=None, y_init=None,
                      mode='bSB', seed=None):
    """
    Simulate the Simulated Bifurcation (SB) algorithm using Numba JIT compilation.
    
    Parameters:
    - J: Coupling matrix (NxN), symmetric, zeros on diagonal
    - h: Local field vector (N,)
    - dt: Time step
    - c0: Scaling constant
    - num_sweeps: Number of simulation steps
    - x_init: Initial x values (N,), if None will be randomly initialized
    - y_init: Initial y values (N,), if None will be randomly initialized
    - mode: 'bSB' for ballistic SB, 'dSB' for discrete SB
    - seed: Random seed for initialization
    
    Returns:
    - x_final: Final x values (N,)
    - y_final: Final y values (N,)
    - spins: Final spin states (N,) in {-1, +1}
    """
    # Validate inputs according to Edge-Case Ledger
    if len(h) == 0 or J.shape == (0, 0):
        raise ValueError("Empty model: len(h) == 0 or J.shape == (0,0)")
    
    if len(h) != J.shape[0] or J.shape[0] != J.shape[1]:
        raise ValueError(
            "Dimension mismatch: len(h) != J.shape[0] or "
            "J.shape[0] != J.shape[1]"
        )
    
    if not np.allclose(J, J.T):
        raise ValueError("J matrix must be symmetric")
    
    if np.any(np.diag(J) != 0):
        raise ValueError("J matrix must have zeros on diagonal")
    
    if num_sweeps <= 0:
        raise ValueError("num_sweeps must be positive")
    
    # Check for NaN/Inf
    if (np.any(np.isnan(J)) or np.any(np.isinf(J)) or
            np.any(np.isnan(h)) or np.any(np.isinf(h))):
        raise ValueError("NaN or Inf values found in J or h")
    
    N = len(h)
    
    # Apply sign conventions
    J_SB = -J  # J^{SB} = -J
    h_SB = -h  # h^{SB} = -h
    
    # Initialize x and y if not provided
    if x_init is None or y_init is None:
        if seed is None:
            # Use a default seed if none provided
            seed = 42
        x, y = _initialize_xy(N, seed)
    else:
        x = x_init.astype(np.float64)
        y = y_init.astype(np.float64)
    
    # Run the appropriate kernel
    if mode == 'bSB':
        x, y = _simulate_bsb_kernel(J_SB, h_SB, dt, c0, num_sweeps, x, y)
    elif mode == 'dSB':
        x, y = _simulate_dsb_kernel(J_SB, h_SB, dt, c0, num_sweeps, x, y)
    else:
        raise ValueError("mode must be 'bSB' or 'dSB'")
    
    # Final spins
    spins = np.sign(x)
    # Handle case where x[i] = 0 -> sign(0) = 0, but we map to +1
    spins = np.where(spins == 0, 1, spins).astype(np.int8)
    
    return x, y, spins