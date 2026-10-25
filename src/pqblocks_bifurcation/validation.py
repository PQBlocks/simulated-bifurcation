import numpy as np


def validate_j_h(J, h):
    """
    Validate the J matrix and h vector according to the Edge-Case Ledger.
    
    Parameters:
    - J: Coupling matrix (NxN)
    - h: Local field vector (N,)
    
    Raises:
    - ValueError: If any validation fails
    """
    # 1. Empty/N=0 check
    if len(h) == 0 or J.shape == (0, 0):
        raise ValueError("Empty model: len(h) == 0 or J.shape == (0,0)")
    
    # 2. Dimension mismatch check
    if len(h) != J.shape[0] or J.shape[0] != J.shape[1]:
        raise ValueError(
            "Dimension mismatch: len(h) != J.shape[0] or "
            "J.shape[0] != J.shape[1]"
        )
    
    # 3. NaN / Inf check
    if (np.any(np.isnan(J)) or np.any(np.isinf(J)) or
            np.any(np.isnan(h)) or np.any(np.isinf(h))):
        raise ValueError("NaN or Inf values found in J or h")
    
    # 4. Non-symmetric J check
    if not np.allclose(J, J.T):
        raise ValueError("J matrix must be symmetric")
    
    # 5. Non-zero diagonal check
    if np.any(np.diag(J) != 0):
        raise ValueError("J matrix must have zeros on diagonal")


def validate_num_sweeps(num_sweeps):
    """
    Validate the num_sweeps parameter.
    
    Parameters:
    - num_sweeps: Number of simulation steps
    
    Raises:
    - ValueError: If validation fails
    """
    # 7. Negative/Zero Sweeps check
    if num_sweeps <= 0:
        raise ValueError("num_sweeps must be positive")


def validate_initial_conditions(x_init, y_init, N):
    """
    Validate initial conditions if provided.
    
    Parameters:
    - x_init: Initial x values (N,) or None
    - y_init: Initial y values (N,) or None
    - N: Expected size
    
    Raises:
    - ValueError: If validation fails
    """
    if x_init is not None:
        if len(x_init) != N:
            raise ValueError(f"x_init length ({len(x_init)}) must match system size ({N})")
    
    if y_init is not None:
        if len(y_init) != N:
            raise ValueError(f"y_init length ({len(y_init)}) must match system size ({N})")