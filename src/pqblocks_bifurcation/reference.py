import numpy as np


def simulate_sb(J, h, dt, c0, num_sweeps, x_init=None, y_init=None,
                mode='bSB', seed=None):
    """
    Simulate the Simulated Bifurcation (SB) algorithm.

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
        if seed is not None:
            np.random.seed(seed)
        x = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
        y = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
    else:
        x = x_init.astype(np.float64)
        y = y_init.astype(np.float64)

    # Main simulation loop
    for step in range(num_sweeps):
        # Pump parameter
        a = (step + 1) / num_sweeps

        # Compute local fields
        if mode == 'bSB':
            op_x = x  # For ballistic SB, op(x) = x
        elif mode == 'dSB':
            op_x = np.sign(x)  # For discrete SB, op(x) = sign(x)
            # Handle case where x[i] = 0 -> sign(0) = 0, but we want +1
            op_x = np.where(op_x == 0, 1, op_x)
        else:
            raise ValueError("mode must be 'bSB' or 'dSB'")

        # f_i(x) = sum_j J^SB_ij * op(x_j) + h^SB_i
        f = J_SB @ op_x + h_SB

        # Update y
        y += (-(1 - a) * x + c0 * f) * dt

        # Update x
        x += y * dt

        # Apply inelastic boundary walls
        wall_hits = np.abs(x) > 1.0
        x = np.where(wall_hits, np.sign(x), x)
        y = np.where(wall_hits, 0.0, y)

    # Final spins
    spins = np.sign(x)
    # Handle case where x[i] = 0 -> sign(0) = 0, but we map to +1
    spins = np.where(spins == 0, 1, spins).astype(np.int8)

    return x, y, spins
