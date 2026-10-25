import numpy as np
import pytest
from pqblocks_bifurcation.reference import simulate_sb
from pqblocks_bifurcation.kernel import simulate_sb_numba


def test_bsb_bit_identity():
    """Test that bSB kernel produces bit-identical results to reference implementation."""
    # Given
    J = np.array([
        [0, 1, 1, 1],
        [1, 0, -1, -1],
        [1, -1, 0, 1],
        [1, -1, 1, 0]
    ], dtype=np.float64)
    h = np.zeros(4, dtype=np.float64)
    
    # Initial conditions
    x_init = np.array([0.8, -0.8, 0.8, -0.8], dtype=np.float64)
    y_init = np.array([2.5, -2.5, 2.5, -2.5], dtype=np.float64)
    
    # Parameters
    dt = 0.5
    c0 = 0.25
    num_sweeps = 1
    seed = 12345
    
    # When - Reference implementation
    x_ref, y_ref, spins_ref = simulate_sb(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='bSB', seed=seed
    )
    
    # When - Numba kernel
    x_numba, y_numba, spins_numba = simulate_sb_numba(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='bSB', seed=seed
    )
    
    # Then - Check bit identity
    np.testing.assert_array_equal(x_ref, x_numba)
    np.testing.assert_array_equal(y_ref, y_numba)
    np.testing.assert_array_equal(spins_ref, spins_numba)


def test_dsb_bit_identity():
    """Test that dSB kernel produces bit-identical results to reference implementation."""
    # Given
    J = np.array([
        [0, 1, 1, 1],
        [1, 0, -1, -1],
        [1, -1, 0, 1],
        [1, -1, 1, 0]
    ], dtype=np.float64)
    h = np.zeros(4, dtype=np.float64)
    
    # Initial conditions
    x_init = np.array([0.8, -0.8, 0.8, -0.8], dtype=np.float64)
    y_init = np.array([2.5, -2.5, 2.5, -2.5], dtype=np.float64)
    
    # Parameters
    dt = 0.5
    c0 = 0.25
    num_sweeps = 1
    seed = 12345
    
    # When - Reference implementation
    x_ref, y_ref, spins_ref = simulate_sb(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='dSB', seed=seed
    )
    
    # When - Numba kernel
    x_numba, y_numba, spins_numba = simulate_sb_numba(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='dSB', seed=seed
    )
    
    # Then - Check bit identity
    np.testing.assert_array_equal(x_ref, x_numba)
    np.testing.assert_array_equal(y_ref, y_numba)
    np.testing.assert_array_equal(spins_ref, spins_numba)


def test_compute_c0_function():
    """Test the _compute_c0 function."""
    from pqblocks_bifurcation.kernel import _compute_c0
    
    # Test case 1: Zero matrix
    J_zero = np.zeros((4, 4), dtype=np.float64)
    c0_zero = _compute_c0(J_zero)
    # For zero matrix, we now use default scaling: 0.5 / sqrt(N)
    expected_c0_zero = 0.5 / np.sqrt(4)
    assert np.isclose(c0_zero, expected_c0_zero)
    
    # Test case 2: Matrix with non-zero elements
    J_nonzero = np.array([[0, 1, -1], [1, 0, 0], [-1, 0, 0]], dtype=np.float64)
    c0_nonzero = _compute_c0(J_nonzero)
    # For non-zero matrix, we use the standard formula
    # Extract off-diagonal elements: [1, -1, 1, 0, -1, 0] -> std ≈ 0.8165
    # sigma_J = np.std([1, -1, 1, 0, -1, 0]) = sqrt(2/3) ≈ 0.8165
    sigma_J = np.sqrt(2/3)
    expected_c0_nonzero = 0.5 / (sigma_J * np.sqrt(3))
    assert np.isclose(c0_nonzero, expected_c0_nonzero)
    
    # Test case 3: Single spin (1x1 matrix)
    J_single = np.zeros((1, 1), dtype=np.float64)
    c0_single = _compute_c0(J_single)
    # For single spin, we use default scaling: 0.5 / sqrt(1)
    expected_c0_single = 0.5 / np.sqrt(1)
    assert np.isclose(c0_single, expected_c0_single)


def test_initialize_xy_function():
    """Test the _initialize_xy function."""
    from pqblocks_bifurcation.kernel import _initialize_xy
    
    # Test deterministic initialization with seed
    N = 4
    seed1 = 12345
    seed2 = 67890
    
    x1, y1 = _initialize_xy(N, seed1)
    x2, y2 = _initialize_xy(N, seed2)
    x1_again, y1_again = _initialize_xy(N, seed1)
    
    # Same seed should produce same results
    np.testing.assert_array_equal(x1, x1_again)
    np.testing.assert_array_equal(y1, y1_again)
    
    # Different seeds should produce different results
    assert not np.array_equal(x1, x2) or not np.array_equal(y1, y2)


def test_multiple_sweeps_bit_identity():
    """Test bit identity over multiple sweeps."""
    # Given
    np.random.seed(42)
    N = 10
    J = np.random.randn(N, N).astype(np.float64)
    J = (J + J.T) / 2  # Make symmetric
    np.fill_diagonal(J, 0)  # Zero diagonal
    h = np.random.randn(N).astype(np.float64)
    
    # Initial conditions
    x_init = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
    y_init = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
    
    # Parameters
    dt = 0.5
    c0 = 0.25
    num_sweeps = 10
    seed = 99999
    
    # When - Reference implementation
    x_ref, y_ref, spins_ref = simulate_sb(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='bSB', seed=seed
    )
    
    # When - Numba kernel
    x_numba, y_numba, spins_numba = simulate_sb_numba(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='bSB', seed=seed
    )
    
    # Then - Check bit identity
    np.testing.assert_array_equal(x_ref, x_numba)
    np.testing.assert_array_equal(y_ref, y_numba)
    np.testing.assert_array_equal(spins_ref, spins_numba)


def test_edge_case_single_spin():
    """Test bit identity for single spin case."""
    # Given
    J = np.array([[0]], dtype=np.float64)
    h = np.array([0.5], dtype=np.float64)
    
    # Initial conditions
    x_init = np.array([0.05], dtype=np.float64)
    y_init = np.array([-0.05], dtype=np.float64)
    
    # Parameters
    dt = 0.5
    c0 = 0.0  # For single spin, sigma_J = 0, so c0 = 0
    num_sweeps = 5
    seed = 54321
    
    # When - Reference implementation
    x_ref, y_ref, spins_ref = simulate_sb(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='bSB', seed=seed
    )
    
    # When - Numba kernel
    x_numba, y_numba, spins_numba = simulate_sb_numba(
        J, h, dt, c0, num_sweeps, 
        x_init.copy(), y_init.copy(),
        mode='bSB', seed=seed
    )
    
    # Then - Check bit identity
    np.testing.assert_array_equal(x_ref, x_numba)
    np.testing.assert_array_equal(y_ref, y_numba)
    np.testing.assert_array_equal(spins_ref, spins_numba)