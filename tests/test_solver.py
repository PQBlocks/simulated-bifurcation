import numpy as np
import pytest
from pqblocks_bifurcation.solver import SimulatedBifurcationSampler


def test_simulated_bifurcation_sampler_initialization():
    """Test initialization of SimulatedBifurcationSampler."""
    # Test default initialization
    sampler = SimulatedBifurcationSampler()
    assert sampler.dt == 0.5
    assert sampler.num_sweeps == 1000
    assert sampler.mode == 'dSB'  # Changed default mode to dSB
    assert sampler.seed is None
    
    # Test custom initialization
    sampler = SimulatedBifurcationSampler(dt=0.1, num_sweeps=500, mode='bSB', seed=42)  # Explicitly test bSB
    assert sampler.dt == 0.1
    assert sampler.num_sweeps == 500
    assert sampler.mode == 'bSB'
    assert sampler.seed == 42


def test_simulated_bifurcation_sampler_dt_validation():
    """Test dt validation in SimulatedBifurcationSampler initialization."""
    # Test negative dt
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        SimulatedBifurcationSampler(dt=-0.5)
    
    # Test zero dt
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        SimulatedBifurcationSampler(dt=0)
    
    # Test infinite dt
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        SimulatedBifurcationSampler(dt=np.inf)
    
    # Test NaN dt
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        SimulatedBifurcationSampler(dt=np.nan)


def test_simulated_bifurcation_sampler_sample_ising_valid():
    """Test sample_ising with valid inputs."""
    # Create a simple 2-spin problem
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(num_sweeps=10, seed=42)
    spins, energy = sampler.sample_ising(J, h)
    
    # Check that spins are valid
    assert spins.shape == (2,)
    assert np.all((spins == 1) | (spins == -1))
    
    # Check that energy is a float
    assert isinstance(energy, float)


def test_simulated_bifurcation_sampler_sample_ising_bsb_vs_dsb():
    """Test that bSB and dSB produce valid outputs."""
    # Create a simple 2-spin problem
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    
    # Test bSB
    sampler_bsb = SimulatedBifurcationSampler(mode='bSB', num_sweeps=10, seed=42)
    spins_bsb, energy_bsb = sampler_bsb.sample_ising(J, h)
    assert spins_bsb.shape == (2,)
    assert np.all((spins_bsb == 1) | (spins_bsb == -1))
    
    # Test dSB
    sampler_dsb = SimulatedBifurcationSampler(mode='dSB', num_sweeps=10, seed=42)
    spins_dsb, energy_dsb = sampler_dsb.sample_ising(J, h)
    assert spins_dsb.shape == (2,)
    assert np.all((spins_dsb == 1) | (spins_dsb == -1))


def test_simulated_bifurcation_sampler_sample_ising_with_initial_conditions():
    """Test sample_ising with custom initial conditions."""
    # Create a simple 2-spin problem
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    
    # Define initial conditions
    x_init = np.array([0.05, -0.05], dtype=np.float64)
    y_init = np.array([-0.05, 0.05], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(num_sweeps=10, seed=42)
    spins, energy = sampler.sample_ising(J, h, x_init, y_init)
    
    # Check that spins are valid
    assert spins.shape == (2,)
    assert np.all((spins == 1) | (spins == -1))


def test_simulated_bifurcation_sampler_invalid_mode():
    """Test sample_ising with invalid mode."""
    # Create a simple 2-spin problem
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(mode='invalid', num_sweeps=10)
    with pytest.raises(ValueError, match="mode must be 'bSB' or 'dSB'"):
        sampler.sample_ising(J, h)


def test_simulated_bifurcation_sampler_empty_model():
    """Test sample_ising with empty model."""
    J = np.array([]).reshape(0, 0)
    h = np.array([], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler()
    with pytest.raises(ValueError, match=r"Empty model: len\(h\) == 0 or J\.shape == \(0,0\)"):
        sampler.sample_ising(J, h)


def test_simulated_bifurcation_sampler_nan_inf_inputs():
    """Test sample_ising with NaN/Inf inputs."""
    # Test with NaN in J
    J = np.array([[0, np.nan], [np.nan, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler()
    with pytest.raises(ValueError, match="NaN or Inf values found in J or h"):
        sampler.sample_ising(J, h)
    
    # Test with Inf in h
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, np.inf], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler()
    with pytest.raises(ValueError, match="NaN or Inf values found in J or h"):
        sampler.sample_ising(J, h)


def test_simulated_bifurcation_sampler_non_symmetric_j():
    """Test sample_ising with non-symmetric J matrix (should be symmetrized)."""
    J = np.array([[0, 1], [2, 0]], dtype=np.float64)  # Not symmetric
    h = np.array([0, 0], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler()
    # Should not raise ValueError anymore as we symmetrize the matrix
    spins, energy = sampler.sample_ising(J, h)
    
    # Check that spins are valid
    assert spins.shape == (2,)
    assert np.all((spins == 1) | (spins == -1))


def test_simulated_bifurcation_sampler_non_zero_diagonal():
    """Test sample_ising with non-zero diagonal in J matrix (should be zeroed)."""
    J = np.array([[1, 1], [1, 0]], dtype=np.float64)  # Non-zero diagonal
    h = np.array([0, 0], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler()
    # Should not raise ValueError anymore as we zero the diagonal
    spins, energy = sampler.sample_ising(J, h)
    
    # Check that spins are valid
    assert spins.shape == (2,)
    assert np.all((spins == 1) | (spins == -1))


def test_simulated_bifurcation_sampler_invalid_initial_conditions():
    """Test sample_ising with invalid initial conditions."""
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    
    # Wrong size x_init
    x_init = np.array([0.1], dtype=np.float64)  # Should be size 2
    y_init = np.array([0.1, -0.1], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler()
    with pytest.raises(ValueError, match="x_init length"):
        sampler.sample_ising(J, h, x_init, y_init)
    
    # Wrong size y_init
    x_init = np.array([0.1, -0.1], dtype=np.float64)
    y_init = np.array([0.1], dtype=np.float64)  # Should be size 2
    
    sampler = SimulatedBifurcationSampler()
    with pytest.raises(ValueError, match="y_init length"):
        sampler.sample_ising(J, h, x_init, y_init)


def test_simulated_bifurcation_sampler_single_spin():
    """Test sample_ising with single spin."""
    J = np.array([[0]], dtype=np.float64)
    h = np.array([0.5], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(num_sweeps=10, seed=42)
    spins, energy = sampler.sample_ising(J, h)
    
    # Check that spins are valid
    assert spins.shape == (1,)
    assert np.all((spins == 1) | (spins == -1))
    
    # Check energy calculation for single spin
    expected_energy = h[0] * spins[0]  # No quadratic term for single spin
    assert np.isclose(energy, expected_energy)


def test_simulated_bifurcation_sampler_decoupled_model():
    """Test sample_ising with decoupled model (all-zero J matrix)."""
    # N=1, h=[1.5] should yield spin [-1] (energy -1.5) as it minimizes energy
    J = np.zeros((1, 1), dtype=np.float64)
    h = np.array([1.5], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(num_sweeps=100, seed=42)
    spins, energy = sampler.sample_ising(J, h)
    
    # Check that spins are valid
    assert spins.shape == (1,)
    assert np.all((spins == 1) | (spins == -1))
    
    # For positive h, the minimum energy state is spin = -1 (energy = -1.5)
    # For negative h, the minimum energy state is spin = +1 (energy = -|h|)
    expected_energy = h[0] * spins[0]
    assert np.isclose(energy, expected_energy)
    
    # Test with negative h
    h_negative = np.array([-1.5], dtype=np.float64)
    spins_neg, energy_neg = sampler.sample_ising(J, h_negative)
    
    # Check that spins are valid
    assert spins_neg.shape == (1,)
    assert np.all((spins_neg == 1) | (spins_neg == -1))
    
    # For negative h, the minimum energy state is spin = +1
    expected_energy_neg = h_negative[0] * spins_neg[0]
    assert np.isclose(energy_neg, expected_energy_neg)


def test_simulated_bifurcation_sampler_multiple_decoupled_spins():
    """Test sample_ising with multiple decoupled spins."""
    # 3 decoupled spins with different local fields
    N = 3
    J = np.zeros((N, N), dtype=np.float64)
    h = np.array([1.0, -2.0, 0.5], dtype=np.float64)  # Different local fields
    
    sampler = SimulatedBifurcationSampler(num_sweeps=100, seed=42)
    spins, energy = sampler.sample_ising(J, h)
    
    # Check that spins are valid
    assert spins.shape == (N,)
    assert np.all((spins == 1) | (spins == -1))
    
    # Check energy calculation for decoupled spins
    expected_energy = np.dot(h, spins)
    assert np.isclose(energy, expected_energy)


def test_simulated_bifurcation_sampler_asymmetric_matrix():
    """Test sample_ising with asymmetric J matrix (should be symmetrized)."""
    # Create an asymmetric matrix that should be symmetrized
    J_asym = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)  # Not symmetric
    h = np.array([0, 0], dtype=np.float64)
    
    # The symmetrized matrix should be [[0.0, 1.5], [1.5, 0.0]]
    J_sym_expected = np.array([[0.0, 1.5], [1.5, 0.0]], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(num_sweeps=10, seed=42)
    spins, energy = sampler.sample_ising(J_asym, h)
    
    # Check that spins are valid
    assert spins.shape == (2,)
    assert np.all((spins == 1) | (spins == -1))
    
    # Check that energy is calculated correctly with the symmetrized matrix
    expected_energy = J_sym_expected[0, 1] * spins[0] * spins[1] + h[0] * spins[0] + h[1] * spins[1]
    assert np.isclose(energy, expected_energy)


def test_simulated_bifurcation_sampler_nonzero_diagonal():
    """Test sample_ising with non-zero diagonal in J matrix (should be zeroed)."""
    # Create a matrix with non-zero diagonal that should be zeroed
    J_diag = np.array([[1.0, 1.0], [1.0, 2.0]], dtype=np.float64)  # Non-zero diagonal
    h = np.array([0, 0], dtype=np.float64)
    
    # The matrix with zeroed diagonal should be [[0.0, 1.0], [1.0, 0.0]]
    J_zero_diag_expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(num_sweeps=10, seed=42)
    spins, energy = sampler.sample_ising(J_diag, h)
    
    # Check that spins are valid
    assert spins.shape == (2,)
    assert np.all((spins == 1) | (spins == -1))
    
    # Check that energy is calculated correctly with the zeroed diagonal matrix
    expected_energy = J_zero_diag_expected[0, 1] * spins[0] * spins[1] + h[0] * spins[0] + h[1] * spins[1]
    assert np.isclose(energy, expected_energy)


def test_simulated_bifurcation_sampler_asymmetric_and_diagonal():
    """Test sample_ising with both asymmetric matrix and non-zero diagonal."""
    # Create a matrix that is both asymmetric and has non-zero diagonal
    J_mixed = np.array([[1.0, 2.0], [4.0, 3.0]], dtype=np.float64)
    h = np.array([0.5, -0.5], dtype=np.float64)
    
    # Should be symmetrized and diagonal zeroed: [[0.0, 3.0], [3.0, 0.0]]
    J_expected = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float64)
    
    sampler = SimulatedBifurcationSampler(num_sweeps=10, seed=42)
    spins, energy = sampler.sample_ising(J_mixed, h)
    
    # Check that spins are valid
    assert spins.shape == (2,)
    assert np.all((spins == 1) | (spins == -1))
    
    # Check that energy is calculated correctly
    expected_energy = (J_expected[0, 1] * spins[0] * spins[1] + 
                      h[0] * spins[0] + h[1] * spins[1])
    assert np.isclose(energy, expected_energy)