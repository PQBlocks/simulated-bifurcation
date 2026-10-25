import numpy as np
import pytest
from pqblocks_bifurcation.validation import validate_j_h, validate_num_sweeps, validate_initial_conditions


def test_validate_j_h_empty_model():
    """Test validation of empty model."""
    # Test with empty h
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match=r"Empty model: len\(h\) == 0 or J\.shape == \(0,0\)"):
        validate_j_h(J, h)
    
    # Test with (0,0) shaped J
    J = np.array([]).reshape(0, 0)
    h = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match=r"Empty model: len\(h\) == 0 or J\.shape == \(0,0\)"):
        validate_j_h(J, h)


def test_validate_j_h_dimension_mismatch():
    """Test validation of dimension mismatch."""
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0], dtype=np.float64)  # Wrong size
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_j_h(J, h)
    
    J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)  # Wrong size
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_j_h(J, h)


def test_validate_j_h_nan_inf_values():
    """Test validation of NaN/Inf values."""
    # Test with NaN in J
    J = np.array([[0, np.nan], [np.nan, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN or Inf values found in J or h"):
        validate_j_h(J, h)
    
    # Test with Inf in J
    J = np.array([[0, np.inf], [np.inf, 0]], dtype=np.float64)
    h = np.array([0, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN or Inf values found in J or h"):
        validate_j_h(J, h)
    
    # Test with NaN in h
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN or Inf values found in J or h"):
        validate_j_h(J, h)
    
    # Test with Inf in h
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN or Inf values found in J or h"):
        validate_j_h(J, h)


def test_validate_j_h_non_symmetric():
    """Test validation of non-symmetric J matrix."""
    J = np.array([[0, 1], [2, 0]], dtype=np.float64)  # Not symmetric
    h = np.array([0, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="J matrix must be symmetric"):
        validate_j_h(J, h)


def test_validate_j_h_non_zero_diagonal():
    """Test validation of non-zero diagonal in J matrix."""
    J = np.array([[1, 1], [1, 0]], dtype=np.float64)  # Non-zero diagonal
    h = np.array([0, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="J matrix must have zeros on diagonal"):
        validate_j_h(J, h)
    
    # Test with negative diagonal
    J = np.array([[-1, 1], [1, 0]], dtype=np.float64)  # Non-zero diagonal
    h = np.array([0, 0], dtype=np.float64)
    with pytest.raises(ValueError, match="J matrix must have zeros on diagonal"):
        validate_j_h(J, h)


def test_validate_j_h_valid_inputs():
    """Test validation with valid inputs."""
    # Valid case
    J = np.array([[0, 1, -1], [1, 0, 0], [-1, 0, 0]], dtype=np.float64)
    h = np.array([0, 0, 0], dtype=np.float64)
    # Should not raise any exception
    validate_j_h(J, h)
    
    # Another valid case with non-zero h
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([0.5, -0.5], dtype=np.float64)
    # Should not raise any exception
    validate_j_h(J, h)


def test_validate_num_sweeps_negative():
    """Test validation of negative num_sweeps."""
    with pytest.raises(ValueError, match="num_sweeps must be positive"):
        validate_num_sweeps(-1)
    
    with pytest.raises(ValueError, match="num_sweeps must be positive"):
        validate_num_sweeps(0)


def test_validate_num_sweeps_valid():
    """Test validation with valid num_sweeps."""
    # Should not raise any exception
    validate_num_sweeps(1)
    validate_num_sweeps(1000)


def test_validate_initial_conditions_valid():
    """Test validation of valid initial conditions."""
    N = 4
    x_init = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float64)
    y_init = np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float64)
    
    # Should not raise any exception
    validate_initial_conditions(x_init, y_init, N)
    
    # Test with None values
    validate_initial_conditions(None, None, N)
    validate_initial_conditions(x_init, None, N)
    validate_initial_conditions(None, y_init, N)


def test_validate_initial_conditions_invalid():
    """Test validation of invalid initial conditions."""
    N = 4
    x_init = np.array([0.1, -0.1, 0.1], dtype=np.float64)  # Wrong size
    y_init = np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float64)
    
    with pytest.raises(ValueError, match="x_init length"):
        validate_initial_conditions(x_init, y_init, N)
    
    x_init = np.array([0.1, -0.1, 0.1, -0.1], dtype=np.float64)
    y_init = np.array([0.2, -0.2, 0.2], dtype=np.float64)  # Wrong size
    
    with pytest.raises(ValueError, match="y_init length"):
        validate_initial_conditions(x_init, y_init, N)