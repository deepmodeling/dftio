import pytest
import torch
import numpy as np
from dftio.op.grid_int import SingleGridIntegrator
from dftio.datastruct import AtomicBasis
import ase.data as data

try:
    import torch_scatter  # noqa: F401
    _SCATTER_AVAILABLE = True
except ImportError:
    _SCATTER_AVAILABLE = False

needs_scatter = pytest.mark.skipif(not _SCATTER_AVAILABLE, reason="torch-scatter not installed")

class MockAtomicBasis:
    def __init__(self, atomic_numbers):
        self.atomic_numbers = atomic_numbers
        self.rcut = 5.0
        self.irreps = type('obj', (object,), {'dim': 1})
    
    def __getitem__(self, key):
        return self
        
    def __call__(self, rel_pos):
        # Return dummy values
        return torch.ones(rel_pos.shape[0], 1)

@pytest.fixture
def mock_atomic_basis():
    return MockAtomicBasis([1])

def test_single_grid_integrator_init(mock_atomic_basis):
    """Test SingleGridIntegrator initialization."""
    atomic_numbers = [1]
    pbc = [True, True, True]
    cell = np.eye(3)
    coordinates = np.array([[0.0, 0.0, 0.0]])
    grids = np.array([[0.1, 0.1, 0.1]])
    
    sgi = SingleGridIntegrator(
        atomic_numbers=atomic_numbers,
        pbc=pbc,
        cell=cell,
        coordinates=coordinates,
        grids=grids,
        atomic_basis={'H': mock_atomic_basis}
    )
    
    assert sgi.atomic_numbers.shape == (1,)
    assert sgi.cell.shape == (3, 3)
    assert sgi.coordinates.shape == (1, 3)
    assert sgi.grids.shape == (1, 3)

@needs_scatter
def test_integrate(mock_atomic_basis):
    """Test integrate method."""
    atomic_numbers = [1]
    pbc = [True, True, True]
    cell = np.eye(3)
    coordinates = np.array([[0.0, 0.0, 0.0]])
    grids = np.array([[0.1, 0.1, 0.1]])
    
    sgi = SingleGridIntegrator(
        atomic_numbers=atomic_numbers,
        pbc=pbc,
        cell=cell,
        coordinates=coordinates,
        grids=grids,
        atomic_basis={'H': mock_atomic_basis}
    )
    
    # Test without weights
    result = sgi.integrate()
    assert result.shape == (1,)
    
    # Test with weights
    weights = torch.tensor([1.0])
    result_w = sgi.integrate(weights=weights)
    assert result_w.shape == (1,)
