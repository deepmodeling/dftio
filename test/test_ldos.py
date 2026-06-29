import pytest
import torch
import numpy as np
from dftio.calc.ldos import LDOS
from dftio.datastruct import AtomicBasis

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
        return torch.ones(rel_pos.shape[0], 1)

@pytest.fixture
def mock_atomic_basis():
    return MockAtomicBasis([1])

def test_ldos_init(mock_atomic_basis):
    """Test LDOS initialization."""
    atomic_numbers = [1]
    pbc = [True, True, True]
    cell = np.eye(3)
    coordinates = np.array([[0.0, 0.0, 0.0]])
    grids = np.array([[0.1, 0.1, 0.1]])
    
    ldos = LDOS(
        atomicbasis={'H': mock_atomic_basis},
        atomic_numbers=atomic_numbers,
        pbc=pbc,
        cell=cell,
        coordinates=coordinates,
        grids=grids
    )
    
    assert ldos.natoms == 1
    assert ldos.nspin == 2

@needs_scatter
def test_ldos_get(mock_atomic_basis):
    """Test LDOS get method."""
    atomic_numbers = [1]
    pbc = [True, True, True]
    cell = np.eye(3)
    coordinates = np.array([[0.0, 0.0, 0.0]])
    grids = np.array([[0.1, 0.1, 0.1]])
    
    ldos = LDOS(
        atomicbasis={'H': mock_atomic_basis},
        atomic_numbers=atomic_numbers,
        pbc=pbc,
        cell=cell,
        coordinates=coordinates,
        grids=grids
    )
    
    # Mock coefficients and eigenvalues
    # coefficients: [nk, nbands, norbs]
    # eigenvalues: [nk, nbands]
    nk, nbands, norbs = 1, 1, 1
    coefficients = torch.ones(nk, nbands, norbs)
    eigenvalues = torch.zeros(nk, nbands)
    
    result = ldos.get(E=0.0, coefficients=coefficients, eigenvalues=eigenvalues)
    
    assert result.shape == (1,)
