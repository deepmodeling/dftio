# Data Structures (`dftio.data`)

The `dftio.data` module provides the fundamental data structures for storing and managing atomic information and computational results. These structures are designed to be flexible, efficient, and compatible with machine learning workflows.

## `AtomicData`

The `dftio.data.AtomicData` class is the primary container for data associated with a single atomic structure. It is a dictionary-like object that stores properties of the system, such as atomic numbers, positions, and cell parameters, as PyTorch tensors.

### Key Features

- **Tensor-Based:** All data is stored as `torch.Tensor` objects, enabling seamless integration with PyTorch and other machine learning libraries.
- **Extensible:** You can add any custom data to an `AtomicData` object, allowing for flexible and detailed representations of your system.
- **Property-Based Access:** Accessing a key on an `AtomicData` object (e.g., `data['positions']`) returns the corresponding tensor.

### Core Properties

The following are some of the standard keys defined in `dftio.data._keys`:

- `cell`: The lattice vectors of the simulation cell.
- `positions`: The coordinates of each atom.
- `atomic_numbers`: The atomic number of each atom.
- `pbc` (Periodic Boundary Conditions): A boolean tensor indicating which directions are periodic.
- `eigs`: The eigenvalues of the electronic structure.
- `kpoints`: The coordinates of the k-points.

## `AtomicDataDict`

The `dftio.data.AtomicDataDict` class is a specialized dictionary designed to hold multiple `AtomicData` objects. It is the primary data structure returned by the parsers when processing multiple frames or structures.

### Key Features

- **Batching:** Provides methods for collating multiple `AtomicData` objects into a single batch for efficient processing.
- **Transformation Support:** Can be used with the `dftio.data.transforms` module to apply transformations to all `AtomicData` objects in the dictionary.
- **Serialization:** Can be saved to and loaded from disk, typically in `.dat` files (PyTorch's serialization format).

## Example Usage

```python
import torch
from dftio.data import AtomicData

# Create an AtomicData object for a simple system
data = AtomicData(
    positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    atomic_numbers=torch.tensor([14, 14]),  # Silicon
    cell=torch.eye(3) * 5.43,
    pbc=torch.tensor([True, True, True]),
)

# Access properties
print(data.positions)
print(data.cell)
```
