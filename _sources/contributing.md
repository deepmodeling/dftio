# Contributing to dftio

Thank you for your interest in contributing to dftio! We welcome contributions of all kinds, from bug fixes to new features.

## Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/deepmodeling/dftio.git
    cd dftio
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management. To install all required dependencies, including those for development and testing, run:
    ```bash
    uv sync --group dev
    ```

3.  **Run tests:**
    To make sure everything is set up correctly, run the test suite:
    ```bash
    uv run pytest -m "not integration"
    ```

## Code Style

-   Follow PEP 8 guidelines for Python code.
-   Use clear and meaningful names for variables, functions, and classes.
-   Add docstrings to all public functions and classes, explaining their purpose, arguments, and return values.

## Testing

-   All new features and bug fixes should be accompanied by tests.
-   Ensure that the full test suite passes before submitting a pull request.
-   Use `pytest` markers (e.g., `@pytest.mark.integration`) for tests that are slow or require external resources.

## Pull Request Process

1.  Fork the repository on GitHub.
2.  Create a new feature branch from the `main` branch.
3.  Make your changes in the new branch.
4.  Add or update tests as needed.
5.  Run the tests to ensure everything passes.
6.  Submit a pull request to the `main` branch of the original repository.

## Implementing a New Parser

If you are adding support for a new DFT package, please see the [Developer Guide](developer-guide.md) for a general overview. When implementing the parser class, you will need to provide several key methods. Below are the details of what each method should return.

### `get_structure(idx)`

This method should return a dictionary containing the atomic structure for the `idx`-th calculation. The dictionary should have the following keys (defined in `dftio.data._keys`):

-   `_keys.ATOMIC_NUMBERS_KEY`: Atomic numbers as a 1D tensor (`[natom]`).
-   `_keys.PBC_KEY`: Periodic boundary conditions as a boolean tensor (`[3]`).
-   `_keys.POSITIONS_KEY`: Atomic positions in Ångströms (`[nframe, natom, 3]`).
-   `_keys.CELL_KEY`: Lattice vectors in Ångströms (`[nframe, 3, 3]`).

### `get_eigenvalues(idx)`

This method should return a dictionary containing the eigenvalues and k-points:

-   `_keys.KPOINT_KEY`: K-point coordinates (`[nk, 3]`).
-   `_keys.ENERGY_EIGENVALUE_KEY`: Eigenvalues (`[nframe, nk, nband]`).

### `get_basis(idx)`

This method should return a dictionary describing the basis set, for example: `{"Si": "2s2p1d"}`.

### `get_blocks(idx, ...)`

This method should parse the real-space Hamiltonian, overlap, and/or density matrices. It should return a tuple of three lists: `(hamiltonians, overlaps, density_matrices)`. Each list should contain one dictionary per frame, where each dictionary's keys are strings like `"i_j_Rx_Ry_Rz"` (representing the matrix element between orbital `i` and orbital `j` in a neighboring cell at `(Rx, Ry, Rz)`) and the values are the corresponding matrix blocks as NumPy arrays.
