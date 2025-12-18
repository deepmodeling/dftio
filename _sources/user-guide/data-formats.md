# Output Data Formats

The `dftio parse` command can save the processed data in several formats, controlled by the `--format` (or `-f`) option. Each format has its own advantages depending on the use case.

## `dat` (Directory Format)

This is the default format (`-f dat`). It saves the parsed data into a well-organized directory structure, making it easy to inspect and use with simple scripts.

A new directory is created for each parsed calculation, named `{formula}.{index}` (e.g., `Si2.0`).

- **Structure Data:** Saved as plain text files.
  - `cell.dat`: Lattice vectors for each frame.
  - `positions.dat`: Atomic positions for each frame.
  - `atomic_numbers.dat`: A list of atomic numbers for the system.
  - `pbc.dat`: A boolean array indicating periodic boundary conditions (e.g., `[True, True, True]`).

- **Eigenvalue Data:** Saved as NumPy binary files (`.npy`).
  - `kpoints.npy`: The coordinates of each k-point.
  - `eigenvalues.npy`: A 3D array containing the eigenvalues for each frame, k-point, and band index.

- **Matrix Data (Hamiltonian, Overlap, etc.):** Saved in HDF5 format (`.h5`).
  - `hamiltonians.h5`: Contains the real-space Hamiltonian matrices.
  - `overlaps.h5`: Contains the overlap matrices.
  - `density_matrices.h5`: Contains the density matrices.
  Each HDF5 file is organized by frame number, with datasets inside corresponding to the matrix blocks (e.g., `0_0_0_0_0`).
  - `basis.dat`: A text file describing the atomic basis set used.

## `ase` (Atomic Simulation Environment) Format

This format (`-f ase`) is similar to `dat`, but it stores the structural information in a standard ASE trajectory file, which is useful for direct integration with the [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/).

- **Structure Data:**
  - `xdat.traj`: An ASE trajectory file containing the structure (atoms, positions, cell) for every frame.

- **Eigenvalue and Matrix Data:**
  - All other data (eigenvalues, matrices) are saved in the same way as the `dat` format, using `.npy` and `.h5` files within the same output directory.

## `lmdb` (Lightning DB) Format

This format (`-f lmdb`) is designed for high-performance applications where quick access to individual data frames from a very large dataset is required. It stores all information in a single binary database file.

- **Database File:**
  - A single file named `data.{pid}.lmdb` is created, where `{pid}` is the process ID.

- **Structure:**
  - The database contains key-value pairs. Each value is a [pickled](https://docs.python.org/3/library/pickle.html) Python dictionary that represents a single frame of the calculation.
  - Each dictionary contains all data for that frame: `positions`, `cell`, `atomic_numbers`, `eigenvalues`, `hamiltonian`, etc.
  - This structure avoids having to manage thousands of individual files and can significantly speed up data loading.