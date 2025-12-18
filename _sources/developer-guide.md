# Developer Guide

This guide is for developers who want to contribute to `dftio` by adding support for a new Density Functional Theory (DFT) software package.

## Project Structure

The core logic of `dftio` is organized into several key modules:

- **`dftio.data`**: Contains the fundamental data structures, such as `AtomicData` and `AtomicDataDict`, which are used to store and manage atomic information in a standardized format compatible with machine learning workflows.

- **`dftio.datastruct`**: Defines data structures for physical quantities like Hamiltonian matrices, overlap matrices, and fields.

- **`dftio.io`**: This is the I/O module, containing the interfaces to all supported DFT packages. Each package has its own submodule (e.g., `dftio.io.abacus`).

## How to Add a New Parser

Adding support for a new DFT code involves creating a new parser class and registering it with `dftio`. Here is a step-by-step guide:

### 1. Create a New Parser Module

Create a new directory under `dftio/io/` for your DFT package. For example, if you are adding a parser for a package named "NEWCODE," you would create the directory `dftio/io/newcode/`.

Inside this directory, create a Python file for your parser, e.g., `dftio/io/newcode/newcode_parser.py`.

### 2. Implement the Parser Class

In your new parser file, you will need to create a parser class that inherits from a base class (if one is available) or implements the necessary parsing methods. The key methods your parser will need to provide are:

- **`get_structure(idx)`**: This method should read the atomic structure information (atomic numbers, positions, cell, etc.) for a given calculation index (`idx`). It should return a dictionary of this data.

- **`get_eigenvalue(idx)`**: This method should parse the eigenvalues and k-points from the DFT output files.

- **`get_basis(idx)`**: This method should return information about the basis set used in the calculation.

- **`get_blocks(idx, hamiltonian, overlap, density_matrix)`**: This method is responsible for parsing the Hamiltonian, overlap, and/or density matrices.

### 3. Register Your Parser

To make your new parser available through the CLI and the `ParserRegister`, you need to add it to the registry in `dftio/io/parse.py`. Import your new parser class and add it to the `ParserRegister`.

```python
# In dftio/io/parse.py
from dftio.io.newcode.newcode_parser import NewCodeParser

# ...

ParserRegister.register("newcode", NewCodeParser)
```

### 4. Add a Test Case

To ensure your parser works correctly and to prevent future regressions, you should add a new test file in the `test/` directory. You will need to include example output files from your DFT code in the `test/data/` directory.

## Supported DFT Software

`dftio` currently has parsers for the following DFT packages:

- ABACUS
- Gaussian
- PYATB
- RESCU
- SIESTA
- VASP
