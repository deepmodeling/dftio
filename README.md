# dftio Developer Guide
dftio is to assist machine learning communities to transcript DFT output into a format that is easy to read or used by machine learning models.

# How to use
```bash
usage: dftio parse [-h] [-ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-lp LOG_PATH] [-m MODE] [-r ROOT] [-p PREFIX] [-o OUTROOT] [-f FORMAT] [-ham] [-ovp] [-dm] [-eig]

optional arguments:
  -h, --help            show this help message and exit
  -ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not specified, the logs will only be output to console (default: None)
  -m MODE, --mode MODE  The name of the DFT software. (default: abacus)
  -r ROOT, --root ROOT  The root directory of the DFT files. (default: ./)
  -p PREFIX, --prefix PREFIX
                        The prefix of the DFT files under root. (default: frame)
  -o OUTROOT, --outroot OUTROOT
                        The output root directory. (default: ./)
  -f FORMAT, --format FORMAT
                        The output root directory. (default: dat)
  -ham, --hamiltonian   Whether to parse the Hamiltonian matrix. (default: False)
  -ovp, --overlap       Whether to parse the Overlap matrix (default: False)
  -dm, --density_matrix
                        Whether to parse the Density matrix (default: False)
  -eig, --eigenvalue    Whether to parse the kpoints and eigenvalues (default: False)
```

# Package Structure
The main structure of dftio contrains three module:

`data`: Containing the basic graph and dataset / dataloader implemented align with pytorch-geometric convension. By using the dataset and graph class provided, user only need to concentrate on building powerful machine learning models.

`datastruct`: This module contrains the data structure class for the physical quantities that is supported by dftio. For example, the hamiltonian/overlap/density matrix, and the field quantities such as charge density distribution function, potential function and so on.

`io`: IO class is the interfaces of dftio class to all DFT packages. Each DFT that dftio supported cooresponding to a submodule in `io`. Developer only need to implement several parsing functions in each submodule, and summit them by inheriting the parser class and register into the parser collection class. NOTE: for further support of parallization, parsing each DFT snapshot need to be independent.

```
|-- dftio
|   |-- __init__.py
|   |-- __main__.py
|   |-- constants.py
|   |-- data
|   |   `-- _keys.py
|   |-- datastruct
|   |-- io
|   |   |-- __init__.py
|   |   |-- abacus
|   |   |   `-- abacus_parser.py
|   |   |-- parse.py
|   |   `-- rescu
|   |       `-- rescu_parser.py
|   |-- logger.py
|   |-- register.py
|   `-- utils.py
|-- example
|-- pyproject.toml
`-- test
```

# What need to write:
in `io` class, you need to create a submodule for your own DFT package, and implement the following method:

1. get_structure(idx: int): idx is the index of ith target DFT output folder. the output need to be a dict of structure data, containing:
    - _keys.ATOMIC_NUMBERS_KEY: atomic number, shape [natom,]
    - _keys.PBC_KEY: periodic boundary condition, shape [3,]
    - _keys.POSITIONS_KEY: position, unit in Angstrum shape [nframe, natom, 3]
    - _keys.CELL_KEY: cell, unit in Bohr2Angstrom, shape [nframe, 3, 3]
2. get_eigenvalues(idx: int): output also need to be a key, containing:
    - _keys.KPOINT_KEY: kpoints, shape [natom, nband]
    - _keys.ENERGY_EIGENVALUE_KEY: eigenvalues, shape [nframe, natom, nband]
3. get_basis(idx: int): This gives out the basis information used in DFT calculation, the format looks like: {Si: "2s2p1d"}
4. get_blocks(idx: int, hamiltonian: bool=False, overlap: bool=False, density_matrix: bool=False): This function contrains the hamiltonian/overlap/density matrix block data. The returning format is a tuple of three quantities, as ([hamiltonian], [overlap], [density_matrix]). each [...] denote a list of dict, which length equals nframe. Each dict records the blocks data with a structure: {"i_j_Rx_Ry_Rz": np.array(...)}

A example should be added into test/data for further testing

# Supported Physical Quantities：
1. atomic structure
2. field - p(r) V_h(r) V_xc(r) (x,y,z) - f(x,y,z)
    - p(r) - LCAO (C_i)
3. Operator under local basis: O(r,r') - H(r,r'), P(r,r), S(r,r') (i,j,R) -> []
4. Projection of scalar field on local basis
5. kpoint eigenvalue

# Supported DFT software
## done
1. RESCU
2. ABACUS
3. SIESTA

## ing
3. SIESTA - Jijie
4. Gaussian - zixi
5. PYSCF - feitong

## prospective
6. PSI4
7. VASP
8. Wannier

# Functions
1. rotation base class of Operator
2. dtype transform
3. output visualizable structure file (.vasp, .cif) to parse base class
4. key_map
