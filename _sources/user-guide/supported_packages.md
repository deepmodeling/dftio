# Supported DFT Packages

`dftio` supports a variety of DFT software packages. The level of support for each package (i.e., which data types can be parsed) is summarized below.

## Summary of Support

| Package  | Structure | Eigenvalues | Hamiltonian | Density Matrix | Overlap Matrix |
|:--------:|:---------:|:-----------:|:-----------:|:--------------:|:--------------:|
|  ABACUS  |     ✅     |      ✅      |      ✅      |       ✅        |       ✅        |
|  RESCU   |     ✅     |      —      |      ✅      |       —        |       ✅        |
|  SIESTA  |     ✅     |      ✅      |      ✅      |       ✅        |       ✅        |
| Gaussian |     ✅     |      —      |      ✅      |       ✅        |       ✅        |
|   VASP   |     ✅     |      ✅      |      —      |       —        |       —        |
|  PYATB   |     ✅     |      ✅      |      —      |       —        |       —        |

---

## Parser Details

Here is a brief overview of the files each parser typically reads to extract data.

### ABACUS
The ABACUS parser (`--mode abacus`) reads several files from the output directory to construct the system.
- **Structure:** `STRU`
- **Hamiltonian/Overlap:** `data-HR-sparse_SPIN0.csr` (real-space Hamiltonian) and `data-SR-sparse_SPIN0.csr` (overlap matrix).
- **Eigenvalues:** `BANDS_1.dat`

### VASP
The VASP parser (`--mode vasp`) is primarily used for structures and band structure data.
- **Structure:** `OUTCAR` or `POSCAR`
- **Eigenvalues & k-points:** `EIGENVAL` and `KPOINTS`

### SIESTA
The SIESTA parser (`--mode siesta`) extracts the real-space Hamiltonian and other matrices.
- **Structure, Hamiltonian, Overlap:** Reads the `*.TSHS` (TranSIESTA Hamiltonian/Overlap) file. The file extension is specified via the `--prefix` argument (e.g., `--prefix Au_cell` for `Au_cell.TSHS`).
- **Density Matrix:** `*.DM`
- **Eigenvalues:** `*.bands`

### Gaussian
The Gaussian parser (`--mode gaussian`) can parse data from log files.
- **Structure, Hamiltonian, Overlap, Density Matrix:** Reads from Gaussian log files (`.log`) or formatted checkpoint files (`.fchk`).

### RESCU
The RESCU parser (`--mode rescu`) reads data from `.mat` files.
- **Structure, Hamiltonian, Overlap:** Extracts data from the main output file, typically named something like `al_lcao_scf.mat`.

### PYATB
The PYATB parser (`--mode pyatb`) is used for parsing data from the Python-based tight-binding (`pyatb`) tool.
- **Structure & Eigenvalues:** Reads data from `pyatb`'s standard output formats.
