# Parsing DFT Outputs

`dftio` provides a unified command-line interface for parsing the output of various Density Functional Theory (DFT) software packages. This guide provides detailed instructions and examples for each supported program.

## General Usage

The primary command for all parsing tasks is `dftio parse`. The two most important arguments are:

- `--mode`: Specifies the DFT code you are parsing (e.g., `abacus`, `vasp`).
- `--root`: The path to the directory containing the DFT calculation output files.

You can also specify what data to extract using flags like `--hamiltonian`, `--overlap`, `--eigenvalue`, and `--structure`.

## ABACUS

To parse an ABACUS calculation, you need to point `dftio` to the output directory that contains the `running_scf.log` file and the sparse matrix files.

```bash
dftio parse \
  --mode abacus \
  --root /path/to/abacus/output \
  --hamiltonian \
  --overlap \
  -o /path/to/save
```

This command will extract the Hamiltonian and overlap matrices and save them to the specified output directory.

## VASP

For VASP, `dftio` needs access to files like `OUTCAR`, `POSCAR`, `EIGENVAL`, and `KPOINTS`.

```bash
dftio parse \
  --mode vasp \
  --root /path/to/vasp/calculation \
  --eigenvalue \
  -o ./parsed_vasp_data
```

This will parse the eigenvalues and k-points, which can then be used to plot a band structure.

## SIESTA

The SIESTA parser looks for files such as `.TSHS` (for the Hamiltonian and overlap) and `.bands` (for the band structure).

```bash
dftio parse \
  --mode siesta \
  --root /path/to/siesta/output \
  --hamiltonian \
  --overlap \
  --band \
  -o ./siesta_parsed
```

## Gaussian

Gaussian output is typically contained in a single `.log` file. `dftio` can parse multiple directories containing these log files.

```bash
dftio parse \
  --mode gaussian \
  --root /path/to/gaussian/calculations \
  --nproc 4 \
  -o ./gaussian_parsed
```
In this example, `dftio` will search for `.log` files in the subdirectories of `/path/to/gaussian/calculations` and process them in parallel using 4 processes.
