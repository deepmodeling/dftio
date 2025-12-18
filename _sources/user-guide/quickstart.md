# Quickstart: Parsing a VASP Calculation

This guide will walk you through a basic workflow: parsing eigenvalues from a VASP calculation output directory and saving them into a structured format.

## 1. Locate the Example Data

`dftio` comes with example files for different DFT codes. For this guide, we will use the Gallium Arsenide (GaAs) example calculated with VASP, which is located in the `example/vasp/GaAs/` directory.

This directory contains standard VASP output files:
- `OUTCAR`
- `POSCAR`
- `KPOINTS`
- `EIGENVAL`

## 2. Run the Parser

We will use the `dftio parse` command to extract the electronic eigenvalues from these files. The command specifies the DFT code (`vasp`), the location of the data, and what we want to extract (`--eigenvalue`).

Open your terminal in the root of the `dftio` repository and run the following command:

```bash
dftio parse \
  --mode vasp \
  --root example/vasp/GaAs \
  --prefix "" \
  --outroot ./parsed_data \
  --eigenvalue
```

Let's break down the command:
- `dftio parse`: The command to run the parser.
- `--mode vasp`: Specifies that we are parsing VASP output.
- `--root example/vasp/GaAs`: The directory containing the VASP files.
- `--prefix ""`: Since VASP output files don't share a common prefix (like `frame_001`, `frame_002`), we set this to an empty string. `dftio` will look for files like `OUTCAR` and `EIGENVAL` directly inside the `--root` directory.
- `--outroot ./parsed_data`: The directory where the output will be saved.
- `--eigenvalue`: A flag telling `dftio` to parse the eigenvalues and k-points.

## 3. Inspect the Output

After the command finishes, a new directory named `parsed_data` will be created. Inside, you will find a file named `0.dat`. This file contains the parsed data in a PyTorch-readable format.

The `0.dat` file holds a Python dictionary containing:
- `kpoints`: The coordinates of the k-points.
- `eigs`: The eigenvalues for each band at each k-point.
- `weights`: The weight of each k-point.

This file can now be easily loaded and used for further analysis or as input for a machine learning model. For example, to plot the band structure, you could use the `dftio band` command.

```bash
dftio band -r ./parsed_data
```
This command will generate a plot of the band structure from the parsed data.