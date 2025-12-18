# Tutorial: Plotting a Band Structure

This tutorial will guide you through the process of parsing eigenvalue data from a DFT calculation and using `dftio` to plot the electronic band structure.

## 1. Prerequisites

Before you begin, make sure you have:

- Installed `dftio` with its dependencies.
- A completed DFT calculation that includes band structure data. For this tutorial, we will use the VASP example included in the `dftio` repository at `example/vasp/GaAs/`.

## 2. Parse the Eigenvalue Data

First, we need to parse the eigenvalue data from the VASP output files. We will use the `dftio parse` command, specifying the `vasp` mode and the `--eigenvalue` flag.

```bash
dftio parse \
  --mode vasp \
  --root example/vasp/GaAs \
  --outroot ./parsed_bands
```

This command will create a directory named `parsed_bands` containing a file `0.dat`, which holds the parsed eigenvalues and k-points.

## 3. Plot the Band Structure

Now that we have the parsed data, we can use the `dftio band` command to generate a plot.

```bash
dftio band -r ./parsed_bands
```

This command will:

1.  Load the data from the `./parsed_bands` directory.
2.  Generate a plot of the band structure.
3.  Save the plot as a file named `band.png` in the current directory.

You can customize the plot using various options. For example, to plot only a specific range of bands, you can use the `--min` and `--max` flags:

```bash
# Plot bands from index 10 to 20
dftio band -r ./parsed_bands --min 10 --max 20
```

For a full list of options, run `dftio band --help`.

## Conclusion

This tutorial demonstrated how to use `dftio`'s command-line tools to easily parse and visualize electronic band structures from DFT calculations. This two-step process (parse and plot) provides a quick and efficient way to analyze your results without writing any code.
