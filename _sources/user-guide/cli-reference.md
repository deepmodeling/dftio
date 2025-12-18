# Command-Line Interface (CLI)

`dftio` provides a powerful command-line interface for parsing and managing DFT data. The main command is `dftio`, which has several subcommands like `parse` and `band`.

## Main Command

You can get a full list of commands and top-level options by running `dftio --help`.

```text
usage: dftio [-h] [-v] {parse,band} ...

dftio is to assist machine learning communities to transcript DFT output into a format that is easy to read or used by machine learning models.

options:
  -h, --help        show this help message and exit
  -v, --version     show the dftio's version number and exit

Valid subcommands:
  {parse,band}
    parse           parse dataset from DFT output
    band            plot band for eigenvalues data
```

---

## Parser Command: `dftio parse`

This is the primary command for reading DFT output and converting it into a structured format.

```text
usage: dftio parse [-h] [-ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-lp LOG_PATH] [-m MODE] [-n NUM_WORKERS] [-r ROOT] [-p PREFIX] [-o OUTROOT] [-f FORMAT] [-ham] [-ovp] [-dm] [-eig] [-min BAND_INDEX_MIN]

options:
  -h, --help            show this help message and exit
  -ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not specified, the logs will only be output to console (default: None)
  -m MODE, --mode MODE  The name of the DFT software, currently support abacus/rescu/siesta/gaussian/pyatb (default: abacus)
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        The number of workers used to parse the dataset. (For n>1, we use the multiprocessing to accelerate io.) (default: 1)
  -r ROOT, --root ROOT  The root directory of the DFT files. (default: ./)
  -p PREFIX, --prefix PREFIX
                        The prefix of the DFT files under root. (default: frame)
  -o OUTROOT, --outroot OUTROOT
                        The output root directory. (default: ./)
  -f FORMAT, --format FORMAT
                        The output file format, should be dat, ase or lmdb. (default: dat)
  -ham, --hamiltonian   Whether to parse the Hamiltonian matrix. (default: False)
  -ovp, --overlap       Whether to parse the Overlap matrix (default: False)
  -dm, --density_matrix
                        Whether to parse the Density matrix (default: False)
  -eig, --eigenvalue    Whether to parse the kpoints and eigenvalues (default: False)
  -min BAND_INDEX_MIN, --band_index_min BAND_INDEX_MIN
                        The initial band index for eigenvalues to save.(0-band_index_min) bands will be ignored! (default: 0)
```

---

## Band Plotting Command: `dftio band`

This command loads parsed eigenvalue data and generates a band structure plot.

```text
usage: dftio band [-h] [-ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-lp LOG_PATH] [-r ROOT] [-f FORMAT] [-min BAND_INDEX_MIN] [-max BAND_INDEX_MAX]

options:
  -h, --help            show this help message and exit
  -ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not specified, the logs will only be output to console (default: None)
  -r ROOT, --root ROOT  The root directory of eigenvalues data. (default: ./)
  -f FORMAT, --format FORMAT
                        load file format, should be dat or ase. default is None, which means auto detect. (default: None)
  -min BAND_INDEX_MIN, --band_index_min BAND_INDEX_MIN
                        The minimum band index to plot. (default: 0)
  -max BAND_INDEX_MAX, --band_index_max BAND_INDEX_MAX
                        The maximum band index to plot. (default: None)
```