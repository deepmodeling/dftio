# dftio
dftio is to assist machine learning communities in transcribing and manipulating DFT output into a format that is easy to read or used by machine learning models. 

dftio uses multiprocessing to paralleling the processing, and provide a standard dataset class that reads the processed dataset directly.

## Installation
The user can install dftio once located in the root directory, and run:
```bash
pip install .
```
The dependent packages will be installed accordingly.
However, the user can always manage the dependency themselves, here are the packages that dftio requires:

## Supports

Current:

| Package  | Structure | Eigenvalues | Hamiltonian | Density matrix | Overlap matrix |
|  :----:  |  :----:   |   :----:    |    :----:   |     :----:     |     :----:     |
| ABACUS   | √         | √           | √           | √              | √              |
| RESCU    | √         |             | √           |                | √              |
| SIESTA   | √         |             | √           | √              | √              |
| Gaussian | √         |             | √           | √              | √              |

Ongoing:

- Charge density
- Atomic Orbitals
- Wave Function
- Wave Function Coefficients


## How to use
To parse the DFT output files into readable data format, user can follows:

```bash
dftio parse [-h] [-ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-lp LOG_PATH] [-m MODE] [-r ROOT] [-p PREFIX] [-o OUTROOT] [-f FORMAT] [-ham] [-ovp] [-dm] [-eig]

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

## Call for Contributors
dftio is an open-source tool that calls for enthusiastic developers to contribute their talent. One can contribute through raising function requirement issues, or contact the current developer directly.

### Current Contributors (in alphabetical order)
Qiangqiang Gu, Jijie Zou, Mingkang Liu, Zixi Gan, Zhanghao Zhouyin
