# Installation

There are several ways to install `dftio`. The recommended method is to use the provided install script, which handles all dependencies automatically.

## Using the Install Script (Recommended)

The easiest way to install `dftio` is by running the `install.sh` script in the root of the repository.

```bash
# For a standard CPU-only installation
./install.sh

# If you have a CUDA-compatible GPU (e.g., CUDA 12.1)
./install.sh cu121
```

This script ensures that all dependencies, including specific versions of PyTorch and `torch-scatter`, are installed correctly.

## Manual Installation with UV

If you prefer to manage the installation yourself, you can use `uv`.

1.  **Install uv:**
    If you don't have `uv`, install it via pip:
    ```bash
    pip install uv
    ```

2.  **Sync Dependencies:**
    Use `uv sync` to install the required packages from `pyproject.toml`.

    ```bash
    # For a CPU-only installation
    uv sync --group dev

    # For a GPU installation (e.g., CUDA 12.1), specify the PyTorch find-links URL
    uv sync --group dev --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
    ```
    Including the `--group dev` flag will also install the packages required for testing and building documentation.

## Using pip

You can install `dftio` directly with pip:

```bash
# Basic install (all core features except grid integration / LDOS)
pip install dftio

# Install with grid integration support for LDOS calculations
pip install "dftio[scatter]" -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# Install with all optional dependencies (scatter + dev tools)
pip install "dftio[full]" -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# For GPU users, replace 'cpu' with your CUDA version (e.g., cu121, cu124)
pip install "dftio[scatter]" -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Or install from requirements files
pip install -r requirements.txt          # core only
pip install -r requirements-full.txt     # core + scatter
pip install -r requirements-dev.txt      # core + dev tools
```

> **Note:** The `scatter` extra installs `torch-scatter`, which is only needed for LDOS (Local Density of States) calculations via `dftio.calc.ldos`. All other dftio functionality works without it.