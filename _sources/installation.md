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

## Using pip (from PyPI)

*Coming soon. Once `dftio` is published to the Python Package Index (PyPI), you will be able to install it directly with `pip`.*

```bash
# This will be enabled in a future release
# pip install dftio
```