# Welcome to dftio

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://deepmodeling.github.io/dftio/)
[![Tests](https://github.com/deepmodeling/dftio/workflows/Tests/badge.svg)](https://github.com/deepmodeling/dftio/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/downloads/)

`dftio` is a Python library designed to assist the machine learning community by transcribing and manipulating output from Density Functional Theory (DFT) calculations into formats that are easy to read and use with machine learning models.

It leverages multiprocessing to parallelize data processing and provides a standardized dataset class for direct use.

**Key Features:**

- **Broad Compatibility:** Parses output from a wide range of DFT software, including ABACUS, VASP, SIESTA, Gaussian, and more.
- **Efficient I/O:** Uses multiprocessing to accelerate parsing of large datasets.
- **Standardized Data:** Converts varied DFT outputs into consistent data structures.
- **Flexible Output:** Saves processed data in multiple formats like `.dat`, `ase`, or `lmdb`.
- **Command-Line Interface:** Provides a powerful CLI for easy automation and scripting.

This documentation will guide you through installing `dftio`, using its features, and contributing to its development.