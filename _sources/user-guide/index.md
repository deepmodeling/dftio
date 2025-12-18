# User Guide

Welcome to the dftio user guide! This section provides comprehensive documentation on how to use dftio for parsing and processing DFT outputs.

## Contents

```{tableofcontents}
```

## Overview

dftio is designed to make it easy to:

1. **Parse DFT outputs** from various software packages
2. **Extract physical quantities** (structures, eigenvalues, Hamiltonians, etc.)
3. **Convert to ML-friendly formats** (DAT, ASE, LMDB)
4. **Process data in parallel** for large datasets

## Workflow

A typical dftio workflow involves:

1. Organize your DFT output files in a directory structure
2. Run the `dftio parse` command with appropriate options
3. Use the generated dataset in your machine learning pipeline

## Getting Help

- For command-line help: `dftio --help` or `dftio parse --help`
- For API documentation: See the [API Reference](../api/index.md)
- For development: See the [Developer Guide](../developer-guide.md)
