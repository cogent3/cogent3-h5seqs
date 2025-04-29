[![CI](https://github.com/cogent3/cogent3-h5seqs/actions/workflows/ci.yml/badge.svg)](https://github.com/cogent3/cogent3-h5seqs/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/cogent3/cogent3-h5seqs/badge.svg?branch=develop)](https://coveralls.io/github/cogent3/cogent3-h5seqs?branch=develop)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# cogent3-h5seqs: a HDF5 storage driver for cogent3 sequence collections

Install

```
pip install https://github.com/GavinHuttley/cogent3-h5seqs.git
```

Usage
```python
coll = cogent3.load_unaligned_seqs(some_path,
                                   moltype="dna",
                                   new_type=True,
                                   storage_backend="h5seqs_unaligned")
```
For alignments, use `"h5seqs_aligned"`.
