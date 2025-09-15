[![CI](https://github.com/cogent3/cogent3-h5seqs/actions/workflows/ci.yml/badge.svg)](https://github.com/cogent3/cogent3-h5seqs/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/cogent3/cogent3-h5seqs/badge.svg?branch=develop)](https://coveralls.io/github/cogent3/cogent3-h5seqs?branch=develop)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# cogent3-h5seqs: a HDF5 storage driver for cogent3 sequence collections

`cogent3-h5seqs` is a sequence storage plug-in for [cogent3](https://cogent3.org). It uses HDF5 as the storage format for biological sequences, supporting both unaligned sequence collections and alignments. Storage can be in memory (the default) or on disk and sequences are compressed using the lzf compression engine.

The advantage of HDF5 is that once primary sequence formats have been converted from text into numpy arrays, loading and manipulating sequence data is fast and very memory efficient.

Sequences are stored under the hexdigest of their `xxhash.hash64()`. This means duplicated sequences are stored only once and we also store the mapping of sequence names to the hexdigest.

## Installation

```
pip install cogent3-h5seqs
```

## Usage

### Three types of sequence storage

#### Unaligned sequences

For sequences that may not be the same length, select `c3h5u`, or `h5seqs_unaligned`.

#### Aligned sequences, full storage

For sequences that must be the same length, select `c3h5a`, or `h5seqs_aligned`. This is a dense storage format whete every sequence is stored separately.

#### Aligned sequences, sparse storage

For sequences that must be the same length, select `c3h5s`, or `h5seqs_sparse`. This is uses a sparse matrix for storage reducing memory and storage requirements. Faster to create and write than the dense variant.

### Making `cogent3-h5seqs` the default storage

Using `cogent3.set_storage_defaults()`, you can set `cogent3-h5seqs` as the default storage. This means whenever a sequence collection is loaded from disk or created in memory, it will use the storage within this package.

The following statement makes `cogent3-h5seqs` the default for both unaligned and aligned sequence collections.

```python
import cogent3

cogent3.set_storage_defaults(unaligned_seqs="c3h5u",
                             aligned_seqs="c3h5a")
```

You can undo this setting by

```python
cogent3.set_storage_defaults(reset=True)
```

Equivalently, you could define 

### Using `cogent3-h5seqs` as storage per object

You don't have to specify the storage as the default for all instances, but can do it on a per object basis.

```python
coll = cogent3.load_unaligned_seqs(some_path,
                                   moltype="dna",
                                   storage_backend="h5seqs_unaligned")
```

or, for alignments.

```python
aln = cogent3.load_aligned_seqs(some_path,
                                   moltype="dna",
                                   storage_backend="c3h5s")
```

The same values can also be provided to the `make_unaligned_seqs()`, `make_aligned_seqs()` functions in `cogent3`.

> **Note**
> You can turn off compression with `compression=False`. This can speed up operations.

### Saving storage to disk

`cogent3-h5seqs` supports writing to disk, and employs the filename suffix `.c3h5u` for unaligned sequences and `.c3h5a` for aligned sequences. This will work whether your current object is using `cogent3-h5seqs` for storage or not. For example

```python
import cogent3

sample_aln = cogent3.get_dataset("brca1")  # using the cogent3 builtin storage
outpath = "~/Desktop/alignment_output.c3h5s"
sample_aln.write(outpath)  # writes out as cogent3-h5seqs HDF5 storage
```

For a sequence collection, do the following.

```python
sample_coll = cogent3.get_dataset("brca1").degap()
# Note the different suffix
outpath = "~/Desktop/alignment_output.c3h5u"
sample_coll.write(outpath)  # writes out as cogent3-h5seqs HDF5 storage
```
### Loading storage from disk

`cogent3` correctly directs to `cogent3-h5seqs` for loading based on the filename suffix.

```python
inpath = "~/Desktop/alignment_output.c3h5u"
sample_coll = cogent3.load_unaligned_seqs(inpath, moltype="dna")
```

> **Note**
> You cannot write an alignment instance to an unaligned storage type or vice versa. Nor can you read into the different types.
