# cogent3-h5seqs: a storage driver for cogent3 sequence collections

Install

```
pip install "cogent3 @ git+https://github.com/cogent3/cogent3.git@develop"
```

Usage
```python
coll = cogent3.load_unaligned_seqs(some_path,
                                   moltype="dna",
                                   new_type=True,
                                   storage_backend="h5seqs_unaligned")
```
For alignments, use `"h5seqs_aligned"`.
