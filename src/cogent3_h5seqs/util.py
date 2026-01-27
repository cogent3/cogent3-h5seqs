"""Shared utilities for cogent3-h5seqs."""

import contextlib
import pathlib
import typing
import uuid

import h5py
import numpy
import numpy.typing as npt
import xxhash
from cogent3.core import alphabet as c3_alphabet

if typing.TYPE_CHECKING:  # pragma: no cover
    from cogent3.core.alignment import Alignment, SequenceCollection

# following imports are for re-exporting, used by other modules
from cogent3.core.seq_storage import (  # noqa: F401
    compose_gapped_seq,
    decompose_gapped_seq,
)
from cogent3.core.seqview import (  # noqa: F401
    AlignedDataView,
    SeqDataView,
)
from cogent3.core.slice_record import SliceRecord  # noqa: F401

UNALIGNED_SUFFIX = "c3h5u"
ALIGNED_SUFFIX = "c3h5a"
SPARSE_SUFFIX = "c3h5s"
DEFAULT_COMPRESSION = "lzf"

SeqCollTypes = typing.Union["SequenceCollection", "Alignment"]
StrORBytesORArray = str | bytes | numpy.ndarray
StrOrBytes = str | bytes
NumpyIntArrayType = npt.NDArray[numpy.integer]
SeqIntArrayType = npt.NDArray[numpy.unsignedinteger]
PySeqStrType = typing.Sequence[str]

# for storing large dicts in HDF5
# for the annotation offset
offset_dtype = numpy.dtype(
    [("seqid", h5py.special_dtype(vlen=bytes)), ("value", numpy.int64)]
)
# for the seqname to seq hash as hex
name2hash2index_dtype = numpy.dtype(
    [
        ("seqid", h5py.special_dtype(vlen=bytes)),
        ("seqhash", h5py.special_dtype(vlen=bytes)),
        ("index", numpy.int32),
    ]
)

# HDF5 file modes
# x and w- mean create file, fail if exists
# r+ means read/write, file must exist
# w creates file, truncate if exists
# a means append, create if not exists
_writeable_modes = {"r+", "w", "w-", "x", "a"}


def array_hash64(data: SeqIntArrayType) -> str:
    """returns 64-bit hash of numpy array.

    Notes
    -----
    This function does not introduce randomisation and so
    is reproducible between processes.
    """
    return xxhash.xxh64(data.tobytes()).hexdigest()


def open_h5_file(
    path: str | pathlib.Path | None = None,
    mode: str = "r",
    in_memory: bool = False,
) -> h5py.File:
    if not isinstance(path, (str, pathlib.Path, type(None))):
        msg = f"Expected path to be str, Path or None, got {type(path).__name__!r}"
        raise TypeError(msg)

    in_memory = in_memory or "memory" in str(path)
    mode = "w-" if in_memory else mode
    # because h5py automatically uses an in-memory file
    # with the provided name if it already exists, we make a random name
    path = uuid.uuid4().hex if in_memory or not path else path
    mode = "w-" if mode == "w" else mode
    h5_kwargs = (
        {
            "driver": "core",
            "backing_store": False,
        }
        if in_memory
        else {}
    )
    try:
        h5_file: h5py.File = h5py.File(path, mode=mode, **h5_kwargs)
    except OSError as err:
        msg = f"Error opening HDF5 file {path}: {err}"
        raise OSError(msg) from err
    return h5_file


def _assign_attr_if_missing(h5file: h5py.File, attr: str, value: typing.Any) -> bool:
    if attr not in h5file.attrs:
        h5file.attrs[attr] = value
    return h5file.attrs[attr] == value


def _assign_alphabet_if_missing(
    h5file: h5py.File, attr: str, value: typing.Any
) -> bool:
    if attr not in h5file.attrs:
        h5file.attrs.create(attr, value, dtype=f"S{len(value)}")
    return h5file.attrs[attr].tolist() == value


def _valid_h5seqs(h5file: h5py.File, main_seq_grp: str) -> bool:
    # essential attributes, groups
    return all(
        [
            "alphabet" in h5file.attrs,
            "moltype" in h5file.attrs,
            "gap_char" in h5file.attrs,
            "missing_char" in h5file.attrs,
            main_seq_grp in h5file,
            "name_to_hash" in h5file,
        ]
    )


def _set_group(
    h5file: h5py.File,
    group_name: str,
    value: npt.NDArray,
    compression: str | None = None,
    chunk: bool | None = True,
) -> None:
    if group_name in h5file:
        del h5file[group_name]

    h5file.create_dataset(
        name=group_name,
        data=value,
        chunks=chunk or None,
        compression=compression,
        shuffle=True,
    )


def _set_offset(
    h5file: h5py.File, offset: dict[str, int] | None, compression: str | None = None
) -> None:
    # set the offset as a special group
    if not offset or h5file.mode not in _writeable_modes:
        return

    # only create an offset if there's something to store
    data = numpy.array(
        [(k.encode("utf8"), v) for k, v in offset.items() if v], dtype=offset_dtype
    )
    _set_group(h5file, "offset", data, compression=compression, chunk=False)


def _set_reversed_seqs(
    h5file: h5py.File,
    reverse_seqs: typing.Iterable[str] | None,
    compression: str | None = None,
) -> None:
    # set the reverse seqs as a special group
    if not reverse_seqs or h5file.mode not in _writeable_modes:
        return

    data = numpy.array([s.encode("utf8") for s in reverse_seqs], dtype="S")
    _set_group(h5file, "reversed_seqs", data, compression=compression, chunk=False)


def _set_name_to_hash_to_index(
    h5file: h5py.File,
    name_to_hash: dict[str, tuple[str, int]] | None,
    compression: str | None = None,
) -> None:
    # set the name to hash and hash to index mappings as a special group
    if not name_to_hash or h5file.mode not in _writeable_modes:
        return

    if "name_to_hash" in h5file:
        del h5file["name_to_hash"]

    # only create a name to hash mapping if there's something to store
    data = numpy.array(
        [
            (k.encode("utf8"), h.encode("utf8"), idx)
            for k, (h, idx) in name_to_hash.items()
            if h
        ],
        dtype=name2hash2index_dtype,
    )
    _set_group(h5file, "name_to_hash", data, compression=compression, chunk=False)


def _get_name_to_hash(h5file: h5py.File) -> npt.NDArray | None:
    return (
        None
        if "name_to_hash" not in h5file
        else typing.cast("numpy.ndarray", h5file["name_to_hash"])[:]
    )


def _get_name2hash_hash2idx(h5file: h5py.File) -> tuple[dict[str, str], dict[str, int]]:
    n2h, h2i = {}, {}
    n2h2i = _get_name_to_hash(h5file)
    if n2h2i is not None:
        for n, h, i in n2h2i:
            k = h.decode("utf8")
            n2h[n.decode("utf8")] = k
            if i >= 0:
                # exclude refseq which get's assigned a negative index
                h2i[k] = i

    return n2h, h2i


def _best_uint_dtype(index: int) -> numpy.dtype:
    """
    Choose the smallest unsigned integer dtype for values in `arr`.
    """
    for dt in (numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64):
        if index <= numpy.iinfo(dt).max:
            return numpy.dtype(dt)

    msg = "Value too large for uint64."
    raise ValueError(msg)


def duplicate_h5_file(
    *,
    h5file: h5py.File,
    path: str | pathlib.Path,
    in_memory: bool,
    compression: str | None = None,
    exclude_groups: set[str] | None = None,
) -> h5py.File:
    exclude: set[str] = exclude_groups or set()
    result = open_h5_file(path=path, mode="w", in_memory=in_memory)
    for name in h5file:
        if name in exclude:
            continue
        data = h5file[name]
        if isinstance(data, h5py.Group):
            h5file.copy(name, result, name=name)
        else:
            # have to do this explicitly, or we get a segfault
            result.create_dataset(
                name=name,
                data=typing.cast("numpy.ndarray", data)[:],
                dtype=data.dtype,
                compression=compression,
                shuffle=True,
            )

    for attr in h5file.attrs:
        result.attrs[attr] = h5file.attrs[attr]
    return result


def _restore_alphabet(
    *,
    chars: bytes | str,
    moltype: str,
    gap: StrOrBytes | None,
    missing: StrOrBytes | None,
) -> c3_alphabet.CharAlphabet:
    if isinstance(chars, bytes):
        with contextlib.suppress(UnicodeDecodeError):
            chars = chars.decode("utf8")  # type: ignore
    return c3_alphabet.make_alphabet(
        chars=chars,
        gap=gap,
        missing=missing,
        moltype=moltype,
    )


def _data_from_file(h5file: h5py.File, grp: str) -> dict[str, npt.NDArray]:
    data = {}
    for dataset in h5file[grp]:
        data[dataset] = h5file[grp][dataset][:]
    del h5file[grp]
    return data
