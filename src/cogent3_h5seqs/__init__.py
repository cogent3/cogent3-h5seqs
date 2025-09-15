import collections
import contextlib
import functools
import pathlib
import pickle
import typing
import uuid

import h5py
import numba
import numpy
import numpy.typing as npt
import typing_extensions
import xxhash
from cogent3.core import alignment as c3_alignment
from cogent3.core import alphabet as c3_alphabet
from cogent3.core import moltype as c3_moltype
from cogent3.core import sequence as c3_sequence
from cogent3.format.sequence import SequenceWriterBase
from cogent3.parse.sequence import SequenceParserBase
from h5py._hl.dataset import Dataset

__version__ = "0.7.0"

if typing.TYPE_CHECKING:  # pragma: no cover
    from cogent3.core.alignment import Alignment, SequenceCollection


UNALIGNED_SUFFIX = "c3h5u"
ALIGNED_SUFFIX = "c3h5a"
SPARSE_SUFFIX = "c3h5s"
DEFAULT_COMPRESSION = "lzf"

SeqCollTypes = typing.Union["SequenceCollection", "Alignment"]
StrORBytesORArray = str | bytes | numpy.ndarray
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
    gap: c3_alphabet.StrOrBytes | None,
    missing: c3_alphabet.StrOrBytes | None,
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


class UnalignedSeqsData(c3_alignment.SeqsDataABC):
    _ungapped_grp: str = "ungapped"
    _suffix: str = UNALIGNED_SUFFIX

    def __init__(
        self,
        *,
        data: h5py.File,
        alphabet: c3_alphabet.CharAlphabet,
        offset: dict[str, int] | None = None,
        check: bool = False,
        reversed_seqs: set[str] | frozenset[str] | None = None,
        compression: bool = True,
    ) -> None:
        self._compress = DEFAULT_COMPRESSION if compression else None
        self._alphabet: c3_alphabet.CharAlphabet = alphabet
        self._file: h5py.File = data
        self._primary_grp: str = self._ungapped_grp

        reversed_seqs = reversed_seqs or frozenset()
        _set_reversed_seqs(self._file, reversed_seqs, compression=self._compress)
        offset = offset or {}
        _set_offset(self._file, offset=offset, compression=self._compress)
        self._attr_set: bool = False
        self._name_to_hash: dict[str, str] = {}
        self._hash_to_index: dict[str, int] = {}
        if check:
            self._check_file(self._file)

    def __getstate__(self) -> dict[str, typing.Any]:
        if self._file.mode != "r":
            msg = (
                f"Cannot pickle {self.__class__.__name__!r} unless file is "
                f"opened in read-only mode (got mode={self._file.mode!r})"
            )
            raise pickle.PicklingError(msg)

        path = pathlib.Path(self._file.filename)
        return {"path": path, "alphabet": self.alphabet}

    def __setstate__(self, state: dict[str, typing.Any]) -> None:
        """Restore from pickle."""
        h5file = h5py.File(pathlib.Path(state["path"]), mode="r")
        data_kw = (
            "data" if "unaligned" in self.__class__.__name__.lower() else "gapped_seqs"
        )
        kwargs = {data_kw: h5file, "alphabet": state["alphabet"]}
        obj = self.__class__(**kwargs)
        self.__dict__.update(obj.__dict__)
        # we have to avoid garbage colection closing the h5file once this scope
        # cleaned up, so we close it outselves and open it again directly assigning
        # only to self
        self.close()
        self._file = h5py.File(pathlib.Path(state["path"]), mode="r")

    def __repr__(self) -> str:
        self._populate_attrs()
        name = self.__class__.__name__
        path = pathlib.Path(self._file.filename)
        attr_vals = [f"'{path.name}'"]
        attr_vals.extend(
            f"{attr}={self._file.attrs[attr]!r}"
            for attr in self._file.attrs
            if attr != "alphabet"
        )
        if self.alphabet.moltype.name == "bytes":
            attr_vals.append("alphabet=bytes")
        else:
            attr_vals.append(f"alphabet='{''.join(self.alphabet)}'")
        n2h, _ = _get_name2hash_hash2idx(self._file)
        parts = ", ".join(attr_vals)
        return f"{name}({parts}, num_seqs={len(n2h)})"

    def _invalid_seqids(self, seqids: PySeqStrType) -> set[str]:
        """returns seqids not present in self.names"""
        return set(seqids) - set(self.names)

    @classmethod
    def _check_file(cls, file: h5py.File) -> None:
        if not _valid_h5seqs(file, cls._ungapped_grp):
            msg = f"File {file} is not a valid {cls.__name__} file"
            raise ValueError(msg)

    def _populate_attrs(self) -> None:
        if self._attr_set:
            return
        data = self._file
        _assign_alphabet_if_missing(data, "alphabet", self._alphabet.as_bytes())
        _assign_attr_if_missing(data, "gap_char", self._alphabet.gap_char or "")
        _assign_attr_if_missing(data, "missing_char", self._alphabet.missing_char or "")
        _assign_attr_if_missing(
            data, "moltype", getattr(self._alphabet.moltype, "name", None)
        )
        self._attr_set = True

    def _populate_optional_grps(
        self,
        offset: dict[str, int] | None,
        reversed_seqs: frozenset[str] | None,
        name_to_hash: dict[str, str],
        hash_to_index: dict[str, int],
    ) -> None:
        # we always compress these as they tend be loaded in one go
        if offset := offset or {}:
            _set_offset(
                self._file, offset=self.offset | offset, compression=DEFAULT_COMPRESSION
            )

        reversed_seqs = reversed_seqs or frozenset()
        _set_reversed_seqs(self._file, reversed_seqs, compression=DEFAULT_COMPRESSION)
        _set_name_to_hash_to_index(
            self._file,
            {k: (h, hash_to_index.get(h, -1)) for k, h in name_to_hash.items()},
            compression=DEFAULT_COMPRESSION,
        )
        self._name_to_hash |= name_to_hash
        self._hash_to_index |= hash_to_index

    @property
    def filename_suffix(self) -> str:
        """suffix for the files"""
        return self._suffix

    @filename_suffix.setter
    def filename_suffix(self, value: str) -> None:
        """setter for the file name suffix"""
        self._suffix = value.removeprefix(".")

    def get_hash(self, seqid: str) -> str | None:
        """returns xxhash 64-bit hash for seqid"""
        if seqid not in self:
            # the contains method triggers loading of name_to_seqhash
            return None
        return self._name_to_hash.get(seqid)

    def set_attr(self, attr_name: str, attr_value: str, force: bool = False) -> None:
        """Set an attribute on the file

        Parameters
        ----------
        attr_name
            name of the attribute
        attr_value
            value to set, should be small
        force
            if True, deletes the attribute if it exists and sets it to the new value
        """
        if not self.writable:
            msg = "cannot set attributes on a read-only file"
            raise PermissionError(msg)

        if attr_name in self._file.attrs:
            if not force:
                return
            del self._file.attrs[attr_name]

        try:
            self._file.attrs[attr_name] = attr_value
        except TypeError as e:
            msg = (
                f"Cannot set attribute {attr_name!r} to {attr_value!r} with "
                f"type {type(attr_value)=}"
            )
            raise TypeError(msg) from e

    def get_attr(self, attr_name: str) -> str:
        """get attr_name from the file"""
        if attr_name not in self._file.attrs:
            msg = f"attribute {attr_name!r} not found"
            raise KeyError(msg)
        return typing.cast("str", self._file.attrs[attr_name])

    @property
    def writable(self) -> bool:
        """whether the file is writable"""
        return self._file.mode in _writeable_modes

    def __del__(self) -> None:
        if not (getattr(self, "_file", None) and self._file.id):
            return

        # we need to get the file name before closing file
        path = pathlib.Path(self._file.filename)
        self._file.close()
        if path.exists() and not path.suffix:
            # we treat these as a temporary file
            path.unlink(missing_ok=True)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if set(self.names) != set(other.names):
            return False

        # check all meta-data attrs, including
        # dynamically created by user
        attrs_self = set(self._file.attrs.keys())
        attrs_other = set(other._file.attrs.keys())
        if attrs_self != attrs_other:
            return False

        for attr_name in attrs_self:
            self_attr = self._file.attrs[attr_name]
            other_attr = other._file.attrs[attr_name]
            if self_attr != other_attr:
                return False

        # check the non-sequence groups are the same
        group_names = ("reversed_seqs", "offset")
        for field_name in group_names:
            self_field = getattr(self, field_name)
            other_field = getattr(other, field_name)
            if self_field != other_field:
                return False

        # compare individual sequences via hashes
        self_hashes = {name: self.get_hash(seqid=name) for name in self.names}
        other_hashes = {name: other.get_hash(seqid=name) for name in other.names}
        return self_hashes == other_hashes

    def __ne__(
        self,
        other: object,
    ) -> bool:
        return not (self == other)

    def __contains__(self, seqid: str) -> bool:
        """seqid in self"""
        if not self._name_to_hash:
            n2h, h2i = _get_name2hash_hash2idx(self._file)  # populates if empty
            self._name_to_hash = n2h
            self._hash_to_index = h2i
        return seqid in self._name_to_hash

    @functools.singledispatchmethod
    def __getitem__(self, index: str | int) -> c3_alignment.SeqDataView:
        msg = f"__getitem__ not implemented for {type(index)}"
        raise TypeError(msg)

    @__getitem__.register
    def _(self, index: str) -> c3_alignment.SeqDataView:
        return self.get_view(index)

    @__getitem__.register
    def _(self, index: int) -> c3_alignment.SeqDataView:
        return self[self.names[index]]

    def __len__(self) -> int:
        return len(self.names)

    @property
    def alphabet(self) -> c3_alphabet.CharAlphabet:
        return self._alphabet

    @property
    def names(self) -> tuple[str, ...]:
        n2h = _get_name_to_hash(self._file)
        return tuple(n2h["seqid"].astype(str).tolist()) if n2h is not None else ()

    @property
    def offset(self) -> dict[str, int]:
        all_offsets = dict.fromkeys(self.names, 0)
        if "offset" not in self._file:
            return all_offsets
        data = typing.cast("numpy.ndarray", self._file["offset"])[:]

        return all_offsets | {k.decode("utf8"): int(v) for k, v in data}

    @property
    def reversed_seqs(self) -> frozenset[str]:
        if "reversed_seqs" not in self._file:
            return frozenset()

        data = typing.cast("numpy.ndarray", self._file["reversed_seqs"])[:]
        return frozenset(v.decode("utf8") for v in data)

    def _make_new_h5_file(
        self,
        *,
        data: h5py.File | None,
        alphabet: c3_alphabet.CharAlphabet | None,
        offset: dict[str, int] | None,
        reversed_seqs: set[str] | frozenset[str] | None,
        exclude_groups: set[str] | None = None,
    ) -> tuple[
        h5py.File, c3_alphabet.CharAlphabet, dict[str, int] | None, frozenset[str]
    ]:
        datafile: h5py.File = (
            duplicate_h5_file(
                h5file=self._file,
                path="memory",
                in_memory=True,
                exclude_groups=exclude_groups,
            )
            if data is None
            else data
        )
        alphabet = alphabet or self.alphabet

        reversed_seqs = frozenset(reversed_seqs or self.reversed_seqs)
        if alphabet and alphabet != self.alphabet:
            datafile.attrs["alphabet"] = alphabet.as_bytes()
            datafile.attrs["moltype"] = getattr(alphabet.moltype, "name", None)

        if offset := offset or self.offset:
            _set_offset(datafile, offset=offset, compression=self._compress)
        _set_reversed_seqs(datafile, reversed_seqs, compression=self._compress)

        return datafile, alphabet, offset, reversed_seqs

    def copy(
        self,
        data: h5py.File | None = None,
        alphabet: c3_alphabet.CharAlphabet | None = None,
        offset: dict[str, int] | None = None,
        reversed_seqs: set[str] | frozenset[str] | None = None,
        exclude_groups: set[str] | None = None,
    ) -> typing_extensions.Self:
        data, alphabet, offset, reversed_seqs = self._make_new_h5_file(
            data=data,
            alphabet=alphabet,
            offset=offset,
            reversed_seqs=reversed_seqs,
            exclude_groups=exclude_groups,
        )
        return self.__class__(
            data=data,
            alphabet=alphabet,
            offset=offset,
            reversed_seqs=reversed_seqs,
            check=False,
        )

    def _compatible_alphabet(self, alphabet: c3_alphabet.CharAlphabet) -> bool:
        return (
            len(self.alphabet) == len(alphabet)
            and len(
                {
                    (a, b)
                    for a, b in zip(self.alphabet, alphabet, strict=False)
                    if a != b
                },
            )
            == 1
        )

    def to_alphabet(
        self,
        alphabet: c3_alphabet.AlphabetABC,
        check_valid: bool = True,
    ) -> "UnalignedSeqsData":
        alpha = typing.cast("c3_alphabet.CharAlphabet", alphabet)
        if self._compatible_alphabet(alpha):
            return self.copy(alphabet=alpha)

        new_data = {}
        for seqid in self.names:
            seq_data = self.get_seq_array(seqid=seqid)
            as_new_alpha = self.alphabet.convert_seq_array_to(
                seq=seq_data,
                alphabet=alpha,
                check_valid=False,
            )
            if check_valid and not alpha.is_valid(as_new_alpha):
                msg = (
                    f"Changing from old alphabet={self.alphabet} to new "
                    f"{alpha=} is not valid for this data"
                )
                raise c3_alphabet.AlphabetError(
                    msg,
                )
            new_data[seqid] = as_new_alpha

        return make_unaligned(
            "memory",
            data=new_data,
            alphabet=alphabet,
            in_memory=True,
            mode="w",
            offset=self.offset,
            reversed_seqs=self.reversed_seqs,
        )

    def _add_seq(self, seqhash: str, seqarray: SeqIntArrayType) -> None:
        self._file.create_dataset(
            name=f"{self._primary_grp}/{seqhash}",
            data=seqarray,
            compression=self._compress,
            shuffle=True,
        )

    def add_seqs(
        self,
        seqs: dict[str, StrORBytesORArray],
        force_unique_keys: bool = True,
        offset: dict[str, int] | None = None,
        reversed_seqs: frozenset[str] | None = None,
    ) -> "UnalignedSeqsData":
        """Returns self with added sequences

        Parameters
        ----------
        seqs
            sequences to add as {name: value, ...}
        force_unique_keys
            raises ValueError if any names already exist in the collection.
            If False, skips duplicate seqids.
        offset
            offsets relative to parent sequence to add as {name: int, ...}
        """
        if not self.writable:
            msg = "Cannot add sequences to a read-only file"
            raise PermissionError(msg)

        self._populate_attrs()
        n2h, _ = _get_name2hash_hash2idx(self._file)
        name_to_hash = self._name_to_hash | n2h
        overlap = name_to_hash.keys() & seqs.keys()
        if force_unique_keys and overlap:
            msg = f"{overlap} already exist in collection"
            raise ValueError(msg)

        seqhash_to_names: dict[str, list[str]] = collections.defaultdict(list)
        for seqid, seqhash in name_to_hash.items():
            seqhash_to_names[seqhash].append(seqid)

        for seqid, seq in seqs.items():
            if overlap and seqid in overlap:
                continue

            seqarray = typing.cast(
                "SeqIntArrayType", self.alphabet.to_indices(seq, validate=True)
            )
            seqhash = array_hash64(seqarray)
            name_to_hash[seqid] = seqhash
            if seqhash in seqhash_to_names:
                # same seq, different name
                continue
            seqhash_to_names[seqhash].append(seqid)
            self._add_seq(seqhash=seqhash, seqarray=seqarray)

        self._populate_optional_grps(offset, reversed_seqs, name_to_hash, {})
        return self

    def get_seq_array(
        self,
        *,
        seqid: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> SeqIntArrayType:
        """Returns the sequence as a numpy array of indices"""
        if self._invalid_seqids([seqid]):
            msg = f"Sequence {seqid!r} not found"
            raise KeyError(msg)

        start = start or 0
        stop = stop if stop is not None else self.get_seq_length(seqid=seqid)
        step = step or 1

        if start < 0 or stop < 0 or step < 1:
            msg = f"{start=}, {stop=}, {step=} not >= 1"
            raise ValueError(msg)

        out_len = (stop - start + step - 1) // step
        out = numpy.empty(out_len, dtype=self.alphabet.dtype)
        dataset_name = f"{self._ungapped_grp}/{self.get_hash(seqid=seqid)}"
        out[:] = typing.cast("numpy.ndarray", self._file[dataset_name])[start:stop:step]
        return out

    def get_seq_bytes(
        self,
        *,
        seqid: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> bytes:
        return self.get_seq_str(seqid=seqid, start=start, stop=stop, step=step).encode(
            "utf8"
        )

    def get_seq_str(
        self,
        *,
        seqid: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> str:
        return self.alphabet.from_indices(
            self.get_seq_array(seqid=seqid, start=start, stop=stop, step=step)
        )

    def get_view(self, seqid: str) -> c3_alignment.SeqDataView:
        return c3_alignment.SeqDataView(
            parent=self,
            seqid=seqid,
            parent_len=self.get_seq_length(seqid=seqid),
            alphabet=self.alphabet,
            offset=self.offset.get(seqid, 0),
        )

    def get_seq_length(self, seqid: str) -> int:
        """Returns the length of the sequence"""
        dataset_name = f"{self._ungapped_grp}/{self.get_hash(seqid=seqid)}"
        return typing.cast("numpy.ndarray", self._file[dataset_name]).shape[0]

    @classmethod
    def from_seqs(
        cls,
        *,
        data,
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> "UnalignedSeqsData":
        # make in memory
        path = kwargs.pop("storage_path", "memory")
        kwargs = {"mode": "w"} | kwargs
        return make_unaligned(
            path,
            data=data,
            alphabet=alphabet,
            **kwargs,
        )

    @classmethod
    def from_storage(
        cls,
        seqcoll: c3_alignment.SequenceCollection,
        path: str | pathlib.Path | None = None,
        **kwargs,
    ) -> "UnalignedSeqsData":
        """convert a cogent3 SeqsDataABC into UnalignedSeqsData"""
        if type(seqcoll) is not c3_alignment.SequenceCollection:
            msg = f"Expected seqcoll to be an instance of SequenceCollection, got {type(seqcoll).__name__!r}"
            raise TypeError(msg)

        in_memory = kwargs.pop("in_memory", False)
        h5file = open_h5_file(path=path, mode="w", in_memory=in_memory)
        obj = cls(
            data=h5file,
            alphabet=seqcoll.moltype.most_degen_alphabet(),
            check=False,
            **kwargs,
        )
        seqs = {s.name: numpy.array(s) for s in seqcoll.seqs}
        obj.add_seqs(
            seqs=seqs,
            offset=seqcoll.storage.offset,
            reversed_seqs=seqcoll.storage.reversed_seqs,
        )
        return obj

    @classmethod
    def from_file(
        cls, path: str | pathlib.Path, mode: str = "r", check: bool = True
    ) -> "UnalignedSeqsData":
        h5file = open_h5_file(path=path, mode=mode, in_memory=False)
        alphabet = _restore_alphabet(
            chars=h5file.attrs.get("alphabet"),
            moltype=c3_moltype.get_moltype(h5file.attrs.get("moltype")),
            missing=h5file.attrs.get("missing_char") or None,
            gap=h5file.attrs.get("gap_char") or None,
        )
        return cls(data=h5file, alphabet=alphabet, check=check)

    def _write(self, path: str | pathlib.Path, exclude_groups: set[str]) -> None:
        path = pathlib.Path(path).expanduser().absolute()
        curr_path = pathlib.Path(self._file.filename).absolute()
        if path == curr_path:
            # nothing to do
            return
        output = duplicate_h5_file(
            h5file=self._file, path=path, exclude_groups=exclude_groups, in_memory=False
        )
        output.close()

    def write(self, path: str | pathlib.Path) -> None:
        """Write the UnalignedSeqsData object to a file"""
        path = pathlib.Path(path).expanduser().absolute()
        if path.suffix != f".{self.filename_suffix}":
            msg = f"path {path} does not have the expected suffix '.{self.filename_suffix}'"
            raise ValueError(msg)
        self._write(path=path, exclude_groups=set())

    @property
    def h5file(self) -> h5py.File | None:
        """returns the HDF file"""
        return self._file

    def close(self) -> None:
        """close the HDF file"""
        if not (self._file and self._file.id):
            return

        if not self._attr_set:
            self._populate_attrs()

        self._file.close()


class AlignedSeqsData(UnalignedSeqsData, c3_alignment.AlignedSeqsDataABC):
    _gapped_grp: str = "gapped"
    _ungapped_grp: str = "ungapped"
    _gaps_grp: str = "gaps"
    _suffix: str = ALIGNED_SUFFIX

    def __init__(
        self,
        *,
        gapped_seqs: h5py.File,
        alphabet: c3_alphabet.AlphabetABC,
        offset: dict[str, int] | None = None,
        check: bool = True,
        reversed_seqs: frozenset[str] | None = None,
        compression: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            data=gapped_seqs,
            alphabet=alphabet,
            offset=offset,
            check=check,
            reversed_seqs=reversed_seqs,
            compression=compression,
        )
        self._primary_grp = self._gapped_grp

    @classmethod
    def _check_file(cls, file: h5py.File) -> None:
        if not _valid_h5seqs(file, cls._gapped_grp):
            msg = f"File {file} is not a valid {cls.__name__} file"
            raise ValueError(msg)

    @property
    def align_len(self) -> int:
        """length of the alignment"""
        if not self.names:
            return 0
        name = self.names[0]
        return typing.cast(
            "numpy.ndarray",
            self._file[f"{self._gapped_grp}/{self.get_hash(seqid=name)}"],
        ).shape[0]

    def __len__(self) -> int:
        return self.align_len

    def get_seq_length(self, seqid: str) -> int:
        """Returns the length of the sequence"""
        if self._invalid_seqids([seqid]):
            msg = f"Sequence {seqid!r} not found"
            raise KeyError(msg)

        seqhash = self.get_hash(seqid=seqid)
        if seqhash in self._file.get(self._ungapped_grp, {}):
            return typing.cast(
                "numpy.ndarray",
                self._file[f"{self._ungapped_grp}/{self.get_hash(seqid=seqid)}"],
            ).shape[0]

        seqarray = self.get_gapped_seq_array(seqid=seqid)
        nongaps = seqarray != self.alphabet.gap_index
        if self.alphabet.missing_index is not None:
            nongaps |= seqarray != self.alphabet.missing_index
        return nongaps.sum()

    @classmethod
    def from_seqs(
        cls,
        *,
        data: dict[str, StrORBytesORArray],
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> "AlignedSeqsData":
        """Construct an AlignedSeqsData object from a dict of aligned sequences

        Parameters
        ----------
        data
            dict of gapped sequences {name: seq, ...}. sequences must all be
            the same length
        alphabet
            alphabet object for the sequences
        """
        # need to support providing a path
        path = kwargs.pop("storage_path", "memory")
        kwargs = {"mode": "w"} | kwargs
        maker = _aligned_makers[cls]
        return maker(path, data=data, alphabet=alphabet, **kwargs)

    @classmethod
    def from_names_and_array(
        cls,
        *,
        names: PySeqStrType,
        data: SeqIntArrayType,
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> "AlignedSeqsData":
        if len(names) != data.shape[0] or not len(names):
            msg = "Number of names must match number of rows in data."
            raise ValueError(msg)

        data = {name: data[i] for i, name in enumerate(names)}
        path = kwargs.pop("storage_path", None)
        mode = kwargs.pop("mode", "w")
        maker = _aligned_makers[cls]
        return maker(path, data=data, alphabet=alphabet, mode=mode, **kwargs)

    @classmethod
    def from_seqs_and_gaps(
        cls,
        *,
        seqs: dict[str, StrORBytesORArray],
        gaps: dict[str, SeqIntArrayType],
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> "AlignedSeqsData":
        data = {}
        for seqid, seq in seqs.items():
            gp = gaps[seqid]
            gapped = c3_alignment.compose_gapped_seq(
                ungapped_seq=seq,
                gaps=gp,
                gap_index=alphabet.gap_index,
            )
            data[seqid] = gapped

        path = kwargs.pop("storage_path", None)
        mode = kwargs.pop("mode", "w")
        maker = _aligned_makers[cls]
        return maker(path, data=data, alphabet=alphabet, mode=mode, **kwargs)

    @classmethod
    def from_storage(
        cls,
        seqcoll: c3_alignment.Alignment,
        path: str | pathlib.Path | None = None,
        **kwargs,
    ) -> "AlignedSeqsData":
        """convert a cogent3 AlignedSeqsDataABC into AlignedSeqsData"""
        if type(seqcoll) is not c3_alignment.Alignment:
            msg = f"Expected seqcoll to be an instance of Alignment, got {type(seqcoll).__name__!r}"
            raise TypeError(msg)

        in_memory = kwargs.pop("in_memory", False)
        h5file = open_h5_file(path=path, mode="w", in_memory=in_memory)
        obj = cls(
            gapped_seqs=h5file,
            alphabet=seqcoll.moltype.most_degen_alphabet(),
            check=False,
            **kwargs,
        )
        seqs = {s.name: numpy.array(s) for s in seqcoll.seqs}
        obj.add_seqs(
            seqs=seqs,
            offset=seqcoll.storage.offset,
            reversed_seqs=seqcoll.storage.reversed_seqs,
        )
        return obj

    @classmethod
    def from_file(
        cls, path: str | pathlib.Path, mode: str = "r", check: bool = True
    ) -> "AlignedSeqsData":
        h5file = open_h5_file(path=path, mode=mode, in_memory=False)
        alphabet = _restore_alphabet(
            chars=h5file.attrs.get("alphabet"),
            gap=h5file.attrs.get("gap_char"),
            missing=h5file.attrs.get("missing_char"),
            moltype=c3_moltype.get_moltype(h5file.attrs.get("moltype")),
        )
        return cls(gapped_seqs=h5file, alphabet=alphabet, check=check)

    def _make_gaps_and_ungapped(self, seqid: str) -> None:
        seqhash = self.get_hash(seqid=seqid)
        if seqhash is None:
            msg = f"Sequence {seqid!r} not found"
            raise KeyError(msg)

        ungapped, gaps = c3_alignment.decompose_gapped_seq(
            self.get_gapped_seq_array(seqid=seqid),
            alphabet=self.alphabet,
        )
        self._file.create_dataset(
            name=f"{self._gaps_grp}/{seqhash}",
            data=gaps,
            chunks=True,
            compression=self._compress,
            shuffle=True,
        )
        self._file.create_dataset(
            name=f"{self._ungapped_grp}/{seqhash}",
            data=ungapped,
            chunks=True,
            compression=self._compress,
            shuffle=True,
        )

    def _get_gaps(self, seqid: str) -> NumpyIntArrayType:
        seqhash = self.get_hash(seqid=seqid)
        if seqhash not in self._file.get(self._gaps_grp, {}):
            self._make_gaps_and_ungapped(seqid)
        return typing.cast("numpy.ndarray", self._file[f"{self._gaps_grp}/{seqhash}"])[
            :
        ]

    def get_gaps(self, seqid: str) -> NumpyIntArrayType:
        return self._get_gaps(seqid)

    def get_seq_array(
        self,
        *,
        seqid: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> SeqIntArrayType:
        """Returns the sequence as a numpy array of indices"""
        if seqid in self and self.get_hash(seqid) not in self._file.get(
            self._gaps_grp, {}
        ):
            self._make_gaps_and_ungapped(seqid)
        return super().get_seq_array(seqid=seqid, start=start, stop=stop, step=step)

    def _get_gapped_seq_array(
        self,
        seqid: str,
        start: int,
        stop: int,
        step: int,
    ) -> SeqIntArrayType:
        seqhash = self.get_hash(seqid=seqid)
        dataset_name = f"{self._gapped_grp}/{seqhash}"
        return typing.cast("numpy.ndarray", self._file[dataset_name])[start:stop:step]

    def get_gapped_seq_array(
        self,
        seqid: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> SeqIntArrayType:
        if self._invalid_seqids([seqid]):
            msg = f"seqid not present {seqid!r}"
            raise KeyError(msg)

        start = start or 0
        stop = stop if stop is not None else self.align_len
        step = step or 1
        if start < 0 or stop < 0 or step < 1:
            msg = f"{start=}, {stop=}, {step=} not >= 1"
            raise ValueError(msg)
        return self._get_gapped_seq_array(
            seqid=seqid, start=start, stop=stop, step=step
        )

    def get_gapped_seq_str(
        self,
        seqid: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> str:
        data = self.get_gapped_seq_array(seqid=seqid, start=start, stop=stop, step=step)
        return self.alphabet.from_indices(data)

    def get_gapped_seq_bytes(
        self,
        seqid: str,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> bytes:
        return self.get_gapped_seq_str(
            seqid=seqid, start=start, stop=stop, step=step
        ).encode("utf8")

    def get_view(
        self,
        seqid: str,
        slice_record: c3_sequence.SliceRecord | None = None,
    ) -> c3_alignment.AlignedDataView:
        return c3_alignment.AlignedDataView(
            parent=self,
            seqid=seqid,
            alphabet=self.alphabet,
            slice_record=slice_record,
        )

    def get_pos_range(
        self,
        names: PySeqStrType,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> SeqIntArrayType:
        if diff := self._invalid_seqids(names):
            msg = f"these names not present {diff}"
            raise KeyError(msg)

        start = start or 0
        stop = stop or self.align_len
        step = step or 1
        if start < 0 or stop < 0 or step < 1:
            msg = f"{start=}, {stop=}, {step=} not >= 1"
            raise ValueError(msg)

        array_seqs = numpy.empty(
            (len(names), len(range(start, stop, step))), dtype=self.alphabet.dtype
        )
        for index, name in enumerate(names):
            array_seqs[index] = self.get_gapped_seq_array(
                seqid=name,
                start=start,
                stop=stop,
                step=step,
            )
        return array_seqs

    def get_positions(
        self,
        names: typing.Sequence[str],
        positions: typing.Sequence[int],
    ) -> numpy.ndarray[numpy.uint8]:
        """returns alignment positions for names

        Parameters
        ----------
        names
            series of sequence names
        positions
            indices lying within self

        Returns
        -------
            2D numpy.array, oriented by sequence

        Raises
        ------
        IndexError if a provided position is negative or
        greater then alignment length.
        """
        if not len(positions):
            msg = "must provide positions"
            raise NotImplementedError(msg)

        if diff := self._invalid_seqids(names):
            msg = f"these names not present {diff}"
            raise KeyError(msg)

        min_index, max_index = numpy.min(positions), numpy.max(positions)
        if min_index < 0 or max_index > self.align_len:
            msg = f"Out of range: {min_index=} and / or {max_index=}"
            raise IndexError(msg)

        array_seqs = numpy.empty(
            (len(names), len(positions)), dtype=self.alphabet.dtype
        )
        for index, name in enumerate(names):
            array_seqs[index] = self.get_gapped_seq_array(seqid=name)[positions]

        return array_seqs

    def get_ungapped(
        self,
        name_map: dict[str, str],
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> tuple[dict, dict]:
        if (start or 0) < 0 or (stop or 0) < 0 or (step or 1) <= 0:
            msg = f"{start=}, {stop=}, {step=} not >= 0"
            raise ValueError(msg)

        names = tuple(name_map.values())
        seq_array = self.get_pos_range(
            names=names,
            start=start,
            stop=stop,
            step=step,
        )
        # now exclude gaps and missing
        gap_index = self.alphabet.gap_index
        missing_index = self.alphabet.missing_index or -1
        seq_array, seq_lengths = remove_gaps(seq_array, gap_index, missing_index)
        seqs = {name: seq_array[i, : seq_lengths[i]] for i, name in enumerate(names)}
        offset = {n: v for n, v in self.offset.items() if n in names}
        reversed_seqs = self.reversed_seqs.intersection(name_map.keys())
        return seqs, {
            "offset": offset,
            "name_map": name_map,
            "reversed_seqs": reversed_seqs,
        }

    def add_seqs(
        self,
        seqs: dict[str, StrORBytesORArray],
        force_unique_keys: bool = True,
        offset: dict[str, int] | None = None,
        reversed_seqs: frozenset[str] | None = None,
        **kwargs,
    ) -> "AlignedSeqsData":
        """Returns same object with added sequences.

        Parameters
        ----------
        seqs
            dict of sequences to add {name: seq, ...}
        force_unique_keys
            if True, raises ValueError if any sequence names already exist in the collection
            If False, skips duplicate seqids.
        offset
            dict of offsets relative to parent for the new sequences.
        """
        lengths = {len(seq) for seq in seqs.values()}

        if len(lengths) > 1 or (self.align_len and self.align_len not in lengths):
            msg = f"not all lengths equal {lengths=}"
            raise ValueError(msg)

        super().add_seqs(
            seqs=seqs,
            force_unique_keys=force_unique_keys,
            offset=offset,
            reversed_seqs=reversed_seqs,
        )
        return self

    def copy(
        self,
        data: h5py.File | None = None,
        alphabet: c3_alphabet.CharAlphabet | None = None,
        offset: dict[str, int] | None = None,
        reversed_seqs: set[str] | frozenset[str] | None = None,
    ) -> "AlignedSeqsData":
        data, alphabet, offset, reversed_seqs = self._make_new_h5_file(
            data=data,
            alphabet=alphabet,
            offset=offset,
            reversed_seqs=reversed_seqs,
        )
        return self.__class__(
            gapped_seqs=data,
            alphabet=alphabet,
            offset=offset,
            reversed_seqs=reversed_seqs,
            check=False,
        )

    def to_alphabet(
        self,
        alphabet: c3_alphabet.CharAlphabet,
        check_valid: bool = True,
    ) -> "AlignedSeqsData":
        """Returns a new AlignedSeqsData object with the same underlying data
        with a new alphabet."""
        if self._compatible_alphabet(alphabet):
            return self.copy(alphabet=alphabet)

        gapped = {}
        for name in self.names:
            seq_data = self.get_gapped_seq_array(seqid=name)
            as_new_alpha = self.alphabet.convert_seq_array_to(
                seq=seq_data,
                alphabet=alphabet,
                check_valid=False,
            )
            if check_valid and not alphabet.is_valid(as_new_alpha):
                msg = (
                    f"Changing from old alphabet={self.alphabet} to new "
                    f"{alphabet=} is not valid for this data"
                )
                raise c3_alphabet.AlphabetError(msg)

            gapped[name] = as_new_alpha

        return self.from_seqs(
            data=gapped,
            alphabet=alphabet,
            offset=self.offset,
            reversed_seqs=self.reversed_seqs,
            check=False,
        )

    def write(self, path: str | pathlib.Path) -> None:
        """Write the AlignedSeqsData object to a file"""
        path = pathlib.Path(path)
        if path.suffix != f".{self.filename_suffix}":
            msg = f"path {path} does not have the expected suffix '.{self.filename_suffix}'"
            raise ValueError(msg)
        self._write(path=path, exclude_groups={self._ungapped_grp, self._gaps_grp})

    def variable_positions(
        self,
        names: typing.Sequence[str],
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> numpy.ndarray:
        """returns absolute indices of positions that have more than one state

        Parameters
        ----------
        names
            selected seqids
        start
            absolute start
        stop
            absolute stop
        step
            step

        Returns
        -------
        Absolute indices (as distinct from an index relative to start) of
        variable positions.
        """
        start = start or 0
        if len(names) < 2:
            return numpy.array([])

        array_seqs = self.get_pos_range(names, start=start, stop=stop, step=step)
        if array_seqs.size == 0:
            return numpy.array([])

        step = step or 1
        indices = (array_seqs != array_seqs[0]).any(axis=0)
        # because we need to return absolute indices, we add start
        # to the result
        indices = numpy.where(indices)[0]
        if step > 1:
            indices *= step
        indices += start
        return indices


def _get_indices_diffs(
    ref_seq: SeqIntArrayType, seqarray: SeqIntArrayType
) -> tuple[NumpyIntArrayType, SeqIntArrayType]:
    diff_indices = numpy.where(ref_seq != seqarray)[0]
    diffs = seqarray[diff_indices]
    return diff_indices, diffs


def _get_diff_indices_vals_for_index(
    *,
    index: int,
    all_indices: NumpyIntArrayType,
    all_values: SeqIntArrayType,
    row_ptrs: NumpyIntArrayType,
) -> tuple[NumpyIntArrayType, SeqIntArrayType]:
    start, stop = row_ptrs[index], row_ptrs[index + 1]
    return all_indices[start:stop], all_values[start:stop]


def _inflate_seq(
    *,
    index: int,
    ref_seq: SeqIntArrayType,
    all_indices: NumpyIntArrayType,
    all_values: SeqIntArrayType,
    row_ptrs: NumpyIntArrayType,
) -> SeqIntArrayType:
    idx, vals = _get_diff_indices_vals_for_index(
        index=index, all_indices=all_indices, all_values=all_values, row_ptrs=row_ptrs
    )
    seqarray = ref_seq.copy()
    seqarray[idx] = vals
    return seqarray


def _make_pointers(
    *,
    new_indices: list[NumpyIntArrayType],
    old_pointers: Dataset | None = None,
) -> NumpyIntArrayType:
    # we are representing a multiple alignment as a sparse matrix
    # the first row is complete
    # all subsequent rows have only the indices and values of the seq
    # that differ from the first row
    # "pointers" record how many differences per seq and allow us to
    # slice out each "seq" for reassembly
    if old_pointers is None:
        start = 0
        old_pointers = numpy.array([0], dtype=numpy.int64)
    else:
        old_pointers = numpy.asarray(old_pointers)
        start = old_pointers[-1]

    num_diffs = [len(idx) for idx in new_indices]
    # create the offsets given last value from old pointers
    new_offsets = numpy.cumsum(num_diffs, dtype=old_pointers.dtype) + start
    return typing.cast(
        "NumpyIntArrayType", numpy.concatenate([old_pointers, new_offsets])
    )


def _update_sparse_elements(
    *,
    h5file: h5py.File,
    new_elements: list[NumpyIntArrayType],
    grp_name: str,
    dtype: npt.DTypeLike,
) -> NumpyIntArrayType:
    old_elements = h5file.get(grp_name, [])[:]
    if len(old_elements):
        old_elements: list[NumpyIntArrayType] = [old_elements]

    old_elements += new_elements
    return numpy.concatenate(old_elements).astype(dtype)


def _replace_grp_in_file(
    *,
    h5file: h5py.File,
    compression: str | None,
    grp_name: str,
    new_value: NumpyIntArrayType,
) -> None:
    if grp_name in h5file:
        del h5file[grp_name]

    h5file.create_dataset(
        grp_name, data=new_value, compression=compression, shuffle=True
    )


class SparseSeqsData(AlignedSeqsData):
    """sparse alignment data"""

    _diff_idx_grp: str = "diff_indices"
    _diff_val_grp: str = "diff_vals"
    _gapped_grp: str = "gapped"
    _gaps_grp: str = "gaps"
    _seq_ptr_grp: str = "seq_ptrs"
    _suffix: str = SPARSE_SUFFIX
    _ungapped_grp: str = "ungapped"
    _var_pos_grp: str = "variable_posns"

    def __init__(
        self,
        *,
        gapped_seqs: h5py.File,
        alphabet: c3_alphabet.AlphabetABC,
        offset: dict[str, int] | None = None,
        check: bool = True,
        reversed_seqs: frozenset[str] | None = None,
        compression: bool = True,
        ref_name: str = "",
        **kwargs: dict[str, typing.Any],  # noqa: ARG002
    ) -> None:
        super().__init__(
            gapped_seqs=gapped_seqs,
            alphabet=alphabet,
            offset=offset,
            check=check,
            reversed_seqs=reversed_seqs,
            compression=compression,
        )
        stored_ref_name = gapped_seqs.attrs.get("ref_name", "")
        if stored_ref_name and ref_name and stored_ref_name != ref_name:
            msg = f"Reference name {ref_name!r} does not match existing attribute {stored_ref_name!r}"
            raise ValueError(msg)

        self._primary_grp = self._gapped_grp
        self._ref_name = stored_ref_name or ref_name
        self._ref_hash: str = gapped_seqs.attrs.get("ref_hash", "")
        self._seqhashes: dict[str, int] = {}
        self._attr_from_file()

    @property
    def _ref_seq(self) -> Dataset:
        dataset = f"{self._primary_grp}/{self._ref_hash}"
        if dataset not in self._file:
            msg = "Reference sequence not found"
            raise ValueError(msg)
        return typing.cast("Dataset", self._file[dataset])

    @property
    def _seq_ptrs(self) -> Dataset:
        return typing.cast("Dataset", self._file[f"{self._seq_ptr_grp}"])

    @property
    def _diff_vals(self) -> Dataset:
        return typing.cast("Dataset", self._file[f"{self._diff_val_grp}"])

    @property
    def _diff_indices(self) -> Dataset:
        return typing.cast("Dataset", self._file[f"{self._diff_idx_grp}"])

    @property
    def _var_pos(self) -> Dataset:
        return typing.cast("Dataset", self._file[f"{self._var_pos_grp}"])

    def _attr_from_file(self) -> None:
        if not self._ref_hash:
            return

        n2h, h2i = _get_name2hash_hash2idx(self._file)
        self._name_to_hash = n2h
        self._hash_to_index = h2i

    def _set_ref_seq(self, ref_name: str, ref_seq: SeqIntArrayType) -> str:
        self._ref_name = ref_name
        _assign_attr_if_missing(self._file, "ref_name", ref_name)
        self._ref_hash = array_hash64(ref_seq)
        _assign_attr_if_missing(self._file, "ref_hash", self._ref_hash)
        dataset = f"{self._primary_grp}/{self._ref_hash}"
        self._file.create_dataset(
            name=dataset,
            data=ref_seq,
            chunks=True,
            compression=self._compress,
            shuffle=True,
        )
        self._name_to_hash[ref_name] = self._ref_hash
        return self._ref_hash

    @property
    def align_len(self) -> int:
        """length of the alignment"""
        if not self.names:
            return 0
        return typing.cast(
            "numpy.ndarray",
            self._file[f"{self._gapped_grp}/{self._ref_hash}"],
        ).shape[0]

    def _seqs_to_sparse_arrays(
        self,
        seqs: dict[str, StrORBytesORArray],
        force_unique_keys: bool,
        ref_name: str,
    ) -> tuple[
        list[NumpyIntArrayType],
        list[SeqIntArrayType],
        dict[str, str],
        dict[str, int],
        int,
    ]:
        ref_name = self._ref_name or ref_name

        to_indices = self.alphabet.to_indices
        if not self._ref_hash:
            self._populate_attrs()

            if ref_name and ref_name not in self and ref_name not in seqs:
                msg = f"no seqs matching {ref_name!r}"
                raise ValueError(msg)

            ref_name = ref_name or next(iter(seqs))

            ref_seq = typing.cast(
                "SeqIntArrayType", to_indices(seqs.pop(ref_name), validate=True)
            )
            self._name_to_hash = {ref_name: self._set_ref_seq(ref_name, ref_seq)}

        n2h, h2i = _get_name2hash_hash2idx(self._file)
        name_to_hash = self._name_to_hash | n2h
        hash_to_index = self._hash_to_index | h2i

        overlap = name_to_hash.keys() & seqs.keys()
        if force_unique_keys and overlap:
            msg = f"{overlap} already exist in collection"
            raise ValueError(msg)

        seqhash_to_names: dict[str, list[str]] = collections.defaultdict(list)
        for seqid, seqhash in name_to_hash.items():
            seqhash_to_names[seqhash].append(seqid)

        diff_indices = []
        diff_vals = []
        seqhashes = []  # non-refseq hashes
        max_index = 0
        for seqid, seq in seqs.items():
            if overlap and seqid in overlap:
                continue

            seqarray = typing.cast("SeqIntArrayType", to_indices(seq, validate=True))
            seqhash = array_hash64(seqarray)
            name_to_hash[seqid] = seqhash
            if seqhash in seqhash_to_names:
                # same seq, different name
                del seqarray
                continue

            hash_to_index[seqhash] = len(hash_to_index)
            seqhash_to_names[seqhash].append(seqid)
            seqhashes.append(seqhash)
            indices, diffs = _get_indices_diffs(
                typing.cast("SeqIntArrayType", self._ref_seq[:]), seqarray
            )
            max_index = max(max_index, indices.max())
            diff_indices.append(indices)
            diff_vals.append(diffs)

        return diff_indices, diff_vals, name_to_hash, hash_to_index, max_index

    def add_seqs(
        self,
        seqs: dict[str, StrORBytesORArray],
        force_unique_keys: bool = True,
        offset: dict[str, int] | None = None,
        reversed_seqs: frozenset[str] | None = None,
        ref_name: str = "",
        **kwargs: dict[str, typing.Any],
    ) -> "SparseSeqsData":
        if not self.writable:
            msg = "Cannot add sequences to a read-only file"
            raise PermissionError(msg)

        lengths = {len(seq) for seq in seqs.values()}
        if len(lengths) > 1 or (self.align_len and self.align_len not in lengths):
            msg = f"not all lengths equal {lengths=}"
            raise ValueError(msg)

        if self._ref_name and ref_name and ref_name != self._ref_name:
            msg = f"provided {ref_name!r} does not match existing {self._ref_name!r}"
            raise ValueError(msg)

        diff_indices, diff_vals, name_to_hash, hash_to_index, max_index = (
            self._seqs_to_sparse_arrays(
                seqs=seqs,
                force_unique_keys=force_unique_keys,
                ref_name=ref_name,
            )
        )
        if not diff_indices:
            # no unique sequences
            self._populate_optional_grps(
                offset, reversed_seqs, name_to_hash, hash_to_index
            )
            return self

        row_ptrs = _make_pointers(
            old_pointers=self._file.get(self._seq_ptr_grp, None),
            new_indices=diff_indices,
        )
        _replace_grp_in_file(
            h5file=self._file,
            compression=self._compress,
            grp_name=self._seq_ptr_grp,
            new_value=row_ptrs,
        )

        # update the variant indices
        min_dtype = _best_uint_dtype(max_index)
        merged_indices = _update_sparse_elements(
            h5file=self._file,
            new_elements=diff_indices,
            grp_name=self._diff_idx_grp,
            dtype=min_dtype,
        )
        _replace_grp_in_file(
            h5file=self._file,
            compression=self._compress,
            grp_name=self._diff_idx_grp,
            new_value=merged_indices,
        )

        # update the variable position attribute
        _replace_grp_in_file(
            h5file=self._file,
            grp_name=self._var_pos_grp,
            compression=self._compress,
            new_value=numpy.unique(self._diff_indices[:]),
        )

        # update the variant values
        merged_vals = _update_sparse_elements(
            h5file=self._file,
            new_elements=diff_vals,
            grp_name=self._diff_val_grp,
            dtype=numpy.uint8,
        )
        _replace_grp_in_file(
            h5file=self._file,
            compression=self._compress,
            grp_name=self._diff_val_grp,
            new_value=merged_vals,
        )

        if offset := offset or {}:
            _set_offset(
                self._file,
                offset=self.offset | offset,
                compression=self._compress,
            )

        del merged_indices, merged_vals

        self._populate_optional_grps(offset, reversed_seqs, name_to_hash, hash_to_index)
        return self

    def _get_gapped_seq_array(
        self,
        seqid: str,
        start: int,
        stop: int,
        step: int,
    ) -> SeqIntArrayType:
        seqhash = self.get_hash(seqid=seqid)
        if seqhash == self._ref_hash:
            return typing.cast("SeqIntArrayType", self._ref_seq)[start:stop:step]

        index = self._hash_to_index[seqhash]
        seqarray = _inflate_seq(
            index=index,
            ref_seq=typing.cast("SeqIntArrayType", self._ref_seq[:]),
            all_indices=self._diff_indices[:],
            all_values=self._diff_vals[:],
            row_ptrs=self._seq_ptrs[:],
        )
        return seqarray[start:stop:step]

    def get_pos_range(
        self,
        names: PySeqStrType,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> SeqIntArrayType:
        start = start or 0
        stop = stop or self.align_len
        step = step or 1
        if start < 0 or stop < 0 or step < 1:
            msg = f"{start=}, {stop=}, {step=} not >= 1"
            raise ValueError(msg)

        if diff := self._invalid_seqids(names):
            msg = f"these names not present {diff}"
            raise KeyError(msg)

        # we don't apply step yet to make applying diffs more efficient
        array_seqs = numpy.tile(
            typing.cast("SeqIntArrayType", self._ref_seq)[start:stop], (len(names), 1)
        )
        if self._diff_idx_grp not in self._file:
            return array_seqs[:, ::step]

        all_indices = self._diff_indices[:]
        all_values = self._diff_vals[:]
        seq_ptrs = self._seq_ptrs[:]
        for index, name in enumerate(names):
            seqhash = self._name_to_hash[name]
            if seqhash == self._ref_hash:
                continue

            seq_ptr_idx = self._hash_to_index[seqhash]
            indices, diffs = _get_diff_indices_vals_for_index(
                index=seq_ptr_idx,
                all_indices=all_indices,
                all_values=all_values,
                row_ptrs=seq_ptrs,
            )
            # select the indices and vals within start-stop
            within_range = (indices >= start) & (indices < stop)
            # adjust indices for the new start
            indices = indices[within_range] - start
            diffs = diffs[within_range]
            array_seqs[index, indices] = diffs

        if step > 1:
            array_seqs = array_seqs[:, ::step]
        return array_seqs

    def get_positions(
        self,
        names: typing.Sequence[str],
        positions: typing.Sequence[int] | npt.NDArray[numpy.integer],
    ) -> numpy.ndarray[numpy.uint8]:
        """returns alignment positions for names

        Parameters
        ----------
        names
            series of sequence names
        positions
            indices lying within self

        Returns
        -------
            2D numpy.array, oriented by sequence

        Raises
        ------
        IndexError if a provided position is negative or
        greater than the alignment length.
        """
        if not len(positions):
            msg = "must provide positions"
            raise NotImplementedError(msg)

        if diff := self._invalid_seqids(names):
            msg = f"these names not present {diff}"
            raise KeyError(msg)

        min_index, max_index = numpy.min(positions), numpy.max(positions)
        if min_index < 0 or max_index > self.align_len:
            msg = f"Out of range: {min_index=} and / or {max_index=}"
            raise IndexError(msg)

        # we get the hash indices, we don't include the same hash twice
        # we need the hash index order to be sorted
        n2h = self._name_to_hash
        h2i = self._hash_to_index
        ref_hash = self._ref_hash
        ref_present = False
        selected_hashes = {}
        for n in names:
            h = n2h[n]
            if h == ref_hash:
                # no work needed for matches to ref
                ref_present = True
                continue
            i = h2i[h]
            selected_hashes[h] = i

        hash_indices = numpy.array(sorted(selected_hashes.values()), dtype=numpy.int64)
        subalign = extract_subalignment(
            ref_seq=typing.cast("npt.NDArray", self._ref_seq[:]),
            all_indices=typing.cast("npt.NDArray", self._diff_indices[:]),
            all_vals=typing.cast("npt.NDArray", self._diff_vals[:]),
            seq_ptrs=typing.cast("npt.NDArray", self._seq_ptrs[:]),
            seq_ids=hash_indices,
            positions=numpy.array(positions),
            ref_present=ref_present,
        )
        if ref_present:
            selected_hashes[self._ref_hash] = len(self.names)

        seq_indices = names_to_relative_indices(
            names=names,
            subset_hashes=set(selected_hashes.keys()),
            name_to_hash=self._name_to_hash,
            hash_to_index=selected_hashes,
        )
        return subalign[seq_indices]

    def variable_positions(
        self,
        names: PySeqStrType,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> numpy.ndarray[numpy.integer]:
        """returns absolute indices of positions that have more than one state

        Parameters
        ----------
        names
            selected seqids
        start
            absolute start
        stop
            absolute stop
        step
            step

        Returns
        -------
        Absolute indices (as distinct from an index relative to start) of
        variable positions.
        """
        if len(names) < 2 or not self._hash_to_index:
            # no seqs, too few, or all identical
            return numpy.array([], dtype=numpy.int64)

        var_pos = self._var_pos[:]
        if start is None and stop is None and step is None:
            return var_pos

        start = start or 0
        stop = stop or self.align_len
        step = step or 1
        indices = (var_pos >= start) & (var_pos < stop)
        var_pos = var_pos[indices]
        if step > 1:
            var_pos = var_pos[(var_pos - start) % step == 0]
        return var_pos


def names_to_relative_indices(
    *,
    names: typing.Sequence[str],
    subset_hashes: set[str],
    name_to_hash: dict[str, str],
    hash_to_index: dict[str, int],
) -> list[int]:
    """
    returns relative indices for names into their hash subset

    Parameters
    ----------
    names
        sequence names.
    name_to_hash
        {name: sequence hash, ...}
    hash_to_index
        {hash: hash order, ...}

    Returns
    -------
    Relative indices that map names onto the subset of hashes.
    """
    ordered_hashes = sorted(subset_hashes, key=lambda h: hash_to_index[h])
    hash_to_rel = {h: i for i, h in enumerate(ordered_hashes)}
    # translate names into relative indices
    return [hash_to_rel[name_to_hash[n]] for n in names]


@numba.njit(cache=True, nogil=True)
def extract_subalignment(
    ref_seq: npt.NDArray[numpy.uint8],
    all_indices: npt.NDArray[numpy.integer],
    all_vals: npt.NDArray[numpy.integer],
    seq_ptrs: npt.NDArray[numpy.integer],
    seq_ids: npt.NDArray[numpy.integer],
    positions: npt.NDArray[numpy.integer],
    ref_present: bool,
) -> SeqIntArrayType:  # pragma: no cover
    """
    Extracts a dense subalignment matrix from sparse CSR-like MSA,
    optimized for when `positions` is sorted.

    Parameters
    ----------
    ref_seq
        Reference sequence.
    all_indices
        Concatenated positions of differences (sorted within each seq).
    all_vals
        Values corresponding to `all_indices`.
    seq_ptrs
        CSR row pointer array
    seq_ids
        Sequence indices to extract.
    positions
        Alignment column positions to extract (must be sorted).
    ref_present
        the reference sequence is present, this will be the last
        row in the result object

    Returns
    -------
    numpy.ndarray (uint8) of shape (len(seq_ids), len(positions))
    """
    num_seqs = len(seq_ids)
    num_pos = len(positions)

    # initialize with reference sequence values
    ref_vals = ref_seq[positions]
    num_rows = num_seqs + 1 if ref_present else num_seqs
    result = numpy.empty((num_rows, num_pos), dtype=numpy.uint8)
    result[:, numpy.arange(ref_vals.size)] = ref_vals

    for i in range(num_seqs):
        seq_id = seq_ids[i]
        start = seq_ptrs[seq_id]
        end = seq_ptrs[seq_id + 1]

        idxs = all_indices[start:end]
        vals = all_vals[start:end]

        # a merge scan assuming positions and idxs are sorted
        idx_ptr = 0  # index in idxs
        pos_ptr = 0  # index in positions

        while idx_ptr < len(idxs) and pos_ptr < num_pos:
            pos = positions[pos_ptr]

            if idxs[idx_ptr] < pos:
                idx_ptr += 1
            elif idxs[idx_ptr] > pos:
                pos_ptr += 1
            else:
                # populate this sequence difference
                result[i, pos_ptr] = vals[idx_ptr]
                idx_ptr += 1
                pos_ptr += 1

    return result


@numba.njit(cache=True, nogil=True)
def remove_gaps(arr, gap_index, missing_index=-1):  # pragma: no cover
    nrows, ncols = arr.shape
    num_non_gaps = numpy.empty(nrows, dtype=numpy.int32)
    if missing_index == -1:
        missing_index = gap_index

    for i in range(nrows):
        write_pos = 0
        for j in range(ncols):
            val = arr[i, j]
            if val not in (gap_index, missing_index):
                arr[i, write_pos] = val
                write_pos += 1
        num_non_gaps[i] = write_pos
    return arr, num_non_gaps


@functools.singledispatch
def make_unaligned(
    path: str | pathlib.Path | None,
    *,
    data=None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
) -> UnalignedSeqsData:
    msg = f"make_unaligned not implemented for {type(path)}"
    raise TypeError(msg)


@make_unaligned.register
def _(
    path: str,
    *,
    data=None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
) -> UnalignedSeqsData:
    h5file = open_h5_file(path=path, mode=mode, in_memory=in_memory)
    if (mode != "r" or in_memory) and alphabet is None:
        msg = "alphabet must be provided for write mode"
        raise ValueError(msg)

    if alphabet is None:
        mt = c3_moltype.get_moltype(h5file.attrs.get("moltype"))
        alphabet = _restore_alphabet(
            chars=h5file.attrs.get("alphabet"),
            gap=h5file.attrs.get("gap_char"),
            missing=h5file.attrs.get("missing_char"),
            moltype=mt,
        )
    check = h5file.mode == "r" if check is None else check

    useqs = UnalignedSeqsData(
        data=h5file,
        alphabet=alphabet,
        offset=offset,
        reversed_seqs=reversed_seqs,
        check=check,
        compression=compression,
    )
    useqs.filename_suffix = suffix
    if data is not None:
        _ = useqs.add_seqs(seqs=data, offset=offset, reversed_seqs=reversed_seqs)
    return useqs


@make_unaligned.register
def _(
    path: pathlib.Path,
    *,
    data=None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
) -> UnalignedSeqsData:
    return make_unaligned(
        str(path.expanduser()),
        data=data,
        mode=mode,
        in_memory=in_memory,
        alphabet=alphabet,
        offset=offset,
        reversed_seqs=reversed_seqs,
        check=check,
        suffix=suffix,
        compression=compression,
    )


@make_unaligned.register
def _(
    path: None,
    *,
    data=None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
) -> UnalignedSeqsData:
    # create a writeable in memory record
    mode = "w"
    in_memory = True
    return make_unaligned(
        "memory",
        data=data,
        mode=mode,
        in_memory=in_memory,
        alphabet=alphabet,
        offset=offset,
        reversed_seqs=reversed_seqs,
        check=check,
        suffix=suffix,
        compression=compression,
    )


def make_aligned(
    path: str,
    *,
    data=None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = ALIGNED_SUFFIX,
    compression: bool = True,
    ref_name: str = "",
    sparse: bool = False,
) -> AlignedSeqsData | SparseSeqsData:
    h5file = open_h5_file(path=path, mode=mode, in_memory=in_memory)
    if (mode != "r" or in_memory) and alphabet is None:
        msg = "alphabet must be provided for write mode"
        raise ValueError(msg)

    if alphabet is None:
        mt = c3_moltype.get_moltype(h5file.attrs.get("moltype"))
        alphabet = _restore_alphabet(
            chars=h5file.attrs["alphabet"],
            gap=h5file.attrs["gap_char"],
            missing=h5file.attrs["missing_char"],
            moltype=mt,
        )
    check = h5file.mode == "r" if check is None else check
    cls = SparseSeqsData if sparse else AlignedSeqsData
    kwargs = {"ref_name": ref_name} if sparse else {}
    asd = cls(
        gapped_seqs=h5file,
        check=check,
        alphabet=alphabet,
        offset=offset,
        reversed_seqs=reversed_seqs,
        compression=compression,
        **kwargs,
    )

    asd.filename_suffix = suffix
    if data is not None:
        _ = asd.add_seqs(seqs=data, offset=offset, reversed_seqs=reversed_seqs)
    return asd


def make_sparse(*args, **kwargs) -> SparseSeqsData:
    kwargs["sparse"] = True
    kwargs["suffix"] = kwargs.get("suffix", SPARSE_SUFFIX)
    return make_aligned(*args, **kwargs)


_aligned_makers = {AlignedSeqsData: make_aligned, SparseSeqsData: make_sparse}


def load_seqs_data_unaligned(
    path: str | pathlib.Path,
    mode: str = "r",
    check: bool = True,
    suffix: str = UNALIGNED_SUFFIX,
) -> UnalignedSeqsData:
    """load hdf5 unaligned sequence data from file"""
    path = pathlib.Path(path)
    if path.suffix != f".{suffix}":
        msg = f"File {path} does not have an expected suffix {suffix!r}"
        raise ValueError(msg)
    klass = UnalignedSeqsData
    result = klass.from_file(path=path, mode=mode, check=check)
    result.filename_suffix = suffix
    return result


def load_seqs_data_aligned(
    path: str | pathlib.Path,
    mode: str = "r",
    check: bool = True,
    suffix: str = ALIGNED_SUFFIX,
) -> AlignedSeqsData:
    """load hdf5 aligned sequence data from file"""
    path = pathlib.Path(path)
    if path.suffix != f".{suffix}":
        msg = f"File {path} does not have an expected suffix {suffix!r}"
        raise ValueError(msg)
    klass = AlignedSeqsData

    result = klass.from_file(path=path, mode=mode, check=check)
    result.filename_suffix = suffix
    return result


def load_seqs_data_sparse(
    path: str | pathlib.Path,
    mode: str = "r",
    check: bool = True,
    suffix: str = SPARSE_SUFFIX,
) -> SparseSeqsData:
    """load hdf5 aligned sequence data from file"""
    path = pathlib.Path(path)
    if path.suffix != f".{suffix}":
        msg = f"File {path} does not have an expected suffix {suffix!r}"
        raise ValueError(msg)
    klass = SparseSeqsData

    result = klass.from_file(path=path, mode=mode, check=check)
    result.filename_suffix = suffix
    return result


def write_seqs_data(
    *,
    path: pathlib.Path,
    seqcoll: SeqCollTypes,
    unaligned_suffix: str = UNALIGNED_SUFFIX,
    aligned_suffix: str = ALIGNED_SUFFIX,
    sparse_suffix: str = SPARSE_SUFFIX,
    **kwargs,
) -> pathlib.Path:
    path = pathlib.Path(path)
    supported_suffixes = {
        aligned_suffix: c3_alignment.Alignment,
        unaligned_suffix: c3_alignment.SequenceCollection,
        sparse_suffix: c3_alignment.Alignment,
    }
    suffix = path.suffix[1:]
    if suffix not in supported_suffixes:
        msg = f"path {path} does not have a supported suffix {supported_suffixes}"
        raise ValueError(msg)

    if type(seqcoll) is not supported_suffixes[suffix]:
        msg = f"{suffix=} invalid for {type(seqcoll).__name__!r}"
        raise TypeError(msg)

    # check that the collection is modified relative to the underlying storage
    # this will be names of collection and storage are not equal
    # slice_record of Alignment is not generic

    if isinstance(seqcoll.storage, AlignedSeqsData | SparseSeqsData):
        # we want aligned data to remain compact
        no_gap_data = "gaps" not in seqcoll.storage.h5file
    else:
        no_gap_data = False

    if (
        no_gap_data
        and not seqcoll.modified
        and isinstance(
            seqcoll.storage, (UnalignedSeqsData, AlignedSeqsData | SparseSeqsData)
        )
    ):
        storage = seqcoll.storage
        storage.h5file.flush()
        image = storage.h5file.id.get_file_image()
        with open(path, "wb") as out_file:
            out_file.write(image)
        return path

    # the following results in storing the primary data only, gapped sequences
    # in the case of an alignment
    cls = UnalignedSeqsData if suffix == unaligned_suffix else AlignedSeqsData
    alphabet = seqcoll.storage.alphabet
    data = {s.name: numpy.array(s) for s in seqcoll.seqs}
    offset = seqcoll.storage.offset
    reversed_seqs = seqcoll.storage.reversed_seqs
    kwargs = {
        "data": data,
        "alphabet": alphabet,
        "offset": offset,
        "reversed_seqs": reversed_seqs,
    } | kwargs
    store = cls.from_seqs(**kwargs)
    store.filename_suffix = suffix
    store.write(path=path)
    return path


class H5SeqsUnalignedParser(SequenceParserBase):
    @property
    def name(self) -> str:
        return "c3h5u"

    @property
    def supports_unaligned(self) -> bool:
        """True if the loader supports unaligned sequences"""
        return True

    @property
    def supports_aligned(self) -> bool:
        """True if the loader supports aligned sequences"""
        return False

    @property
    def supported_suffixes(self) -> set[str]:
        return {UNALIGNED_SUFFIX}

    @property
    def result_is_storage(self) -> bool:
        return True

    @property
    def loader(
        self,
    ) -> typing.Callable[[pathlib.Path], UnalignedSeqsData | AlignedSeqsData]:
        return load_seqs_data_unaligned


class H5SeqsAlignedParser(SequenceParserBase):
    @property
    def name(self) -> str:
        return ALIGNED_SUFFIX

    @property
    def supports_unaligned(self) -> bool:
        """True if the loader supports unaligned sequences"""
        return False

    @property
    def supports_aligned(self) -> bool:
        """True if the loader supports aligned sequences"""
        return True

    @property
    def supported_suffixes(self) -> set[str]:
        return {ALIGNED_SUFFIX}

    @property
    def result_is_storage(self) -> bool:
        return True

    @property
    def loader(
        self,
    ) -> typing.Callable[[pathlib.Path], AlignedSeqsData]:
        return load_seqs_data_aligned


class H5SeqsSparseParser(H5SeqsAlignedParser):
    @property
    def name(self) -> str:
        return SPARSE_SUFFIX

    @property
    def supported_suffixes(self) -> set[str]:
        return {SPARSE_SUFFIX}

    @property
    def loader(
        self,
    ) -> typing.Callable[[pathlib.Path], SparseSeqsData]:
        return load_seqs_data_sparse


class H5UnalignedSeqsWriter(SequenceWriterBase):
    @property
    def name(self) -> str:
        return UNALIGNED_SUFFIX

    @property
    def supports_unaligned(self) -> bool:
        """True if the loader supports unaligned sequences"""
        return True

    @property
    def supports_aligned(self) -> bool:
        """True if the loader supports aligned sequences"""
        return False

    @property
    def supported_suffixes(self) -> set[str]:
        return {UNALIGNED_SUFFIX}

    def write(
        self,
        *,
        path: pathlib.Path,
        seqcoll: SeqCollTypes,
        **kwargs,
    ) -> pathlib.Path:
        path = pathlib.Path(path)
        kwargs.pop("order", None)
        return write_seqs_data(
            path=path,
            seqcoll=seqcoll,
            **kwargs,
        )


class H5AlignedSeqsWriter(H5UnalignedSeqsWriter):
    @property
    def name(self) -> str:
        return ALIGNED_SUFFIX

    @property
    def supports_unaligned(self) -> bool:
        """True if the loader supports unaligned sequences"""
        return False

    @property
    def supports_aligned(self) -> bool:
        """True if the loader supports aligned sequences"""
        return True

    @property
    def supported_suffixes(self) -> set[str]:
        return {ALIGNED_SUFFIX}


class H5SparseSeqsWriter(H5UnalignedSeqsWriter):
    @property
    def name(self) -> str:
        return SPARSE_SUFFIX

    @property
    def supports_unaligned(self) -> bool:
        """True if the loader supports unaligned sequences"""
        return False

    @property
    def supports_aligned(self) -> bool:
        """True if the loader supports aligned sequences"""
        return True

    @property
    def supported_suffixes(self) -> set[str]:
        return {SPARSE_SUFFIX}

    def write(
        self,
        *,
        path: pathlib.Path,
        seqcoll: SeqCollTypes,
        **kwargs,
    ) -> pathlib.Path:
        return super().write(
            path=path,
            seqcoll=seqcoll,
            sparse=True,
            **kwargs,
        )
