"""Unaligned sequence data storage."""

import collections
import contextlib
import functools
import pathlib
import pickle
import typing

import h5py
import numpy
import typing_extensions
from cogent3.core import alignment as c3_alignment
from cogent3.core import alphabet as c3_alphabet
from cogent3.core import moltype as c3_moltype

from .seq_transform import pack_nucleic, unpack_nucleic
from .util import (
    DEFAULT_COMPRESSION,
    UNALIGNED_SUFFIX,
    SeqDataView,
    SeqIntArrayType,
    StrORBytesORArray,
    _assign_alphabet_if_missing,
    _assign_attr_if_missing,
    _data_from_file,
    _get_name2hash_hash2idx,
    _restore_alphabet,
    _set_name_to_hash_to_index,
    _set_offset,
    _set_reversed_seqs,
    _valid_h5seqs,
    _writeable_modes,
    array_hash64,
    duplicate_h5_file,
    open_h5_file,
)


class UnalignedSeqsData(c3_alignment.SeqsDataABC):
    """HDF5 storage for unaligned (variable-length) sequences.

    Sequences are deduplicated by xxhash64 digest and optionally 2-bit
    encoded for DNA/RNA moltypes.

    Parameters
    ----------
    data
        An open HDF5 file used as the backing store.
    alphabet
        cogent3 alphabet instance for the sequence moltype.
    offset
        Mapping of sequence names to integer annotation offsets.
    check
        If True, validate the HDF5 file structure.
    reversed_seqs
        Set of sequence names that are reverse-complemented.
    compression
        If True, compress datasets with lzf.
    packed
        If True, apply 2-bit encoding for nucleic acid sequences. Default
        is None (auto-detect from alphabet).
    """

    _ungapped_grp: str = "ungapped"
    _noncanonical_grp: str = "noncanonical"
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
        packed: bool | None = None,
    ) -> None:
        self._compress = DEFAULT_COMPRESSION if compression else None
        self._alphabet: c3_alphabet.CharAlphabet = alphabet
        self._file: h5py.File = data
        self._primary_grp: str = self._ungapped_grp

        reversed_seqs = frozenset(reversed_seqs or frozenset())
        offset = offset or {}
        self._attr_set: bool = False
        self._name_to_hash: dict[str, str] = {}
        self._hash_to_index: dict[str, int] = {}
        self._index_loaded: bool = False
        self._offset_cache: dict[str, int] | None = dict(offset) if offset else None
        self._reversed_cache: frozenset[str] | None = (
            reversed_seqs if reversed_seqs else None
        )
        self._dirty: bool = bool(offset) or bool(reversed_seqs)

        # Determine whether to use 2-bit packed encoding
        # Packing only applies to UnalignedSeqsData
        can_pack = type(self) is UnalignedSeqsData
        if packed is None:
            self._is_packed = bool(
                data.attrs.get("packed", False)
                or (
                    can_pack
                    and data.mode != "r"
                    and getattr(alphabet.moltype, "is_nucleic", False)
                )
            )
        else:
            self._is_packed = packed and can_pack

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
        if not self._index_loaded:
            self._ensure_index()
        parts = ", ".join(attr_vals)
        return f"{name}({parts}, num_seqs={len(self._name_to_hash)})"

    def _invalid_seqids(self, seqids: typing.Sequence[str]) -> set[str]:
        """returns seqids not present in self.names"""
        return set(seqids) - set(self.names)

    @classmethod
    def _check_file(cls, file: h5py.File) -> None:
        if not _valid_h5seqs(file, cls._ungapped_grp):
            msg = f"File {file} is not a valid {cls.__name__} file"
            raise ValueError(msg)

    @classmethod
    def new_type(cls, file: h5py.File) -> bool:
        if not file.keys() or _valid_h5seqs(file, cls._ungapped_grp):
            # no keys means no groups
            return True

        if cls._ungapped_grp in file:
            return False

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
        _assign_attr_if_missing(data, "packed", self._is_packed)
        self._attr_set = True

    def _ensure_index(self) -> None:
        """Lazily decode the on-disk name_to_hash into in-memory caches."""
        if self._index_loaded:
            return
        n2h, h2i = _get_name2hash_hash2idx(self._file)
        # dict insertion order matches the on-disk dataset order
        self._name_to_hash = n2h
        self._hash_to_index = h2i
        self._index_loaded = True

    def _populate_optional_grps(
        self,
        offset: dict[str, int] | None,
        reversed_seqs: frozenset[str] | None,
        name_to_hash: dict[str, str],
        hash_to_index: dict[str, int],
    ) -> None:
        # in-memory state is the source of truth, flush() writes to disk
        self._name_to_hash |= name_to_hash
        self._hash_to_index |= hash_to_index
        if offset:
            merged = (self._offset_cache or {}) | offset
            self._offset_cache = merged
        if reversed_seqs:
            existing = self._reversed_cache or frozenset()
            self._reversed_cache = existing | frozenset(reversed_seqs)
        self._dirty = True

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
            # the contains method triggers loading of _name_to_hash
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
        # __del__ may run during interpreter shutdown when h5py's internal
        # locks have been torn down to None — accessing self._file.id then
        # raises TypeError from h5py's context-manager guard. Suppress all
        # exceptions: the OS will reclaim file descriptors on exit anyway.
        with contextlib.suppress(Exception):
            self._del_impl()

    def _del_impl(self) -> None:
        if not (getattr(self, "_file", None) and self._file.id):
            return

        # best-effort flush of any deferred writes, getattr-guarded so a
        # partially-initialised instance (where __init__ raised before
        # assigning _dirty) does not raise AttributeError from __del__
        if getattr(self, "_dirty", False):
            with contextlib.suppress(Exception):
                self.flush()

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
        if not self._index_loaded:
            self._ensure_index()
        return seqid in self._name_to_hash

    def __getitem__(self, index: str | int) -> SeqDataView:
        if isinstance(index, int):
            return self[self.names[index]]

        if isinstance(index, str):
            return self.get_view(index)

        msg = f"__getitem__ not implemented for {type(index)}"
        raise TypeError(msg)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def alphabet(self) -> c3_alphabet.CharAlphabet:
        return self._alphabet

    @property
    def names(self) -> tuple[str, ...]:
        if not self._index_loaded:
            self._ensure_index()
        # dict insertion order matches the on-disk dataset row order, so this
        # preserves the canonical sequence ordering without re-decoding bytes
        return tuple(self._name_to_hash)

    @property
    def offset(self) -> dict[str, int]:
        if self._offset_cache is None:
            offsets: dict[str, int] = {}
            if "offset" in self._file:
                data = typing.cast("numpy.ndarray", self._file["offset"])[:]
                offsets = {k.decode("utf8"): int(v) for k, v in data}
            self._offset_cache = offsets
        all_offsets = typing.cast("dict[str, int]", collections.defaultdict(int))
        return all_offsets | self._offset_cache

    @property
    def reversed_seqs(self) -> frozenset[str]:
        if self._reversed_cache is None:
            if "reversed_seqs" in self._file:
                data = typing.cast("numpy.ndarray", self._file["reversed_seqs"])[:]
                self._reversed_cache = frozenset(v.decode("utf8") for v in data)
            else:
                self._reversed_cache = frozenset()
        return self._reversed_cache

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
        # commit any deferred writes before another reader observes the file
        self.flush()
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
            packed=self._is_packed,
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
        if self._is_packed:
            packed, ambig_posns, num_canon = pack_nucleic(seqarray)
            dataset = self._file.create_dataset(
                name=f"{self._primary_grp}/{seqhash}",
                data=packed,
                compression=self._compress,
                shuffle=True,
            )
            dataset.attrs["length"] = len(seqarray)
            dataset.attrs["num_canonical"] = num_canon
            self._file.create_dataset(
                name=f"{self._noncanonical_grp}/{seqhash}",
                data=ambig_posns,
                compression=self._compress,
                shuffle=True,
            )
        else:
            dataset = self._file.create_dataset(
                name=f"{self._primary_grp}/{seqhash}",
                data=seqarray,
                compression=self._compress,
                shuffle=True,
            )
            dataset.attrs["length"] = len(seqarray)

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
        if not self._index_loaded:
            self._ensure_index()
        name_to_hash = dict(self._name_to_hash)
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

        seqhash = self.get_hash(seqid=seqid)
        dataset_name = f"{self._ungapped_grp}/{seqhash}"

        if self._is_packed:
            # Pass h5py dataset reference - slicing inside unpack_nucleic loads
            # only the minimal required bytes from disk
            packed_dataset = self._file[dataset_name]
            num_canon = packed_dataset.attrs["num_canonical"]
            ambig_posns = self._file[f"{self._noncanonical_grp}/{seqhash}"][:]
            result = unpack_nucleic(packed_dataset, ambig_posns, num_canon, start, stop)
            if step > 1:
                result = result[::step]
            return result

        out_len = (stop - start + step - 1) // step
        out = numpy.empty(out_len, dtype=self.alphabet.dtype)
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

    def get_view(self, seqid: str) -> SeqDataView:
        return SeqDataView(
            parent=self,
            seqid=seqid,
            parent_len=self.get_seq_length(seqid=seqid),
            alphabet=self.alphabet,
        )

    def get_seq_length(self, seqid: str) -> int:
        """Returns the length of the sequence"""
        dataset_name = f"{self._ungapped_grp}/{self.get_hash(seqid=seqid)}"
        dataset = self._file[dataset_name]
        if "length" in dataset.attrs:
            return int(dataset.attrs["length"])
        return typing.cast("numpy.ndarray", dataset).shape[0]

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
        if not cls.new_type(h5file):
            data = _data_from_file(h5file, cls._ungapped_grp)
        else:
            data = None

        alphabet = _restore_alphabet(
            chars=h5file.attrs.get("alphabet"),
            moltype=c3_moltype.get_moltype(h5file.attrs.get("moltype")),
            missing=h5file.attrs.get("missing_char") or None,
            gap=h5file.attrs.get("gap_char") or None,
        )

        result = cls(data=h5file, alphabet=alphabet, check=check)
        if data:
            result = result.add_seqs(data)
        return result

    def _write(self, path: str | pathlib.Path, exclude_groups: set[str]) -> None:
        # commit any deferred writes so duplicate_h5_file sees the latest state
        self.flush()
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

    def flush(self) -> None:
        """Write any dirty in-memory caches to the HDF5 file."""
        if not self._dirty or not self.writable:
            return
        if self._index_loaded:
            _set_name_to_hash_to_index(
                self._file,
                {
                    k: (h, self._hash_to_index.get(h, -1))
                    for k, h in self._name_to_hash.items()
                },
                compression=DEFAULT_COMPRESSION,
            )
        _set_offset(
            self._file,
            offset=self._offset_cache or {},
            compression=DEFAULT_COMPRESSION,
        )
        _set_reversed_seqs(
            self._file,
            self._reversed_cache or frozenset(),
            compression=DEFAULT_COMPRESSION,
        )
        # subclass hook for sparse aggregate datasets, no-op on the base
        self._flush_extras()
        # ensure h5py commits its own buffered metadata so a second handle
        # opened on the same on-disk path observes the writes
        self._file.flush()
        self._dirty = False

    def _flush_extras(self) -> None:
        """Subclass hook. ``SparseSeqsData`` overrides to flush its diff
        arrays alongside the metadata datasets.
        """

    def close(self) -> None:
        """close the HDF file"""
        if not (self._file and self._file.id):
            return

        if not self._attr_set:
            self._populate_attrs()

        self.flush()
        self._file.close()


@functools.singledispatch
def make_unaligned(
    path: str | pathlib.Path | None,
    *,
    data: h5py.File | None = None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
    packed: bool | None = None,
) -> UnalignedSeqsData:
    """Create or load an UnalignedSeqsData instance.

    Dispatches on the type of ``path``: a string or Path opens (or creates)
    an on-disk HDF5 file, while None creates a writeable in-memory store.

    Parameters
    ----------
    path
        Filesystem path to an HDF5 file, or None for in-memory storage.
    data
        Optional mapping of sequence names to numpy arrays to add on
        creation.
    mode
        HDF5 file open mode (e.g. ``"r"``, ``"w"``).
    in_memory
        If True, use the HDF5 core driver for in-memory storage.
    alphabet
        cogent3 alphabet instance. Required when opening in write mode.
    offset
        Mapping of sequence names to integer annotation offsets.
    reversed_seqs
        Set of sequence names that are reverse-complemented.
    check
        If True, validate the HDF5 file structure on load.
    suffix
        Filename suffix for the storage type.
    compression
        If True, compress datasets with lzf.
    packed
        If True, apply 2-bit encoding for nucleic acid sequences. Default
        is None (auto-detect from alphabet).

    Returns
    -------
    UnalignedSeqsData
    """
    msg = f"make_unaligned not implemented for {type(path)}"
    raise TypeError(msg)


@make_unaligned.register
def _(
    path: str,
    *,
    data: h5py.File | None = None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
    packed: bool | None = None,
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
        packed=packed,
    )
    useqs.filename_suffix = suffix
    if data is not None:
        _ = useqs.add_seqs(seqs=data, offset=offset, reversed_seqs=reversed_seqs)
    return useqs


@make_unaligned.register
def _(
    path: pathlib.Path,
    *,
    data: h5py.File | None = None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
    packed: bool | None = None,
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
        packed=packed,
    )


@make_unaligned.register
def _(
    path: None,
    *,
    data: h5py.File | None = None,
    mode: str = "r",
    in_memory: bool = False,
    alphabet: c3_alphabet.AlphabetABC | None = None,
    offset: dict[str, int] | None = None,
    reversed_seqs: frozenset[str] | None = None,
    check: bool = False,
    suffix: str = UNALIGNED_SUFFIX,
    compression: bool = True,
    packed: bool | None = None,
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
        packed=packed,
    )


def load_seqs_data_unaligned(
    path: str | pathlib.Path,
    mode: str = "r",
    check: bool = True,
    suffix: str = UNALIGNED_SUFFIX,
) -> UnalignedSeqsData:
    """Load unaligned sequence data from an HDF5 file.

    Parameters
    ----------
    path
        Path to a ``.c3h5u`` file.
    mode
        HDF5 file open mode.
    check
        If True, validate the HDF5 file structure on load.
    suffix
        Expected filename suffix.

    Returns
    -------
    UnalignedSeqsData

    Raises
    ------
    ValueError
        If the file suffix does not match ``suffix``.
    """
    path = pathlib.Path(path)
    if path.suffix != f".{suffix}":
        msg = f"File {path} does not have an expected suffix {suffix!r}"
        raise ValueError(msg)

    klass = UnalignedSeqsData
    result = klass.from_file(path=path, mode=mode, check=check)
    result.filename_suffix = suffix
    return result
