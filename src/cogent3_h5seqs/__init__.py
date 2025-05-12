import functools
import hashlib
import pathlib
import typing
import uuid

import h5py
import hdf5plugin
import numpy
import typing_extensions
from cogent3.core import new_alignment as c3_alignment
from cogent3.core import new_alphabet as c3_alphabet
from cogent3.core import new_moltype as c3_moltype
from cogent3.core import new_sequence as c3_sequence
from cogent3.format.sequence import SequenceWriterBase
from cogent3.parse.sequence import SequenceParserBase

__version__ = "0.3.0"

if typing.TYPE_CHECKING:  # pragma: no cover
    from cogent3.core.new_alignment import Alignment, SequenceCollection


UNALIGNED_SUFFIX = "c3h5u"
ALIGNED_SUFFIX = "c3h5a"

HDF5_BLOSC2_KWARGS = hdf5plugin.Blosc2(
    cname="blosclz",
    clevel=9,
    filters=hdf5plugin.Blosc2.BITSHUFFLE,
)

SeqsTypes = typing.Union["SequenceCollection", "Alignment"]
StrORBytesORArray = str | bytes | numpy.ndarray
OptInt = int | None
OptStr = str | None

# for storing large dicts in HDF5
offset_dtype = numpy.dtype(
    [("key", h5py.special_dtype(vlen=bytes)), ("value", numpy.int64)]
)
# HDF5 file modes
# x and w- mean create file, fail if exists
# r+ means read/write, file must exist
# w creates file, truncate if exists
# a means append, create if not exists
_writeable_modes = {"r+", "w", "w-", "x", "a"}


def array_hash64(data: numpy.ndarray) -> int:
    """returns 64-bit hash of numpy array.

    Notes
    -----
    This function does not introduce randomisation and so
    is reproducible between processes.
    """
    # if this proves a performance bottleneck, we can
    # switch to using xxhash.hash64()
    h = hashlib.md5(data.tobytes(), usedforsecurity=False)
    return int.from_bytes(h.digest()[:8], byteorder="little")


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


def _assign_attr_if_missing(
    h5file: h5py.File, attr: str, value: typing_extensions.Any
) -> bool:
    if attr not in h5file.attrs:
        h5file.attrs[attr] = value
    return h5file.attrs[attr] == value


def _valid_h5seqs(h5file: h5py.File, main_seq_grp: str) -> bool:
    # essential attributes, groups
    return all(
        [
            "alphabet" in h5file.attrs,
            "moltype" in h5file.attrs,
            main_seq_grp in h5file,
            "gap_char" in h5file.attrs,
            "missing_char" in h5file.attrs,
        ]
    )


def _set_group(h5file: h5py.File, group_name: str, value: numpy.ndarray) -> None:
    if group_name in h5file:
        del h5file[group_name]

    h5file.create_dataset(
        name=group_name,
        data=value,
        chunks=True,
        **HDF5_BLOSC2_KWARGS,
    )


def _set_offset(h5file: h5py.File, offset: dict[str, int] | None) -> None:
    # set the offset as a special group
    if not offset or h5file.mode not in _writeable_modes:
        return

    data = numpy.array(
        [(k.encode("utf8"), v) for k, v in offset.items()], dtype=offset_dtype
    )
    _set_group(h5file, "offset", data)


def _set_reversed_seqs(h5file: h5py.File, reverse_seqs: frozenset[str] | None) -> None:
    # set the reverse seqs as a special group
    if not reverse_seqs or h5file.mode not in _writeable_modes:
        return

    data = numpy.array([s.encode("utf8") for s in reverse_seqs], dtype="S")
    _set_group(h5file, "reversed_seqs", data)


def duplicate_h5_file(
    *, h5file: h5py.File, path: str | pathlib.Path, in_memory: bool
) -> h5py.File:
    result = open_h5_file(path=path, mode="w", in_memory=in_memory)
    for grp in h5file:
        h5file.copy(grp, result)
    for attr in h5file.attrs:
        result.attrs[attr] = h5file.attrs[attr]
    return result


class UnalignedSeqsData(c3_alignment.SeqsDataABC):
    _ungapped_grp: str = "ungapped"

    def __init__(
        self,
        *,
        data: h5py.File,
        alphabet: c3_alphabet.AlphabetABC,
        offset: dict[str, int] | None = None,
        check: bool = False,
        reversed_seqs: frozenset[str] | None = None,
    ) -> None:
        self._alphabet = alphabet
        self._file = data
        self._primary_grp = self._ungapped_grp

        reversed_seqs = reversed_seqs or frozenset()
        _set_reversed_seqs(self._file, reversed_seqs)
        offset = offset or {}
        _set_offset(self._file, offset=offset)
        self._attr_set = False
        self._hashes = None
        if check:
            self._check_file(self._file)

    @classmethod
    def _check_file(cls, file: h5py.File) -> None:
        if not _valid_h5seqs(file, cls._ungapped_grp):
            msg = f"File {file} is not a valid {cls.__name__} file"
            raise ValueError(msg)

    def _populate_attrs(self) -> None:
        if self._attr_set:
            return
        data = self._file
        _assign_attr_if_missing(data, "alphabet", "".join(self._alphabet))
        _assign_attr_if_missing(data, "gap_char", self._alphabet.gap_char)
        _assign_attr_if_missing(data, "missing_char", self._alphabet.missing_char)
        _assign_attr_if_missing(data, "moltype", self._alphabet.moltype.name)
        self._attr_set = True

    def get_hash(self, seqid: str) -> str | None:
        """returns 64-bit hash for seqid"""
        if seqid not in self:
            return None

        self._hashes = self._hashes or {}
        if seqid not in self._hashes:
            # we use the self indexing to get a view, which then returns
            # the correct array for the storage type (unaligned, aligned)
            seq = numpy.array(self[seqid])
            self._hashes[seqid] = array_hash64(seq)

        return self._hashes[seqid]

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
            msg = f"Cannot set attribute {attr_name!r} to {attr_value!r} with type {type(attr_value)=}"
            raise TypeError(msg) from e

    def get_attr(self, attr_name: str) -> str:
        """get attr_name from the file"""
        if attr_name not in self._file.attrs:
            msg = f"attribute {attr_name!r} not found"
            raise KeyError(msg)
        return self._file.attrs[attr_name]

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
        other: typing_extensions.Self,
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
        return all(
            self.get_hash(seqid=name) == other.get_hash(seqid=name)
            for name in self.names
        )

    def __ne__(
        self,
        other: typing_extensions.Self,
    ) -> bool:
        return not (self == other)

    def __contains__(self, seqid: str) -> bool:
        """seqid in self"""
        group_ref = self._file.get(self._primary_grp, {})
        return seqid in group_ref

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
    def alphabet(self) -> c3_alphabet.AlphabetABC:
        return self._alphabet

    @property
    def names(self) -> tuple[str, ...]:
        grp = self._primary_grp
        return tuple(self._file[grp]) if grp in self._file else ()

    @property
    def offset(self) -> dict[str, int]:
        all_offsets = dict.fromkeys(self.names, 0)
        if "offset" not in self._file:
            return all_offsets
        data = self._file["offset"][:]

        return all_offsets | {k.decode("utf8"): int(v) for k, v in data}

    @property
    def reversed_seqs(self) -> frozenset[str]:
        if "reversed_seqs" not in self._file:
            return frozenset()

        data = self._file["reversed_seqs"][:]
        return frozenset(v.decode("utf8") for v in data)

    def _make_new_h5_file(
        self,
        data: h5py.File,
        alphabet: c3_alphabet.CharAlphabet | None,
        offset: dict[str, int] | None,
        reversed_seqs: set[str] | None,
    ) -> tuple[h5py.File, c3_alphabet.CharAlphabet, dict[str, int], set[str]]:
        if data is None:
            data = duplicate_h5_file(h5file=self._file, path="memory", in_memory=True)
        alphabet = alphabet or self.alphabet

        reversed_seqs = reversed_seqs or self.reversed_seqs
        if alphabet and alphabet != self.alphabet:
            data.attrs["alphabet"] = "".join(alphabet)
            data.attrs["moltype"] = alphabet.moltype.name

        if offset := offset or self.offset:
            _set_offset(data, offset=offset)
        _set_reversed_seqs(data, reversed_seqs)

        return data, alphabet, offset, reversed_seqs

    def copy(
        self,
        data: h5py.File | None = None,
        alphabet: c3_alphabet.CharAlphabet | None = None,
        offset: dict[str, int] | None = None,
        reversed_seqs: set[str] | None = None,
    ) -> typing_extensions.Self:
        data, alphabet, offset, reversed_seqs = self._make_new_h5_file(
            data=data,
            alphabet=alphabet,
            offset=offset,
            reversed_seqs=reversed_seqs,
        )
        return self.__class__(
            data=data,
            alphabet=alphabet,
            offset=offset,
            reversed_seqs=reversed_seqs,
            check=False,
        )

    def to_alphabet(
        self,
        alphabet: c3_alphabet.AlphabetABC,
        check_valid: bool = True,
    ) -> typing_extensions.Self:
        if (
            len(self.alphabet) == len(alphabet)
            and len(
                {
                    (a, b)
                    for a, b in zip(self.alphabet, alphabet, strict=False)
                    if a != b
                },
            )
            == 1
        ):
            # rna <-> dna swap just replace alphabet
            return self.copy(alphabet=alphabet)

        new_data = {}
        for seqid in self.names:
            seq_data = self.get_seq_array(seqid=seqid)
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

    def add_seqs(
        self,
        seqs: dict[str, StrORBytesORArray],
        force_unique_keys: bool = True,
        offset: dict[str, int] | None = None,
        reversed_seqs: frozenset[str] | None = None,
    ) -> typing_extensions.Self:
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

        group_ref = self._file.get(self._primary_grp, {})
        exists = any(seqid in group_ref for seqid in seqs)
        if force_unique_keys and exists:
            msg = f"{exists} already exist in collection"
            raise ValueError(msg)

        if exists:
            present = {seqid for seqid in seqs if seqid in group_ref}

        for seqid, seq in seqs.items():
            if exists and seqid in present:
                continue

            self._file.create_dataset(
                name=f"{self._primary_grp}/{seqid}",
                data=self.alphabet.to_indices(seq),
                chunks=True,
                **HDF5_BLOSC2_KWARGS,
            )
        if offset := offset or {}:
            _set_offset(self._file, offset=self.offset | offset)
        reversed_seqs = reversed_seqs or frozenset()
        _set_reversed_seqs(self._file, reversed_seqs)

        return self

    def get_seq_array(
        self,
        *,
        seqid: str,
        start: OptInt = None,
        stop: OptInt = None,
        step: OptInt = None,
    ) -> numpy.ndarray:
        """Returns the sequence as a numpy array of indices"""
        if seqid not in self.names:
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

        out[:] = self._file[f"{self._ungapped_grp}/{seqid}"][start:stop:step]
        return out

    def get_seq_bytes(
        self,
        *,
        seqid: str,
        start: OptInt = None,
        stop: OptInt = None,
        step: OptInt = None,
    ) -> bytes:
        return self.get_seq_str(seqid=seqid, start=start, stop=stop, step=step).encode(
            "utf8"
        )

    def get_seq_str(
        self,
        *,
        seqid: str,
        start: OptInt = None,
        stop: OptInt = None,
        step: OptInt = None,
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
        if seqid not in self.names:
            msg = f"Sequence {seqid!r} not found"
            raise KeyError(msg)
        return len(self._file[f"{self._ungapped_grp}/{seqid}"])

    @classmethod
    def from_seqs(
        cls,
        *,
        data,
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> typing_extensions.Self:
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
    ) -> typing_extensions.Self:
        """convert a cogent3 SeqsDataABC into UnalignedSeqsData"""
        if type(seqcoll) is not c3_alignment.SequenceCollection:
            msg = f"Expected seqcoll to be an instance of new_type SequenceCollection, got {type(seqcoll).__name__!r}"
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
    ) -> typing_extensions.Self:
        h5file = open_h5_file(path=path, mode=mode, in_memory=False)
        alphabet = c3_alphabet.make_alphabet(
            chars=h5file.attrs.get("alphabet"),
            gap=h5file.attrs.get("gap_char"),
            missing=h5file.attrs.get("missing_char"),
            moltype=c3_moltype.get_moltype(h5file.attrs.get("moltype")),
        )
        return cls(data=h5file, alphabet=alphabet, check=check)

    def _write(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path).expanduser().absolute()
        curr_path = pathlib.Path(self._file.filename).absolute()
        if path == curr_path:
            # nothing to do
            return
        output = duplicate_h5_file(h5file=self._file, path=path, in_memory=False)
        output.close()

    def write(self, path: str | pathlib.Path) -> None:
        """Write the UnalignedSeqsData object to a file"""
        path = pathlib.Path(path).expanduser().absolute()
        if path.suffix != f".{UNALIGNED_SUFFIX}":
            msg = f"path {path} does not have the expected suffix '.{UNALIGNED_SUFFIX}'"
            raise ValueError(msg)
        self._write(path=path)

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

    def __init__(
        self,
        *,
        gapped_seqs: h5py.File,
        alphabet: c3_alphabet.AlphabetABC,
        offset: dict[str, int] | None = None,
        check: bool = True,
        reversed_seqs: frozenset[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data=gapped_seqs,
            alphabet=alphabet,
            offset=offset,
            check=check,
            reversed_seqs=reversed_seqs,
        )
        self._primary_grp = self._gapped_grp

    @classmethod
    def _check_file(cls, file: h5py.File) -> None:
        if not _valid_h5seqs(file, cls._gapped_grp):
            msg = f"File {file} is not a valid {cls.__name__} file"
            raise ValueError(msg)

    @property
    def names(self) -> tuple[str, ...]:
        grp = self._gapped_grp
        return tuple(self._file[grp]) if grp in self._file else ()

    @property
    def align_len(self) -> int:
        """length of the alignment"""
        name = next(iter(self._file.get(self._gapped_grp, [])), None)
        return len(self._file[f"{self._gapped_grp}/{name}"]) if name else 0

    def __len__(self) -> int:
        return self.align_len

    def get_seq_length(self, seqid: str) -> int:
        """Returns the length of the sequence"""
        if seqid not in self.names:
            msg = f"Sequence {seqid!r} not found"
            raise KeyError(msg)
        if seqid not in self._ungapped_grp:
            self._make_gaps_and_ungapped(seqid)
        return len(self._file[f"{self._ungapped_grp}/{seqid}"])

    @classmethod
    def from_seqs(
        cls,
        *,
        data: dict[str, StrORBytesORArray],
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> typing_extensions.Self:
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
        return make_aligned(path, data=data, alphabet=alphabet, **kwargs)

    @classmethod
    def from_names_and_array(
        cls,
        *,
        names: list[str],
        data: numpy.ndarray,
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> typing_extensions.Self:
        if len(names) != data.shape[0] or not len(names):
            msg = "Number of names must match number of rows in data."
            raise ValueError(msg)

        data = {name: data[i] for i, name in enumerate(names)}
        path = kwargs.pop("storage_path", None)
        mode = kwargs.pop("mode", "w")
        return make_aligned(path, data=data, alphabet=alphabet, mode=mode, **kwargs)

    @classmethod
    def from_seqs_and_gaps(
        cls,
        *,
        seqs: dict[str, StrORBytesORArray],
        gaps: dict[str, numpy.ndarray],
        alphabet: c3_alphabet.AlphabetABC,
        **kwargs,
    ) -> typing_extensions.Self:
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
        return make_aligned(path, data=data, alphabet=alphabet, mode=mode, **kwargs)

    @classmethod
    def from_storage(
        cls,
        seqcoll: c3_alignment.Alignment,
        path: str | pathlib.Path | None = None,
        **kwargs,
    ) -> typing_extensions.Self:
        """convert a cogent3 AlignedSeqsDataABC into AlignedSeqsData"""
        if type(seqcoll) is not c3_alignment.Alignment:
            msg = f"Expected seqcoll to be an instance of new_type Alignment, got {type(seqcoll).__name__!r}"
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
    ) -> typing_extensions.Self:
        h5file = open_h5_file(path=path, mode=mode, in_memory=False)
        alphabet = c3_alphabet.make_alphabet(
            chars=h5file.attrs.get("alphabet"),
            gap=h5file.attrs.get("gap_char"),
            missing=h5file.attrs.get("missing_char"),
            moltype=c3_moltype.get_moltype(h5file.attrs.get("moltype")),
        )
        return cls(gapped_seqs=h5file, alphabet=alphabet, check=check)

    def _make_gaps_and_ungapped(self, seqid: str) -> None:
        if seqid in self._file.get(self._gaps_grp, {}) and seqid in self._file.get(
            self._ungapped_grp, {}
        ):
            # job already done
            return

        ungapped, gaps = c3_alignment.decompose_gapped_seq(
            self.get_gapped_seq_array(seqid=seqid),
            alphabet=self.alphabet,
        )
        self._file.create_dataset(
            name=f"{self._gaps_grp}/{seqid}",
            data=gaps,
            chunks=True,
            **HDF5_BLOSC2_KWARGS,
        )
        self._file.create_dataset(
            name=f"{self._ungapped_grp}/{seqid}",
            data=ungapped,
            chunks=True,
            **HDF5_BLOSC2_KWARGS,
        )

    def _get_gaps(self, seqid: str) -> numpy.ndarray:
        if seqid not in self._file.get(self._gaps_grp, {}):
            self._make_gaps_and_ungapped(seqid)
        return self._file[f"{self._gaps_grp}/{seqid}"][:]

    def get_gaps(self, seqid: str) -> numpy.ndarray:
        return self._get_gaps(seqid)

    def get_gapped_seq_array(
        self, seqid: str, start: OptInt = None, stop: OptInt = None, step: OptInt = None
    ) -> numpy.ndarray:
        start = start or 0
        stop = stop if stop is not None else self.align_len
        step = step or 1
        if start < 0 or stop < 0 or step < 1:
            msg = f"{start=}, {stop=}, {step=} not >= 1"
            raise ValueError(msg)
        return self._file[f"{self._gapped_grp}/{seqid}"][start:stop:step]

    def get_gapped_seq_str(
        self, seqid: str, start: OptInt = None, stop: OptInt = None, step: OptInt = None
    ) -> str:
        data = self.get_gapped_seq_array(seqid=seqid, start=start, stop=stop, step=step)
        return self.alphabet.from_indices(data)

    def get_gapped_seq_bytes(
        self, seqid: str, start: OptInt = None, stop: OptInt = None, step: OptInt = None
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

    def get_positions(
        self,
        names: list[str],
        start: OptInt = None,
        stop: OptInt = None,
        step: OptInt = None,
    ) -> numpy.ndarray:
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
            if name not in self.names:
                msg = f"Sequence {name!r} not found"
                raise KeyError(msg)
            array_seqs[index] = self.get_gapped_seq_array(
                seqid=name,
                start=start,
                stop=stop,
                step=step,
            )
        return array_seqs

    def get_ungapped(
        self,
        name_map: dict[str, str],
        start: OptInt = None,
        stop: OptInt = None,
        step: OptInt = None,
    ) -> tuple[dict, dict]:
        if (start or 0) < 0 or (stop or 0) < 0 or (step or 1) <= 0:
            msg = f"{start=}, {stop=}, {step=} not >= 0"
            raise ValueError(msg)

        seq_array = numpy.empty(
            (len(name_map), self.align_len),
            dtype=self.alphabet.dtype,
        )
        names = tuple(name_map.values())
        for i, name in enumerate(names):
            seq_array[i] = self._file[f"{self._gapped_grp}/{name}"][:]
        seq_array = seq_array[:, start:stop:step]
        # now exclude gaps and missing
        seqs = {}
        for i, name in enumerate(names):
            seq = seq_array[i]
            indices = seq != self.alphabet.gap_index
            if self.alphabet.missing_index is not None:
                indices &= seq != self.alphabet.missing_index
            seqs[name] = seq[indices]

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
    ) -> typing_extensions.Self:
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
        return

    def copy(
        self,
        data: h5py.File | None = None,
        alphabet: c3_alphabet.CharAlphabet | None = None,
        offset: dict[str, int] | None = None,
        reversed_seqs: set[str] | None = None,
    ) -> typing_extensions.Self:
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
        alphabet: c3_alphabet.AlphabetABC,
        check_valid: bool = True,
    ) -> typing_extensions.Self:
        """Returns a new AlignedSeqsData object with the same underlying data
        with a new alphabet."""
        if (
            len(alphabet) == len(self.alphabet)
            and len(
                {
                    (a, b)
                    for a, b in zip(self.alphabet, alphabet, strict=False)
                    if a != b
                },
            )
            == 1
        ):
            # special case where mapping between dna and rna
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
        """Write the UnalignedSeqsData object to a file"""
        if path.suffix != f".{ALIGNED_SUFFIX}":
            msg = f"path {path} does not have the expected suffix '.{ALIGNED_SUFFIX}'"
            raise ValueError(msg)
        self._write(path=path)


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
) -> UnalignedSeqsData:
    h5file = open_h5_file(path=path, mode=mode, in_memory=in_memory)
    if (mode != "r" or in_memory) and alphabet is None:
        msg = "alphabet must be provided for write mode"
        raise ValueError(msg)

    mode = "w" if in_memory else mode
    useqs = UnalignedSeqsData(
        data=h5file,
        alphabet=alphabet,
        offset=offset,
        reversed_seqs=reversed_seqs,
        check=check or mode == "r",
    )
    if data is not None:
        data = useqs.add_seqs(seqs=data, offset=offset, reversed_seqs=reversed_seqs)
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
) -> AlignedSeqsData:
    h5file = open_h5_file(path=path, mode=mode, in_memory=in_memory)
    if (mode != "r" or in_memory) and alphabet is None:
        msg = "alphabet must be provided for write mode"
        raise ValueError(msg)

    asd = AlignedSeqsData(
        gapped_seqs=h5file,
        check=check or h5file.mode == "r",
        alphabet=alphabet,
        offset=offset,
        reversed_seqs=reversed_seqs,
    )
    if data is not None:
        data = asd.add_seqs(seqs=data, offset=offset, reversed_seqs=reversed_seqs)
    return asd


def load_seqs_data_unaligned(
    path: str | pathlib.Path, mode: str = "r", check: bool = True
) -> UnalignedSeqsData:
    """load hdf5 unaligned sequence data from file"""
    path = pathlib.Path(path)
    if path.suffix != f".{UNALIGNED_SUFFIX}":
        msg = f"File {path} does not have an expected suffix {UNALIGNED_SUFFIX!r}"
        raise ValueError(msg)
    klass = UnalignedSeqsData
    return klass.from_file(path=path, mode=mode, check=check)


def load_seqs_data_aligned(
    path: str | pathlib.Path, mode: str = "r", check: bool = True
) -> AlignedSeqsData:
    """load hdf5 aligned sequence data from file"""
    path = pathlib.Path(path)
    if path.suffix != f".{ALIGNED_SUFFIX}":
        msg = f"File {path} does not have an expected suffix {ALIGNED_SUFFIX!r}"
        raise ValueError(msg)
    klass = AlignedSeqsData

    return klass.from_file(path=path, mode=mode, check=check)


def write_seqs_data(
    *,
    path: pathlib.Path,
    seqcoll: SeqsTypes,
    **kwargs,
) -> pathlib.Path:
    if hasattr(seqcoll, "storage") and isinstance(
        seqcoll.storage, (UnalignedSeqsData, AlignedSeqsData)
    ):
        return seqcoll.storage.write(path=path)

    path = pathlib.Path(path)
    supported_suffixes = {
        ALIGNED_SUFFIX: c3_alignment.Alignment,
        UNALIGNED_SUFFIX: c3_alignment.SequenceCollection,
    }
    suffix = path.suffix[1:]
    if suffix not in supported_suffixes:
        msg = f"path {path} does not have a supported suffix {supported_suffixes}"
        raise ValueError(msg)

    if type(seqcoll) is not supported_suffixes[suffix]:
        msg = f"{suffix=} invalid for {type(seqcoll).__name__!r}"
        raise TypeError(msg)

    cls = UnalignedSeqsData if suffix == UNALIGNED_SUFFIX else AlignedSeqsData
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
    store.write(path=path)
    return path


class H5SeqsUnalignedParser(SequenceParserBase):
    @property
    def name(self) -> tuple[str, ...]:
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
    def name(self) -> tuple[str, ...]:
        return "c3h5a"

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
    ) -> typing.Callable[[pathlib.Path], UnalignedSeqsData | AlignedSeqsData]:
        return load_seqs_data_aligned


class H5UnalignedSeqsWriter(SequenceWriterBase):
    @property
    def name(self) -> tuple[str, ...]:
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

    def write(
        self,
        *,
        path: pathlib.Path,
        seqcoll: SeqsTypes,
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
    def name(self) -> tuple[str, ...]:
        return "c3h5a"

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
