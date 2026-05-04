"""Aligned sequence data storage."""

import collections
import pathlib
import typing

import h5py
import numba
import numpy
import numpy.typing as npt
from cogent3.core import alignment as c3_alignment
from cogent3.core import alphabet as c3_alphabet
from cogent3.core import moltype as c3_moltype
from h5py._hl.dataset import Dataset

from .unaligned import UnalignedSeqsData
from .util import (
    ALIGNED_SUFFIX,
    SPARSE_SUFFIX,
    AlignedDataView,
    NumpyIntArrayType,
    PySeqStrType,
    SeqIntArrayType,
    SliceRecord,
    StrORBytesORArray,
    _assign_attr_if_missing,
    _best_uint_dtype,
    _restore_alphabet,
    _valid_h5seqs,
    array_hash64,
    compose_gapped_seq,
    decompose_gapped_seq,
    open_h5_file,
)


class AlignedSeqsData(UnalignedSeqsData, c3_alignment.AlignedSeqsDataABC):
    """Dense HDF5 storage for aligned sequences.

    Stores gapped and ungapped representations of equal-length sequences.
    Gapped sequences are deduplicated by their xxhash64 digest.

    Parameters
    ----------
    gapped_seqs
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
    """

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
        self._align_len: int | None = None

    @classmethod
    def _check_file(cls, file: h5py.File) -> None:
        if not _valid_h5seqs(file, cls._gapped_grp):
            msg = f"File {file} is not a valid {cls.__name__} file"
            raise ValueError(msg)

    @property
    def align_len(self) -> int:
        """length of the alignment"""
        if self._align_len is not None:
            return self._align_len
        if not self._index_loaded:
            self._ensure_index()
        if not self._name_to_hash:
            # genuinely empty — do NOT cache, first add_seqs will set it
            return 0
        seqhash = next(iter(self._name_to_hash.values()))
        self._align_len = typing.cast(
            "numpy.ndarray",
            self._file[f"{self._gapped_grp}/{seqhash}"],
        ).shape[0]
        return self._align_len

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
            gapped = compose_gapped_seq(
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

        ungapped, gaps = decompose_gapped_seq(
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
        slice_record: SliceRecord | None = None,
    ) -> AlignedDataView:
        return AlignedDataView(
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

        # align_len returns 0 for an empty alignment (the "no constraint yet"
        # state), so compare against 0 explicitly rather than using truthiness
        existing_len = self.align_len
        if len(lengths) > 1 or (existing_len > 0 and existing_len not in lengths):
            msg = f"not all lengths equal {lengths=}"
            raise ValueError(msg)

        super().add_seqs(
            seqs=seqs,
            force_unique_keys=force_unique_keys,
            offset=offset,
            reversed_seqs=reversed_seqs,
        )
        # opportunistic monotonic set: lengths is non-empty and uniform here
        if self._align_len is None and lengths:
            self._align_len = next(iter(lengths))
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
        # _align_len is inherited from AlignedSeqsData with sentinel None
        # sparse aggregate caches, populated lazily by _ensure_sparse_arrays
        self._diff_indices_cache: NumpyIntArrayType | None = None
        self._diff_vals_cache: SeqIntArrayType | None = None
        self._seq_ptrs_cache: NumpyIntArrayType | None = None
        self._var_pos_cache: NumpyIntArrayType | None = None
        self._sparse_loaded: bool = False

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
        # ref-seq registration counts as a mutation that must persist
        self._dirty = True
        return self._ref_hash

    def _ensure_sparse_arrays(self) -> None:
        """Lazily load four sparse aggregate datasets into in-memory."""
        if self._sparse_loaded:
            return
        if self._diff_idx_grp in self._file:
            self._diff_indices_cache = numpy.asarray(self._file[self._diff_idx_grp][:])
        else:
            self._diff_indices_cache = numpy.array([], dtype=numpy.uint8)
        if self._diff_val_grp in self._file:
            self._diff_vals_cache = numpy.asarray(self._file[self._diff_val_grp][:])
        else:
            self._diff_vals_cache = numpy.array([], dtype=numpy.uint8)
        if self._seq_ptr_grp in self._file:
            self._seq_ptrs_cache = numpy.asarray(self._file[self._seq_ptr_grp][:])
        else:
            self._seq_ptrs_cache = numpy.array([0], dtype=numpy.int64)
        if self._var_pos_grp in self._file:
            self._var_pos_cache = numpy.asarray(self._file[self._var_pos_grp][:])
        else:
            self._var_pos_cache = numpy.array([], dtype=numpy.int64)
        self._sparse_loaded = True

    def _flush_extras(self) -> None:
        """Persist the sparse aggregate caches."""
        if self._diff_indices_cache is None or self._diff_indices_cache.size == 0:
            # nothing accumulated, could be a fresh instance with only a ref
            # seq registered, or an existing file we never mutated
            return
        _replace_grp_in_file(
            h5file=self._file,
            compression=self._compress,
            grp_name=self._seq_ptr_grp,
            new_value=self._seq_ptrs_cache,
        )
        _replace_grp_in_file(
            h5file=self._file,
            compression=self._compress,
            grp_name=self._diff_idx_grp,
            new_value=self._diff_indices_cache,
        )
        _replace_grp_in_file(
            h5file=self._file,
            compression=self._compress,
            grp_name=self._diff_val_grp,
            new_value=self._diff_vals_cache,
        )
        _replace_grp_in_file(
            h5file=self._file,
            compression=self._compress,
            grp_name=self._var_pos_grp,
            new_value=self._var_pos_cache,
        )

    # align_len inherited from AlignedSeqsData — uses self._gapped_grp + any
    # hash from _name_to_hash, the ref seq is in _name_to_hash so this works

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
            # _set_ref_seq seeds _name_to_hash with the ref entry and marks dirty
            self._set_ref_seq(ref_name, ref_seq)
            self._index_loaded = True

        if not self._index_loaded:
            self._ensure_index()
        name_to_hash = dict(self._name_to_hash)
        hash_to_index = dict(self._hash_to_index)

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
        existing_len = self.align_len
        if len(lengths) > 1 or (existing_len > 0 and existing_len not in lengths):
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
        # opportunistic monotonic align_len set
        if self._align_len is None and lengths:
            self._align_len = next(iter(lengths))

        if not diff_indices:
            # no unique sequences — only metadata changes
            self._populate_optional_grps(
                offset, reversed_seqs, name_to_hash, hash_to_index
            )
            return self

        if not self._sparse_loaded:
            self._ensure_sparse_arrays()

        # all aggregate updates land in memory, flush() persists them
        self._seq_ptrs_cache = _make_pointers(
            old_pointers=self._seq_ptrs_cache,
            new_indices=diff_indices,
        )

        new_indices_concat = numpy.concatenate(diff_indices)
        if self._diff_indices_cache is None or self._diff_indices_cache.size == 0:
            self._diff_indices_cache = new_indices_concat.astype(
                _best_uint_dtype(int(max_index))
            )
        else:
            target_dtype = _best_uint_dtype(
                max(int(max_index), int(self._diff_indices_cache.max(initial=0)))
            )
            self._diff_indices_cache = numpy.concatenate(
                [
                    self._diff_indices_cache,
                    new_indices_concat,
                ]
            ).astype(target_dtype)

        new_vals_concat = numpy.concatenate(diff_vals).astype(numpy.uint8)
        if self._diff_vals_cache is None or self._diff_vals_cache.size == 0:
            self._diff_vals_cache = new_vals_concat
        else:
            self._diff_vals_cache = numpy.concatenate(
                [
                    self._diff_vals_cache,
                    new_vals_concat,
                ]
            )

        self._var_pos_cache = numpy.unique(self._diff_indices_cache)

        self._populate_optional_grps(offset, reversed_seqs, name_to_hash, hash_to_index)
        return self

    def _get_gapped_seq_array(
        self,
        seqid: str,
        start: int,
        stop: int,
        step: int,
    ) -> SeqIntArrayType:
        if not self._index_loaded:
            self._ensure_index()
        seqhash = self._name_to_hash[seqid]
        if seqhash == self._ref_hash:
            return typing.cast("SeqIntArrayType", self._ref_seq)[start:stop:step]

        if not self._sparse_loaded:
            self._ensure_sparse_arrays()
        index = self._hash_to_index[seqhash]
        seqarray = _inflate_seq(
            index=index,
            ref_seq=typing.cast("SeqIntArrayType", self._ref_seq[:]),
            all_indices=self._diff_indices_cache,
            all_values=self._diff_vals_cache,
            row_ptrs=self._seq_ptrs_cache,
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
        if not self._sparse_loaded:
            self._ensure_sparse_arrays()
        if self._diff_indices_cache is None or self._diff_indices_cache.size == 0:
            return array_seqs[:, ::step]

        all_indices = self._diff_indices_cache
        all_values = self._diff_vals_cache
        seq_ptrs = self._seq_ptrs_cache
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
        if not self._sparse_loaded:
            self._ensure_sparse_arrays()
        subalign = extract_subalignment(
            ref_seq=typing.cast("npt.NDArray", self._ref_seq[:]),
            all_indices=typing.cast("npt.NDArray", self._diff_indices_cache),
            all_vals=typing.cast("npt.NDArray", self._diff_vals_cache),
            seq_ptrs=typing.cast("npt.NDArray", self._seq_ptrs_cache),
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
        if not self._index_loaded:
            self._ensure_index()
        if len(names) < 2 or not self._hash_to_index:
            # no seqs, too few, or all identical
            return numpy.array([], dtype=numpy.int64)

        if not self._sparse_loaded:
            self._ensure_sparse_arrays()
        var_pos = self._var_pos_cache
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


def make_aligned(
    path: str,
    *,
    data: dict[str, numpy.ndarray] | None = None,
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
    """Create or load an AlignedSeqsData or SparseSeqsData instance.

    Parameters
    ----------
    path
        Filesystem path to an HDF5 file.
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
    ref_name
        Name of the reference sequence for sparse storage.
    sparse
        If True, create a SparseSeqsData instance instead of
        AlignedSeqsData.

    Returns
    -------
    AlignedSeqsData or SparseSeqsData
    """
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


def load_seqs_data_aligned(
    path: str | pathlib.Path,
    mode: str = "r",
    check: bool = True,
    suffix: str = ALIGNED_SUFFIX,
) -> AlignedSeqsData:
    """Load dense aligned sequence data from an HDF5 file.

    Parameters
    ----------
    path
        Path to an ``.c3h5a`` file.
    mode
        HDF5 file open mode.
    check
        If True, validate the HDF5 file structure on load.
    suffix
        Expected filename suffix.

    Returns
    -------
    AlignedSeqsData

    Raises
    ------
    ValueError
        If the file suffix does not match ``suffix``.
    """
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
    """Load sparse aligned sequence data from an HDF5 file.

    Parameters
    ----------
    path
        Path to a ``.c3h5s`` file.
    mode
        HDF5 file open mode.
    check
        If True, validate the HDF5 file structure on load.
    suffix
        Expected filename suffix.

    Returns
    -------
    SparseSeqsData

    Raises
    ------
    ValueError
        If the file suffix does not match ``suffix``.
    """
    path = pathlib.Path(path)
    if path.suffix != f".{suffix}":
        msg = f"File {path} does not have an expected suffix {suffix!r}"
        raise ValueError(msg)
    klass = SparseSeqsData

    result = klass.from_file(path=path, mode=mode, check=check)
    result.filename_suffix = suffix
    return result
