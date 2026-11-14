"""cogent3-h5seqs: HDF5 storage backend for cogent3 sequences."""

from __future__ import annotations

import pathlib
import typing

import numpy
from cogent3.core import alignment as c3_alignment
from cogent3.format.sequence import SequenceWriterBase
from cogent3.parse.sequence import SequenceParserBase

__version__ = "0.10.0"

# Re-exports
# Re-export from aligned
from .aligned import (
    AlignedSeqsData,
    SparseSeqsData,
    load_seqs_data_aligned,
    load_seqs_data_sparse,
    make_aligned,  # noqa: F401
    make_sparse,  # noqa: F401
)

# Re-export from unaligned
from .unaligned import (
    UnalignedSeqsData,
    _raise_if_open_store_reads,
    load_seqs_data_unaligned,
    make_unaligned,  # noqa: F401
)
from .util import (
    _METADATA,
    ALIGNED_SUFFIX,
    DEFAULT_COMPRESSION,  # noqa: F401
    SPARSE_SUFFIX,
    UNALIGNED_SUFFIX,
    SeqCollTypes,
    _offset_and_reversed_by_name,
    _replace_on_success,
    _set_metadata,
    duplicate_h5_file,
    open_h5_file,  # noqa: F401
)


def write_seqs_data(
    *,
    path: pathlib.Path,
    seqcoll: SeqCollTypes,
    unaligned_suffix: str = UNALIGNED_SUFFIX,
    aligned_suffix: str = ALIGNED_SUFFIX,
    sparse_suffix: str = SPARSE_SUFFIX,
    **kwargs,
) -> pathlib.Path:
    path = pathlib.Path(path).expanduser()
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

    _raise_if_open_store_reads(path)
    metadata = seqcoll.metadata_to_save()

    # check that the collection is modified relative to the underlying storage
    # this will be names of collection and storage are not equal
    # slice_record of Alignment is not generic

    # a store is copied only into a file of its own layout, and
    # SparseSeqsData subclasses AlignedSeqsData, so the type must match exactly
    copyable_store = {aligned_suffix: AlignedSeqsData, sparse_suffix: SparseSeqsData}
    if (
        type(seqcoll.storage) is copyable_store.get(suffix)
        and not seqcoll.modified
        # we want aligned data to remain compact
        and "gaps" not in seqcoll.storage.h5file
    ):
        storage = seqcoll.storage
        # commits deferred offset and reversed-seqs writes before copying
        storage.flush()
        with _replace_on_success(path) as tmp_path:
            # a deleted HDF5 dataset keeps its bytes in the file, so the old
            # metadata is left out of the copy rather than deleted from it
            output = duplicate_h5_file(
                h5file=storage.h5file,
                path=tmp_path,
                in_memory=False,
                exclude_groups={_METADATA},
            )
            try:
                _set_metadata(output, metadata)
            finally:
                output.close()
        return path

    # the following results in storing the primary data only, gapped sequences
    # in the case of an alignment
    cls = UnalignedSeqsData if suffix == unaligned_suffix else AlignedSeqsData
    alphabet = seqcoll.storage.alphabet
    data = {s.name: numpy.array(s) for s in seqcoll.seqs}
    offset, reversed_seqs = _offset_and_reversed_by_name(seqcoll)
    kwargs = {
        "data": data,
        "alphabet": alphabet,
        "offset": offset,
        "reversed_seqs": reversed_seqs,
    } | kwargs
    if suffix != unaligned_suffix:
        kwargs["sparse"] = suffix == sparse_suffix
    store = cls.from_seqs(**kwargs)
    store.filename_suffix = suffix
    _set_metadata(store.h5file, metadata)
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
    def loader(  # type: ignore[override]
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
    def loader(  # type: ignore[override]
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
    def loader(  # type: ignore[override]
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

    @property
    def saves_metadata(self) -> bool:
        return True

    def write(  # type: ignore[override]
        self,
        *,
        path: pathlib.Path,
        seqcoll: SeqCollTypes,
        **kwargs: typing.Any,  # noqa: ANN401
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
