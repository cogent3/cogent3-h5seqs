"""cogent3-h5seqs: HDF5 storage backend for cogent3 sequences."""

import pathlib
import typing

import numpy
from cogent3.core import alignment as c3_alignment
from cogent3.format.sequence import SequenceWriterBase
from cogent3.parse.sequence import SequenceParserBase

__version__ = "0.8.0"

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
    load_seqs_data_unaligned,
    make_unaligned,  # noqa: F401
)
from .util import (
    ALIGNED_SUFFIX,
    DEFAULT_COMPRESSION,  # noqa: F401
    SPARSE_SUFFIX,
    UNALIGNED_SUFFIX,
    SeqCollTypes,
    duplicate_h5_file,  # noqa: F401
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
        # storage.flush() commits any deferred Python-level metadata writes
        # into the HDF5 datasets, storage.h5file.flush() then commits h5py's
        # own buffered writes so the in-memory file image reflects them.
        # Both calls are required and target different layers.
        storage.flush()
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
