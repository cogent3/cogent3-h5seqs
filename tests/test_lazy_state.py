"""Tests for deferred-write / lazy-state-sync behaviour."""

import cogent3
import pytest

import cogent3_h5seqs
from cogent3_h5seqs import (
    UnalignedSeqsData,
    make_aligned,
    make_sparse,
    make_unaligned,
)
from cogent3_h5seqs import aligned as c3h5_aligned
from cogent3_h5seqs import util as c3h5_util


@pytest.fixture
def dna_alpha():
    return cogent3.get_moltype("dna").most_degen_alphabet()


@pytest.fixture
def raw_data():
    return {"s1": "ACGG", "s2": "TGGGCAGTA", "s3": "ACGGTTAA"}


@pytest.fixture
def raw_aligned_data():
    return {"s1": "TGG--ACGG", "s2": "TGGGCAGTA", "s3": "TGGGCACGG"}


@pytest.fixture
def small_unaligned(raw_data, dna_alpha):
    return make_unaligned("memory", data=raw_data, in_memory=True, alphabet=dna_alpha)


@pytest.fixture
def small_aligned(raw_aligned_data, dna_alpha):
    return make_aligned(
        "memory", data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
    )


@pytest.fixture
def small_sparse(raw_aligned_data, dna_alpha):
    return make_sparse(
        "memory", data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
    )


def test_empty_unaligned_align_attrs(dna_alpha, tmp_path):
    # An empty file's __contains__ / names / align_len return correctly
    # without erroring.
    path = tmp_path / "empty.c3h5u"
    obj = make_unaligned(path, alphabet=dna_alpha, mode="w")
    assert "anything" not in obj
    assert obj.names == ()
    assert len(obj) == 0


def test_empty_aligned_align_len(dna_alpha, tmp_path):
    path = tmp_path / "empty.c3h5a"
    obj = make_aligned(path, alphabet=dna_alpha, mode="w")
    assert obj.align_len == 0
    assert "anything" not in obj
    assert obj.names == ()


def test_load_once_invariant_unaligned(dna_alpha, raw_data, tmp_path, monkeypatch):
    # _get_name2hash_hash2idx is invoked exactly once across a sequence
    # of __contains__ / names / get_seq_array calls on a populated on-disk
    # file. The patch must be installed BEFORE the instance is opened, so
    # we observe the first lazy load.
    # build and close a populated on-disk file (counter not yet active)
    path = tmp_path / "x.c3h5u"
    writer = make_unaligned(path, data=raw_data, alphabet=dna_alpha, mode="w")
    writer.close()

    # install the counter
    calls = {"n": 0}
    real = c3h5_util._get_name2hash_hash2idx

    def counting(file):
        calls["n"] += 1
        return real(file)

    monkeypatch.setattr("cogent3_h5seqs.unaligned._get_name2hash_hash2idx", counting)

    # open the file: __init__ must not eagerly decode the index
    obj = cogent3_h5seqs.load_seqs_data_unaligned(path)
    assert calls["n"] == 0

    # first read triggers the lazy load
    assert "s1" in obj
    assert calls["n"] == 1

    # subsequent reads must not reload
    _ = obj.names
    _ = obj.names
    _ = "s2" in obj
    _ = obj.get_seq_array(seqid="s1")
    assert calls["n"] == 1


def test_load_once_invariant_after_add(dna_alpha, tmp_path, monkeypatch):
    # add_seqs must not re-decode name_to_hash from disk.
    calls = {"n": 0}
    real = c3h5_util._get_name2hash_hash2idx

    def counting(file):
        calls["n"] += 1
        return real(file)

    monkeypatch.setattr("cogent3_h5seqs.unaligned._get_name2hash_hash2idx", counting)
    obj = make_unaligned(tmp_path / "x.c3h5u", alphabet=dna_alpha, mode="w")
    obj.add_seqs({"a": "ACGT"})
    obj.add_seqs({"b": "TTTT"})
    obj.add_seqs({"c": "GGGG"})
    # one load max — for the first add or first __contains__ access
    assert calls["n"] == 1


def test_align_len_set_by_first_add(small_aligned):
    # align_len is cached after first add_seqs and does not re-probe.
    expected = small_aligned.align_len
    assert expected > 0
    # repeat access should hit the cache
    assert small_aligned.align_len == expected
    assert small_aligned._align_len == expected


def test_deferred_write_invariant_unaligned(dna_alpha, tmp_path, monkeypatch):
    # _set_name_to_hash_to_index is called once per flush, not per add.
    calls = {"n": 0}
    real = c3h5_util._set_name_to_hash_to_index

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("cogent3_h5seqs.unaligned._set_name_to_hash_to_index", counting)
    obj = make_unaligned(tmp_path / "x.c3h5u", alphabet=dna_alpha, mode="w")
    obj.add_seqs({"a": "ACGT"})
    obj.add_seqs({"b": "TTTT"})
    obj.add_seqs({"c": "GGGG"})
    assert calls["n"] == 0  # nothing flushed yet
    obj.flush()
    assert calls["n"] == 1


def test_flush_correctness_aligned(dna_alpha, tmp_path):
    # A fresh handle on the same file sees the flushed metadata.
    path = tmp_path / "x.c3h5a"
    obj = make_aligned(path, alphabet=dna_alpha, mode="w")
    obj.add_seqs({"a": "ACGT", "b": "TTTT"})
    obj.close()  # auto-flush + close

    fresh = cogent3_h5seqs.load_seqs_data_aligned(path)
    assert set(fresh.names) == {"a", "b"}
    assert fresh.align_len == 4


def test_explicit_flush_visible_to_second_handle(dna_alpha, tmp_path):
    # obj.flush() mid-session followed by a second read-only handle on the
    # same path sees the flushed state.
    path = tmp_path / "x.c3h5u"
    writer = make_unaligned(path, alphabet=dna_alpha, mode="w")
    writer.add_seqs({"a": "ACGT", "b": "GGGG"})
    writer.flush()
    # open a second read-only handle while writer is still alive
    reader = cogent3_h5seqs.load_seqs_data_unaligned(path)
    assert set(reader.names) == {"a", "b"}
    reader.close()
    writer.close()


def test_plugin_write_path_a_persists_metadata(dna_alpha, tmp_path):
    # write_seqs_data must capture post-add_seqs name_to_hash
    # regression guard for the deferred-write Path A bug where a stale
    # file image would otherwise be captured.
    storage = make_unaligned(
        "memory",
        data={"a": "ACGT", "b": "GGGG"},
        in_memory=True,
        alphabet=dna_alpha,
    )
    seqcoll = cogent3.make_unaligned_seqs(storage, moltype="dna")
    out_path = tmp_path / "out.c3h5u"
    cogent3_h5seqs.write_seqs_data(path=out_path, seqcoll=seqcoll)
    fresh = cogent3_h5seqs.load_seqs_data_unaligned(out_path)
    assert set(fresh.names) == {"a", "b"}


def test_repr_consistency_with_unflushed_adds(dna_alpha, tmp_path):
    # repr() of a writeable, dirty instance reports the post-add count
    # (would have been stale before the __repr__ fix).
    obj = make_unaligned(tmp_path / "x.c3h5u", alphabet=dna_alpha, mode="w")
    obj.add_seqs({"a": "ACGT", "b": "GGGG", "c": "TTTT"})
    text = repr(obj)
    assert "num_seqs=3" in text


def test_sparse_aggregate_deferral(dna_alpha, tmp_path, monkeypatch):
    # Sparse aggregate datasets (diff_indices etc.) are written exactly
    # once per flush, not once per add_seqs.

    calls = {"n": 0}
    real = c3h5_aligned._replace_grp_in_file

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(c3h5_aligned, "_replace_grp_in_file", counting)
    obj = make_sparse(
        tmp_path / "x.c3h5s",
        alphabet=dna_alpha,
        mode="w",
    )
    obj.add_seqs({"r": "ACGTACGT", "a": "ACATACGT"})
    pre_flush = calls["n"]
    obj.add_seqs({"b": "ACGTACAT"})
    pre_flush_2 = calls["n"]
    assert pre_flush_2 == pre_flush  # second add did not write anything
    obj.flush()
    # after flush: 4 _replace_grp_in_file calls (seq_ptrs, diff_indices,
    # diff_vals, var_pos)
    assert calls["n"] - pre_flush_2 == 4


def test_sparse_align_len_no_eager_load(dna_alpha, tmp_path, monkeypatch):
    # Constructing SparseSeqsData from an existing populated file does
    # NOT trigger name_to_hash decode in __init__
    path = tmp_path / "x.c3h5s"
    writer = make_sparse(
        path,
        alphabet=dna_alpha,
        mode="w",
    )
    writer.add_seqs(
        {"r": "ACGTACGT", "a": "ACATACGT", "b": "ACGTACAT"},
    )
    writer.close()

    calls = {"n": 0}
    real = c3h5_util._get_name2hash_hash2idx

    def counting(file):
        calls["n"] += 1
        return real(file)

    monkeypatch.setattr("cogent3_h5seqs.unaligned._get_name2hash_hash2idx", counting)
    reader = cogent3_h5seqs.load_seqs_data_sparse(path)
    # construction should NOT have decoded name_to_hash
    assert calls["n"] == 0
    # first query that needs it triggers the load
    _ = reader.align_len
    assert calls["n"] == 1


def test_del_partial_init_safety():
    # If __init__ raises after assigning self._file but before
    # self._dirty, __del__ must not raise AttributeError.
    # construct a half-initialised instance manually
    obj = UnalignedSeqsData.__new__(UnalignedSeqsData)
    obj._file = None  # type: ignore[assignment]
    # __del__ must early-return cleanly
    obj.__del__()
    # also test the case where _file is set but _dirty is missing

    h5 = c3h5_util.open_h5_file(path=None, mode="w-", in_memory=True)
    obj2 = UnalignedSeqsData.__new__(UnalignedSeqsData)
    obj2._file = h5
    # do not assign _dirty — simulate __init__ raising before that point
    obj2.__del__()  # must not raise


def test_align_len_sentinel_is_none(dna_alpha, tmp_path):
    # _align_len uses None as the unset sentinel
    obj = make_aligned(tmp_path / "x.c3h5a", alphabet=dna_alpha, mode="w")
    assert obj._align_len is None
    obj.add_seqs({"a": "ACGT", "b": "GGGG"})
    assert obj._align_len == 4


def test_flush_no_op_on_readonly(tmp_path, dna_alpha):
    # flush() short-circuits on read-only file without raising.
    path = tmp_path / "x.c3h5u"
    writer = make_unaligned(path, alphabet=dna_alpha, mode="w")
    writer.add_seqs({"a": "ACGT"})
    writer.close()

    reader = cogent3_h5seqs.load_seqs_data_unaligned(path)
    # flush on read-only must be a no-op, not raise
    reader.flush()
    reader.close()


def test_deferred_offset_persistence(dna_alpha, tmp_path):
    # Offsets passed to add_seqs are deferred but appear after flush.
    path = tmp_path / "x.c3h5u"
    writer = make_unaligned(path, alphabet=dna_alpha, mode="w")
    writer.add_seqs({"a": "ACGT"}, offset={"a": 10})
    assert writer.offset["a"] == 10
    writer.close()

    reader = cogent3_h5seqs.load_seqs_data_unaligned(path)
    assert reader.offset["a"] == 10


def test_deferred_reversed_seqs_persistence(dna_alpha, tmp_path):
    # reversed_seqs passed to add_seqs are deferred but appear after flush.
    path = tmp_path / "x.c3h5u"
    writer = make_unaligned(path, alphabet=dna_alpha, mode="w")
    writer.add_seqs({"a": "ACGT"}, reversed_seqs=frozenset({"a"}))
    assert writer.reversed_seqs == frozenset({"a"})
    writer.close()

    reader = cogent3_h5seqs.load_seqs_data_unaligned(path)
    assert reader.reversed_seqs == frozenset({"a"})


def test_sparse_round_trip_after_flush(dna_alpha, tmp_path, raw_aligned_data):
    # Sparse data round-trips correctly via deferred-write.
    path = tmp_path / "x.c3h5s"
    writer = make_sparse(path, alphabet=dna_alpha, mode="w")
    writer.add_seqs(raw_aligned_data)
    writer.close()

    reader = cogent3_h5seqs.load_seqs_data_sparse(path)
    assert all(
        reader.get_gapped_seq_str(name) == raw_aligned_data[name]
        for name in raw_aligned_data
    )
