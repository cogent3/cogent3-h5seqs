import cogent3_h5seqs
import cogent3
import numpy
import pytest


@pytest.fixture
def raw_data():
    return {"s1": "ACGG", "s2": "TGGGCAGTA"}


@pytest.fixture
def raw_aligned_data():
    return {"s1": "TGG--ACGG", "s2": "TGGGCAGTA"}


@pytest.fixture
def small(raw_data):
    alpha = cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()
    return cogent3_h5seqs.make_unaligned(
        "memory", data=raw_data, in_memory=True, alphabet=alpha
    )


@pytest.mark.parametrize("offset", [None, {"s1": 2}])
def test_make_unaligned(raw_data, offset):
    alpha = cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()
    ua = cogent3_h5seqs.make_unaligned(
        "memory", data=raw_data, in_memory=True, alphabet=alpha, offset=offset
    )
    assert ua.names == ("s1", "s2")
    assert len(ua) == 2
    assert numpy.allclose(
        ua.get_seq_array(seqid="s1"), alpha.to_indices(raw_data["s1"])
    )
    assert ua.get_seq_str(seqid="s1") == raw_data["s1"]
    assert ua.get_seq_str(seqid="s2") == raw_data["s2"]
    assert ua.get_seq_bytes(seqid="s2") == raw_data["s2"].encode("utf-8")
    assert ua.get_seq_length(seqid="s1") == len(raw_data["s1"])
    assert ua.offset == (offset or {})
    assert ua.reversed_seqs == frozenset()


def test_unaligned_get_view(small, raw_data):
    view = small.get_view(seqid="s1")
    assert view.parent is small
    assert view.seqid == "s1"
    assert str(view) == raw_data["s1"]
    nv = view[2:4]
    assert str(nv) == raw_data["s1"][2:4]


@pytest.mark.parametrize("seqid", ["s1", "s2"])
def test_unaligned_index(small, raw_data, seqid):
    sv = small[seqid]
    assert sv.seqid == seqid
    assert str(sv) == raw_data[seqid]
    index = small.names.index(seqid)
    sv = small[index]
    assert sv.seqid == seqid


def test_unaligned_copy(small):
    copy = small.copy()
    copy.add_seqs({"s3": "ACGT"})
    assert copy.names != small.names


def test_unaligned_eq(small):
    copy = small.copy()
    assert copy == small


def test_unaligned_neq(small):
    copy = small.copy()
    copy.add_seqs({"s3": "ACGT"})
    assert copy != small


def test_unaligned_to_rna(small):
    rna = cogent3.get_moltype("rna", new_type=True).most_degen_alphabet()
    mod = small.to_alphabet(rna)
    assert numpy.allclose(numpy.array(mod["s1"]), numpy.array(small["s1"]))
    assert str(mod["s2"]) == str(small["s2"]).replace("T", "U")


def test_unaligned_to_alphabet_text(small):
    text = cogent3.get_moltype("text", new_type=True).most_degen_alphabet()
    mod = small.to_alphabet(text)
    # arrays now different
    assert not numpy.allclose(numpy.array(mod["s1"]), numpy.array(small["s1"]))
    # but str is the same
    assert str(mod["s2"]) == str(small["s2"])
    assert mod.alphabet == text


def test_unaligned_offset(small):
    copy = small.copy(offset={"s1": 2})
    assert copy.offset == {"s1": 2}
    s1 = copy.get_view(seqid="s1")
    assert s1.offset == 2
    s2 = copy.get_view(seqid="s2")
    assert s2.offset == 0


def test_unaligned_reversed_seqs(small):
    copy = small.copy(reversed_seqs={"s2"})
    assert copy.reversed_seqs == {"s2"}
    s2 = copy.get_view(seqid="s2")
    assert s2.is_reversed


def test_write(tmp_path, small):
    path = tmp_path / f"unaligned.{cogent3_h5seqs.UNALIGNED_SUFFIX}"
    small.write(path)
    # assert path.is_file()
    loaded = cogent3_h5seqs.load_seqs_data(path)
    assert loaded == small


def test_write_invalid(tmp_path, small):
    path = tmp_path / "unaligned.h5seqs"
    with pytest.raises(ValueError):
        small.write(path)


def test_load_invalid(tmp_path):
    path = tmp_path / "unaligned.h5seqs"
    with pytest.raises(ValueError):
        cogent3_h5seqs.load_seqs_data(path)


def test_make_alignedseqsdata(raw_aligned_data):
    alpha = cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()
    asd = cogent3_h5seqs.make_aligned(
        path=None, data=raw_aligned_data, in_memory=True, alphabet=alpha
    )
    assert len(asd) == len(raw_aligned_data["s2"])
    assert asd.names == ("s1", "s2")


def test_driver_unaligned(raw_data):
    seqs = cogent3.make_unaligned_seqs(
        data=raw_data, moltype="dna", storage_backend="h5seqs_unaligned", new_type=True
    )
    assert isinstance(seqs.storage, cogent3_h5seqs.UnalignedSeqsData)


def test_driver_aligned(raw_aligned_data):
    seqs = cogent3.make_aligned_seqs(
        data=raw_aligned_data,
        moltype="dna",
        storage_backend="h5seqs_aligned",
        new_type=True,
    )
    assert isinstance(seqs.storage, cogent3_h5seqs.AlignedSeqsData)


@pytest.fixture
def small_unaligned(raw_data):
    alpha = cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()
    return cogent3_h5seqs.make_unaligned(
        None, data=raw_data, in_memory=True, alphabet=alpha
    )


@pytest.fixture
def h5_unaligned_path(small_unaligned, tmp_path):
    outpath = tmp_path / f"aligned_output.{cogent3_h5seqs.UNALIGNED_SUFFIX}"
    small_unaligned.write(outpath)
    return outpath


def test_load_h5_unaligned(h5_unaligned_path, raw_data):
    seqs = cogent3.load_unaligned_seqs(h5_unaligned_path, moltype="dna", new_type=True)
    assert seqs.to_dict() == raw_data


@pytest.fixture
def small_aligned(raw_aligned_data):
    alpha = cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()
    return cogent3_h5seqs.make_aligned(
        path=None, data=raw_aligned_data, in_memory=True, alphabet=alpha
    )


@pytest.fixture
def h5_aligned_path(small_aligned, tmp_path):
    outpath = tmp_path / f"aligned_output.{cogent3_h5seqs.ALIGNED_SUFFIX}"
    small_aligned.write(outpath)
    return outpath


def test_load_h5_aligned(h5_aligned_path, raw_aligned_data):
    aln = cogent3.load_aligned_seqs(h5_aligned_path, moltype="dna", new_type=True)
    assert aln.to_dict() == raw_aligned_data


@pytest.mark.parametrize(
    "cls", [cogent3_h5seqs.UnalignedSeqsData, cogent3_h5seqs.AlignedSeqsData]
)
def test_check_init(cls):
    h5file = cogent3_h5seqs.open_h5_file(path=None, mode="w", in_memory=True)
    alpha = cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()
    kwargs = (
        {"data": h5file}
        if cls == cogent3_h5seqs.UnalignedSeqsData
        else {"gapped_seqs": h5file}
    )
    with pytest.raises(ValueError):
        cls(alphabet=alpha, check=True, **kwargs)
    h5file.close()
