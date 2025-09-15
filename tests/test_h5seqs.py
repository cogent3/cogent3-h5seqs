import itertools
import pathlib

import cogent3
import numpy
import pytest

import cogent3_h5seqs

c3_load_funcs = {
    cogent3_h5seqs.ALIGNED_SUFFIX: cogent3.load_aligned_seqs,
    cogent3_h5seqs.SPARSE_SUFFIX: cogent3.load_aligned_seqs,
    cogent3_h5seqs.UNALIGNED_SUFFIX: cogent3.load_unaligned_seqs,
}

c3_make_funcs = {
    cogent3_h5seqs.ALIGNED_SUFFIX: cogent3.make_aligned_seqs,
    cogent3_h5seqs.SPARSE_SUFFIX: cogent3.make_aligned_seqs,
    cogent3_h5seqs.UNALIGNED_SUFFIX: cogent3.make_unaligned_seqs,
}
c3h5_load_funcs = {
    cogent3_h5seqs.ALIGNED_SUFFIX: cogent3_h5seqs.load_seqs_data_aligned,
    cogent3_h5seqs.SPARSE_SUFFIX: cogent3_h5seqs.load_seqs_data_sparse,
    cogent3_h5seqs.UNALIGNED_SUFFIX: cogent3_h5seqs.load_seqs_data_unaligned,
}


c3h5_make_funcs = {
    cogent3_h5seqs.ALIGNED_SUFFIX: cogent3_h5seqs.make_aligned,
    cogent3_h5seqs.SPARSE_SUFFIX: cogent3_h5seqs.make_sparse,
    cogent3_h5seqs.UNALIGNED_SUFFIX: cogent3_h5seqs.make_unaligned,
}


@pytest.fixture
def dna_alpha():
    return cogent3.get_moltype("dna").most_degen_alphabet()


@pytest.fixture
def raw_data():
    return {"s1": "ACGG", "s2": "TGGGCAGTA"}


@pytest.fixture
def raw_aligned_data():
    return {"s1": "TGG--ACGG", "s2": "TGGGCAGTA"}


@pytest.fixture
def small(raw_data, dna_alpha):
    return cogent3_h5seqs.make_unaligned(
        "memory", data=raw_data, in_memory=True, alphabet=dna_alpha
    )


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_make_from_empty(dna_alpha, tmp_path, suffix):
    mk_obj = c3h5_make_funcs[suffix]
    store_path = tmp_path / f"empty.{suffix}"
    init = mk_obj(store_path, alphabet=dna_alpha, mode="w")
    init.close()  # this forces attributes to be written
    got = mk_obj(store_path, mode="r", check=False)
    assert got.get_attr("moltype") == "dna"
    assert got.alphabet == dna_alpha


@pytest.mark.parametrize("offset", [None, {"s1": 2}])
def test_make_unaligned(raw_data, offset, dna_alpha):
    offset_expect = dict.fromkeys(raw_data, 0) | (offset or {})
    ua = cogent3_h5seqs.make_unaligned(
        "memory", data=raw_data, in_memory=True, alphabet=dna_alpha, offset=offset
    )
    assert ua.names == ("s1", "s2")
    assert len(ua) == 2
    assert numpy.allclose(
        ua.get_seq_array(seqid="s1"), dna_alpha.to_indices(raw_data["s1"])
    )
    assert ua.get_seq_str(seqid="s1") == raw_data["s1"]
    assert ua.get_seq_str(seqid="s2") == raw_data["s2"]
    assert ua.get_seq_bytes(seqid="s2") == raw_data["s2"].encode("utf-8")
    assert ua.get_seq_length(seqid="s1") == len(raw_data["s1"])
    assert ua.offset == offset_expect
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


@pytest.mark.parametrize("fxt", ["small", "small_aligned"])
def test_dna_to_rna(fxt, request):
    small = request.getfixturevalue(fxt)
    # convert to rna
    rna = cogent3.get_moltype("rna").most_degen_alphabet()
    mod = small.to_alphabet(rna)
    assert numpy.allclose(numpy.array(mod["s1"]), numpy.array(small["s1"]))
    assert str(mod["s2"]) == str(small["s2"]).replace("T", "U")


@pytest.mark.parametrize("fxt", ["small", "small_aligned"])
def test_dna_to_text(fxt, request):
    small = request.getfixturevalue(fxt)
    text = cogent3.get_moltype("text").most_degen_alphabet()
    mod = small.to_alphabet(text)
    # arrays now different
    assert not numpy.allclose(numpy.array(mod["s1"]), numpy.array(small["s1"]))
    # but str is the same
    assert str(mod["s2"]) == str(small["s2"])
    assert mod.alphabet == text


def test_unaligned_offset(small):
    copy = small.copy(offset={"s1": 2})
    assert copy.offset == {"s1": 2, "s2": 0}
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
    assert path.is_file()
    loaded = cogent3_h5seqs.load_seqs_data_unaligned(path)
    assert loaded == small


@pytest.mark.parametrize("fxt", ["small", "small_aligned", "small_aligned_sparse"])
def test_write_invalid_suffix(tmp_path, fxt, request):
    small = request.getfixturevalue(fxt)
    # wrong suffix
    path = tmp_path / "seqs.blah"
    with pytest.raises(ValueError):
        small.write(path)


def test_close(small):
    # successive calls should not fail
    small.close()
    small.close()


def test_write_twice(tmp_path, small):
    path = tmp_path / f"unaligned.{cogent3_h5seqs.UNALIGNED_SUFFIX}"
    small.write(path)
    loaded = cogent3_h5seqs.load_seqs_data_unaligned(path, mode="r+")
    assert loaded == small
    # write has no effect
    loaded.write(path)


def test_write_invalid(tmp_path, small):
    path = tmp_path / "unaligned.h5seqs"
    with pytest.raises(ValueError):
        small.write(path)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_load_invalid(tmp_path, suffix):
    func = c3h5_load_funcs[suffix]
    path = tmp_path / "wrong-suffix.h5seqs"
    with pytest.raises(ValueError):
        func(path)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_equality(raw_aligned_data, dna_alpha, suffix):
    mk_obj = c3h5_make_funcs[suffix]
    store1 = mk_obj(
        None, data=raw_aligned_data.copy(), in_memory=True, alphabet=dna_alpha
    )
    store2 = mk_obj(
        None, data=raw_aligned_data.copy(), in_memory=True, alphabet=dna_alpha
    )
    assert store1 == store2


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_inequality(raw_aligned_data, dna_alpha, suffix):
    mk_obj = c3h5_make_funcs[suffix]
    store1 = mk_obj(
        None, data=raw_aligned_data.copy(), in_memory=True, alphabet=dna_alpha
    )
    # wrong type
    assert store1 != "string"
    # seqnames different
    store2 = mk_obj(
        None,
        data={k: v for k, v in raw_aligned_data.items() if k != "s1"},
        in_memory=True,
        alphabet=dna_alpha,
    )
    assert store1 != store2
    # sequence different
    data_edited = raw_aligned_data.copy()
    data_edited["s1"] = data_edited["s1"][:-1] + "N"
    store2 = mk_obj(
        None,
        data=data_edited,
        in_memory=True,
        alphabet=dna_alpha,
    )
    assert store1 != store2
    # attrs different
    store2 = mk_obj(
        None,
        data=raw_aligned_data,
        in_memory=True,
        alphabet=dna_alpha,
    )
    store2.set_attr("test", "1")
    assert store1 != store2
    # attrs different values
    store1.set_attr("test", "2")
    assert store1 != store2
    # fields different
    store2 = mk_obj(
        None,
        data=raw_aligned_data,
        in_memory=True,
        alphabet=dna_alpha,
        offset={"s1": 2},
    )
    store2.set_attr("test", "2")
    assert store1 != store2


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_make_alignedseqsdata(raw_aligned_data, dna_alpha, suffix):
    mk_obj = c3h5_make_funcs[suffix]
    asd = mk_obj(path=None, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha)
    assert len(asd) == len(raw_aligned_data["s2"])
    assert asd.names == ("s1", "s2")


def test_driver_unaligned(raw_data):
    seqs = cogent3.make_unaligned_seqs(
        raw_data, moltype="dna", storage_backend="h5seqs_unaligned"
    )
    assert isinstance(seqs.storage, cogent3_h5seqs.UnalignedSeqsData)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_driver_aligned(raw_aligned_data, suffix):
    seqs = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", storage_backend=suffix
    )
    classes = {
        cogent3_h5seqs.ALIGNED_SUFFIX: cogent3_h5seqs.AlignedSeqsData,
        cogent3_h5seqs.SPARSE_SUFFIX: cogent3_h5seqs.SparseSeqsData,
    }
    assert isinstance(seqs.storage, classes[suffix])


@pytest.fixture
def small_unaligned(raw_data, dna_alpha):
    return cogent3_h5seqs.make_unaligned(
        None, data=raw_data, in_memory=True, alphabet=dna_alpha
    )


@pytest.fixture
def h5_unaligned_path(small_unaligned, tmp_path):
    outpath = tmp_path / f"aligned_output.{cogent3_h5seqs.UNALIGNED_SUFFIX}"
    small_unaligned.write(outpath)
    return outpath


def test_load_h5_unaligned(h5_unaligned_path, raw_data):
    seqs = cogent3.load_unaligned_seqs(h5_unaligned_path, moltype="dna")
    assert seqs.to_dict() == raw_data


@pytest.fixture
def small_aligned(raw_aligned_data, dna_alpha):
    return cogent3_h5seqs.make_aligned(
        path=None, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
    )


@pytest.fixture
def small_aligned_sparse(raw_aligned_data, dna_alpha):
    return cogent3_h5seqs.make_aligned(
        path=None,
        data=raw_aligned_data.copy(),
        in_memory=True,
        alphabet=dna_alpha,
        suffix=cogent3_h5seqs.SPARSE_SUFFIX,
        sparse=True,
    )


@pytest.fixture
def h5_aligned_path(small_aligned, tmp_path):
    outpath = tmp_path / f"aligned_output.{cogent3_h5seqs.ALIGNED_SUFFIX}"
    small_aligned.write(outpath)
    return outpath


@pytest.fixture
def h5_sparse_path(small_aligned_sparse, tmp_path):
    outpath = tmp_path / f"aligned_output.{cogent3_h5seqs.SPARSE_SUFFIX}"
    small_aligned_sparse.write(outpath)
    return outpath


@pytest.mark.parametrize("fxt", ["h5_aligned_path", "h5_sparse_path"])
def test_load_h5_aligned(fxt, request, raw_aligned_data):
    aligned_path = request.getfixturevalue(fxt)
    aln = cogent3.load_aligned_seqs(aligned_path, moltype="dna")
    assert aln.to_dict() == raw_aligned_data


@pytest.mark.parametrize(
    "cls",
    [
        cogent3_h5seqs.UnalignedSeqsData,
        cogent3_h5seqs.AlignedSeqsData,
        cogent3_h5seqs.SparseSeqsData,
    ],
)
def test_check_init(cls, dna_alpha):
    h5file = cogent3_h5seqs.open_h5_file(path=None, mode="w", in_memory=True)
    kwargs = (
        {"data": h5file}
        if cls == cogent3_h5seqs.UnalignedSeqsData
        else {"gapped_seqs": h5file}
    )
    with pytest.raises(ValueError):
        cls(alphabet=dna_alpha, check=True, **kwargs)
    h5file.close()


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_unaligned", "small_aligned_sparse"]
)
def test_getitem_str_int(fxt, request):
    obj = request.getfixturevalue(fxt)
    seqid = "s1"
    index = obj.names.index(seqid)
    str_got = obj[seqid]
    int_got = obj[index]
    assert str(str_got) == str(int_got)


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_unaligned", "small_aligned_sparse"]
)
def test_getitem_err(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(TypeError):
        obj[20.0]


@pytest.fixture(params=c3h5_make_funcs.keys())
def h5seq_file(request, tmp_path, dna_alpha):
    suffix = request.param
    make = c3h5_make_funcs[suffix]
    path = tmp_path / f"test.{suffix}"
    obj = make(path, mode="w", alphabet=dna_alpha)
    obj.close()
    yield path
    path.unlink(missing_ok=True)


def test_add_seqs_not_writeable(h5seq_file):
    load = c3h5_load_funcs[h5seq_file.suffix[1:]]
    obj = load(path=h5seq_file, mode="r", check=False)
    with pytest.raises(PermissionError):
        obj.add_seqs({"seq1": "ATGC"})


def test_make_empty_aligned(dna_alpha):
    h5file = cogent3_h5seqs.open_h5_file("memory", mode="w", in_memory=True)
    asd = cogent3_h5seqs.AlignedSeqsData(
        gapped_seqs=h5file, alphabet=dna_alpha, check=False
    )
    assert asd.align_len == 0
    assert len(asd) == 0


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_aligned_add_seqs_duplicates_disallowed(fxt, request, raw_aligned_data):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(ValueError):
        obj.add_seqs(raw_aligned_data, force_unique_keys=True)


def test_sparse_add_seqs_wrong_ref(small_aligned_sparse):
    other = {"bad": "A" * small_aligned_sparse.align_len}
    with pytest.raises(ValueError):
        small_aligned_sparse.add_seqs(other, ref_name="bad")


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_aligned_add_seqs_duplicates_allowed(fxt, request, raw_aligned_data):
    obj = request.getfixturevalue(fxt)
    num_seqs = len(obj.names)
    obj.add_seqs(raw_aligned_data, force_unique_keys=False)
    assert len(obj.names) == num_seqs


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_unaligned_add_seqs_duplicates_disallowed(fxt, request, raw_data):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(ValueError):
        obj.add_seqs(raw_data, force_unique_keys=True)


def test_unaligned_add_seqs_duplicates_allowed(small_unaligned, raw_data):
    num_seqs = len(small_unaligned.names)
    small_unaligned.add_seqs(raw_data, force_unique_keys=False)
    assert len(small_unaligned.names) == num_seqs


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_unaligned", "small_aligned_sparse"]
)
def test_get_seq_length(fxt, request, raw_data):
    obj = request.getfixturevalue(fxt)
    # seq s2 is the same between the aligned and unaligned examples
    l = obj.get_seq_length(seqid="s2")
    expect = len(raw_data["s2"])
    assert l == expect


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_aligned_sparse", "small_unaligned"]
)
def test_get_seq_length_invalid_seqid(fxt, request):
    obj = request.getfixturevalue(fxt)
    # seq s2 is the same between the aligned and unaligned examples
    with pytest.raises(KeyError):
        obj.get_seq_length(seqid="missing")


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_aligned_sparse", "small_unaligned"]
)
def test_get_seq_array(fxt, request, raw_data, dna_alpha):
    obj = request.getfixturevalue(fxt)
    # seq s2 is the same between the aligned and unaligned examples
    s = obj.get_seq_array(seqid="s2")
    expect = dna_alpha.to_indices(raw_data["s2"])
    assert (s == expect).all()


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_aligned_sparse", "small_unaligned"]
)
@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_seq_array_invalid_pos(fxt, request, arg):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(ValueError):
        obj.get_seq_array(seqid="s2", **{arg: -1})


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_aligned_sparse", "small_unaligned"]
)
def test_get_seq_array_invalid_seqid(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_seq_array(seqid="missing")


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_get_gaps_invalid_seqid(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_gaps(seqid="missing")


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_from_storage(suffix, raw_aligned_data):
    mk_obj = c3_make_funcs[suffix]
    storage_backend = suffix
    coll = mk_obj(
        raw_aligned_data,
        moltype="dna",
        storage_backend=storage_backend,
        in_memory=True,
    )
    got = coll.storage.from_storage(coll, in_memory=True)
    assert got is not coll
    assert got == coll.storage


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_from_storage_invalid(suffix, raw_aligned_data):
    storage_backend = suffix
    mk_obj = c3_make_funcs[suffix]
    coll = mk_obj(
        raw_aligned_data,
        moltype="dna",
        storage_backend=storage_backend,
        in_memory=True,
    )
    with pytest.raises(TypeError):
        coll.storage.from_storage({}, in_memory=False)


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_aligned_from_names_and_array(fxt, request):
    obj = request.getfixturevalue(fxt)
    names = obj.names
    data = numpy.array(
        [obj.get_gapped_seq_array(seqid=name) for name in names],
        dtype=obj.alphabet.dtype,
    )
    got = obj.from_names_and_array(
        names=names, data=data, alphabet=obj.alphabet, sparse="sparse" in fxt
    )
    assert got == obj
    assert got is not obj


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_aligned_from_names_and_array_invalid(fxt, request):
    obj = request.getfixturevalue(fxt)
    names = obj.names
    data = numpy.array(
        [obj.get_gapped_seq_array(seqid=name) for name in names],
        dtype=obj.alphabet.dtype,
    )
    with pytest.raises(ValueError):
        obj.from_names_and_array(names=names[:-1], data=data, alphabet=obj.alphabet)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_aligned_from_names_and_array2(raw_aligned_data, dna_alpha, suffix):
    aln = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", storage_backend=suffix
    )
    classes = {
        cogent3_h5seqs.ALIGNED_SUFFIX: cogent3_h5seqs.AlignedSeqsData,
        cogent3_h5seqs.SPARSE_SUFFIX: cogent3_h5seqs.SparseSeqsData,
    }
    cls = classes[suffix]
    seqs = list(aln.seqs)
    gaps = {s.name: s.map.array for s in seqs}
    s = {s.name: numpy.array(s.seq) for s in seqs}
    got = cls.from_seqs_and_gaps(seqs=s, gaps=gaps, alphabet=dna_alpha)
    assert got == aln.storage


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_aligned_get_ungapped(small_aligned, raw_aligned_data, suffix):
    aln = cogent3.make_aligned_seqs(
        small_aligned, moltype="dna", storage_backend=suffix
    )
    ungapped = aln.degap(storage_backend="c3h5u")
    expect = {n: s.replace("-", "") for n, s in raw_aligned_data.items()}
    assert ungapped.to_dict() == expect
    assert isinstance(ungapped.storage, cogent3_h5seqs.UnalignedSeqsData)


@pytest.mark.parametrize(
    "suffix",
    [
        None,
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
@pytest.mark.parametrize(
    "out_name",
    [
        f"aligned_output.{cogent3_h5seqs.ALIGNED_SUFFIX}",
        f"aligned_output.{cogent3_h5seqs.SPARSE_SUFFIX}",
    ],
)
def test_write_aligned(raw_aligned_data, suffix, tmp_path, out_name):
    aln = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", storage_backend=suffix
    )
    outpath = tmp_path / out_name
    aln.write(outpath)
    assert outpath.exists()
    assert outpath.is_file()


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_get_pos_range(raw_aligned_data, suffix):
    c3 = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", storage_backend=None
    )
    h5 = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", storage_backend=suffix
    )
    assert (c3.array_seqs == h5.array_seqs).all()


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_pos_range_invalid_coord(raw_aligned_data, arg, suffix):
    h5 = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", storage_backend=suffix
    )
    with pytest.raises(ValueError):
        h5.storage.get_pos_range(names=["s1", "s2"], **{arg: -1})


def test_set_as_default_drivers_unaligned(raw_aligned_data):
    cogent3.set_storage_defaults(unaligned_seqs="h5seqs_unaligned")

    coll = cogent3.make_unaligned_seqs(raw_aligned_data, moltype="dna")
    assert isinstance(coll.storage, cogent3_h5seqs.UnalignedSeqsData)

    cogent3.set_storage_defaults(reset=True)

    coll = cogent3.make_unaligned_seqs(raw_aligned_data, moltype="dna")
    assert not isinstance(coll.storage, cogent3_h5seqs.UnalignedSeqsData)


@pytest.mark.parametrize(
    "storage",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        "h5seqs_aligned",
        "h5seqs_sparse",
    ],
)
def test_set_as_default_drivers_aligned(raw_aligned_data, storage):
    classes = {
        cls._suffix: cls
        for cls in [cogent3_h5seqs.AlignedSeqsData, cogent3_h5seqs.SparseSeqsData]
    }
    classes["h5seqs_aligned"] = cogent3_h5seqs.AlignedSeqsData
    classes["h5seqs_sparse"] = cogent3_h5seqs.SparseSeqsData

    cogent3.set_storage_defaults(aligned_seqs=storage)

    coll = cogent3.make_aligned_seqs(raw_aligned_data, moltype="dna")
    assert isinstance(coll.storage, classes[storage])

    cogent3.set_storage_defaults(reset=True)

    coll = cogent3.make_aligned_seqs(raw_aligned_data, moltype="dna")
    assert not isinstance(coll.storage, classes[storage])


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_del(raw_aligned_data, tmp_path, dna_alpha, suffix):
    funcs = {
        cogent3_h5seqs.ALIGNED_SUFFIX: cogent3_h5seqs.make_aligned,
        cogent3_h5seqs.SPARSE_SUFFIX: cogent3_h5seqs.make_aligned,
        cogent3_h5seqs.UNALIGNED_SUFFIX: cogent3_h5seqs.make_unaligned,
    }
    mk_obj = funcs[suffix]
    # passing a filename without a suffix means it will be cleaned
    # up on object deletion
    outpath = tmp_path / "output"
    assert not outpath.exists()
    store = mk_obj(
        outpath, data=raw_aligned_data, in_memory=False, alphabet=dna_alpha, mode="w"
    )
    assert outpath.exists()
    del store
    assert not outpath.exists()


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
@pytest.mark.parametrize("path_type", [pathlib.Path, str])
def test_writing_alignment(tmp_path, path_type, suffix):
    outpath = tmp_path / f"alignment_output.{suffix}"
    aln = cogent3.get_dataset("brca1")
    assert not outpath.exists()
    aln.write(path_type(outpath))
    assert outpath.exists()


@pytest.mark.parametrize("path_type", [pathlib.Path, str])
def test_writing_seqcoll(tmp_path, path_type):
    outpath = tmp_path / "alignment_output.c3h5u"
    coll = cogent3.get_dataset("brca1").degap()
    assert not outpath.exists()
    coll.write(path_type(outpath))
    assert outpath.exists()


def test_writing_seqcoll_wrong_suffix(tmp_path):
    outpath = tmp_path / "alignment_output.c3h5a"  # invalid suffix
    coll = cogent3.get_dataset("brca1").degap()
    with pytest.raises(ValueError):
        coll.write(outpath)


def test_writing_alignment_wrong_suffix(tmp_path):
    outpath = tmp_path / "alignment_output.c3h5u"  # invalid suffix
    coll = cogent3.get_dataset("brca1")
    with pytest.raises(ValueError):
        coll.write(outpath)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_load_unaligned_wrong_suffix(tmp_path, suffix):
    outpath = tmp_path / f"alignment_output.{suffix}"
    aln = cogent3.get_dataset("brca1")
    aln.write(outpath)
    # alignment invalid for unaligned
    with pytest.raises(ValueError):
        cogent3.load_unaligned_seqs(outpath, moltype="dna")


def test_load_aligned_wrong_suffix(tmp_path):
    outpath = tmp_path / "alignment_output.c3h5u"
    coll = cogent3.get_dataset("brca1").degap()
    coll.write(outpath)
    # unaligned invalid for aligned
    with pytest.raises(ValueError):
        cogent3.load_aligned_seqs(outpath, moltype="dna")


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_aligned_sparse", "small_unaligned"]
)
def test_set_attr(fxt, request):
    obj = request.getfixturevalue(fxt)
    obj.set_attr("test", "2")
    # calling again has no effect
    obj.set_attr("test", "1")
    # unless you use force
    obj.set_attr("test", "1", force=True)
    assert obj.get_attr("test") == "1"
    copy = obj.copy()
    assert copy.get_attr("test") == "1"


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_set_attr_invalid_type(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(TypeError):
        obj.set_attr("test", numpy.array("acbgdqwertyuiop", dtype="U<15"))

    with pytest.raises(TypeError):
        obj.set_attr("test", {"a": 1, "b": 2})


@pytest.mark.parametrize(
    "fxt", ["small_aligned", "small_aligned_sparse", "small_unaligned"]
)
def test_get_attr_missing(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_attr("missing")


def test_set_attr_invalid(h5seq_file):
    load = c3h5_load_funcs[h5seq_file.suffix[1:]]
    obj = load(path=h5seq_file, mode="r", check=False)
    with pytest.raises(PermissionError):
        obj.set_attr("test", "1")


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
@pytest.mark.parametrize("index", ["s1", 0])
def test_get_ungapped(fxt, request, index):
    obj = request.getfixturevalue(fxt)
    ungapped = obj[index]
    assert ungapped.str_value == "TGGACGG"


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_gapped_seq_invalid_pos(fxt, request, arg):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(ValueError):
        obj.get_gapped_seq_array(seqid="s1", **{arg: -1})


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_gapped_seq_invalid_seqid(fxt, request, arg):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_gapped_seq_array(seqid="s99")


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_get_gapped_seq_str(fxt, request, raw_aligned_data):
    obj = request.getfixturevalue(fxt)
    got = obj[0]
    expect = raw_aligned_data["s1"]
    assert got.gapped_str_value == expect
    s = obj.get_gapped_seq_str(seqid="s1")
    assert s == expect


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_get_gapped_seq_bytes(fxt, request, raw_aligned_data):
    obj = request.getfixturevalue(fxt)
    got = obj[0]
    expect = raw_aligned_data[got.seqid].encode("utf-8")
    assert got.gapped_bytes_value == expect
    s = obj.get_gapped_seq_bytes(seqid="s1")
    assert s == expect


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_get_gapped_seq_array_invalidseqid(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        _ = obj.get_gapped_seq_array(seqid="blah")


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_get_pos_range_invalid_name(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_pos_range(names=["missing"])


@pytest.mark.parametrize(
    "suffix", [cogent3_h5seqs.ALIGNED_SUFFIX, cogent3_h5seqs.SPARSE_SUFFIX]
)
def test_get_pos_range_valid(suffix, dna_alpha):
    func = c3h5_make_funcs[suffix]
    data = {"s1": "TGG--ACGG", "s2": "TGGGCAGTA", "s3": "---GCACTG"}
    obj = func(None, data=data.copy(), in_memory=True, alphabet=dna_alpha)
    full = numpy.array([dna_alpha.to_indices(data[n]) for n in obj.names])
    got = obj.get_pos_range(obj.names)
    numpy.testing.assert_equal(got, full)
    sl = slice(2, 8)
    got = obj.get_pos_range(obj.names, start=sl.start, stop=sl.stop)
    numpy.testing.assert_equal(got, full[:, sl])
    sl = slice(2, 8, 2)
    got = obj.get_pos_range(obj.names, start=sl.start, stop=sl.stop, step=sl.step)
    numpy.testing.assert_equal(got, full[:, sl])


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_ungapped_invalid_coord(fxt, request, arg):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(ValueError):
        obj.get_ungapped(name_map={"s1": "s1"}, **{arg: -1})


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_add_seqs_invalid_length(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(ValueError):
        obj.add_seqs({"s5": "ACGT"})


def test_write_seqs_data_invalid_suffix():
    with pytest.raises(ValueError):
        cogent3_h5seqs.write_seqs_data(path="wrong-suffix.h5seqs", seqcoll={})


def test_write_seqs_data_invalid_coll():
    with pytest.raises(TypeError):
        cogent3_h5seqs.write_seqs_data(path="wrong-type.c3h5u", seqcoll={})


def test_open_file_fails(tmp_path):
    path = tmp_path / "test.h5seqs"
    with pytest.raises(OSError):
        cogent3_h5seqs.open_h5_file(path, mode="r", in_memory=False)


@pytest.mark.parametrize("sparse", [True, False])
def test_get_hash(raw_aligned_data, dna_alpha, sparse):
    unaligned = cogent3_h5seqs.make_unaligned(
        "memory", data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
    )
    aligned = cogent3_h5seqs.make_aligned(
        "memory",
        data=raw_aligned_data,
        in_memory=True,
        alphabet=dna_alpha,
        sparse=sparse,
    )
    seqid = "s1"
    h_u = unaligned.get_hash(seqid)
    h_a = aligned.get_hash(seqid)
    assert h_u == h_a


@pytest.mark.parametrize("fxt", ["small", "small_aligned", "small_aligned_sparse"])
def test_get_hash_missing(fxt, request):
    small = request.getfixturevalue(fxt)
    h = small.get_hash(seqid="missing")
    assert h is None


@pytest.mark.parametrize(
    "func",
    [cogent3_h5seqs.make_aligned, cogent3_h5seqs.make_unaligned],
)
def test_invalid_path(func, raw_aligned_data, dna_alpha):
    with pytest.raises(TypeError):
        func({}, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_invalid_alphabet(suffix, raw_aligned_data):
    func = c3h5_make_funcs[suffix]
    with pytest.raises(ValueError):
        func(None, data=raw_aligned_data, in_memory=True, alphabet=None, mode="w")


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_to_alphabet_invalid(suffix):
    func = c3h5_make_funcs[suffix]
    prot = cogent3.get_moltype("protein").most_degen_alphabet()
    dna = cogent3.get_moltype("dna").most_degen_alphabet()
    data = {"Human": "CGTNTHASSL", "Mouse": "CGTDAHASSL", "Rhesus": "CGTNTHASSL"}

    storage = func(None, data=data, in_memory=True, alphabet=prot)
    with pytest.raises(cogent3.core.alphabet.AlphabetError):
        storage.to_alphabet(dna)


def subset_seqcoll_default(suffix, rename):
    make_func = c3_make_funcs[suffix]
    data = {"s1": "TGG--ACGG", "s2": "TGGGCAGTA", "s3": "---GCACTG"}
    coll = make_func(
        data,
        moltype="dna",
        info={"aligned": make_func == cogent3.make_aligned_seqs},
    )
    names = ["S1", "S3"] if rename else ["s1", "s3"]
    if rename:
        coll = coll.rename_seqs(lambda x: x.upper())
        names = ["S1", "S3"]
    return coll.take_seqs(names)


def subset_seqcoll_h5(suffix, rename):
    make_func = c3_make_funcs[suffix]
    data = {"s1": "TGG--ACGG", "s2": "TGGGCAGTA", "s3": "---GCACTG"}
    coll = make_func(
        data,
        moltype="dna",
        info={"aligned": make_func == cogent3.make_aligned_seqs},
        storage_backend=suffix,
    )
    names = ["S1", "S3"] if rename else ["s1", "s3"]
    if rename:
        coll = coll.rename_seqs(lambda x: x.upper())
        names = ["S1", "S3"]
    return coll.take_seqs(names)


@pytest.mark.parametrize("storage_func", [subset_seqcoll_h5, subset_seqcoll_default])
@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
@pytest.mark.parametrize("rename", [True, False])
def test_write_subsets(storage_func, suffix, rename, tmp_path):
    subset = storage_func(suffix, rename=rename)
    aligned = subset.info["aligned"]
    suffix = (
        cogent3_h5seqs.ALIGNED_SUFFIX if aligned else cogent3_h5seqs.UNALIGNED_SUFFIX
    )
    outpath = tmp_path / f"subset_output.{suffix}"
    subset.write(outpath)
    load_func = cogent3.load_aligned_seqs if aligned else cogent3.load_unaligned_seqs
    got = load_func(outpath, moltype="dna")
    assert got.to_dict() == subset.to_dict()
    cls = (
        cogent3_h5seqs.AlignedSeqsData if aligned else cogent3_h5seqs.UnalignedSeqsData
    )
    assert isinstance(got.storage, cls)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_write_custom_suffix(tmp_path, suffix):
    seqcoll = subset_seqcoll_default(suffix, rename=False)
    aligned = suffix != cogent3_h5seqs.UNALIGNED_SUFFIX
    suffix = "aligned" if aligned else "unaligned"
    kwargs = {"aligned_suffix": suffix} if aligned else {"unaligned_suffix": suffix}
    outpath = tmp_path / f"custom_suffix_output.{suffix}"
    cogent3_h5seqs.write_seqs_data(path=outpath, seqcoll=seqcoll, **kwargs)
    assert outpath.exists()


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_make_custom_suffix(suffix, dna_alpha):
    make_func = c3h5_make_funcs[suffix]
    suffix = "unaligned" if suffix.endswith("u") else "aligned"
    obj = make_func("memory", mode="w", suffix=suffix, alphabet=dna_alpha)
    assert obj.filename_suffix == suffix


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_repr(raw_aligned_data, dna_alpha, suffix):
    make_func = c3h5_make_funcs[suffix]
    obj = make_func("memory", data=raw_aligned_data, mode="w", alphabet=dna_alpha)
    part = f"alphabet='{''.join(dna_alpha)}'"
    assert part in repr(obj)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_repr_bytes(raw_aligned_data, suffix):
    alpha = cogent3.get_moltype("bytes").most_degen_alphabet()
    make_func = c3h5_make_funcs[suffix]
    obj = make_func("memory", data=raw_aligned_data, mode="w", alphabet=alpha)
    part = "alphabet=bytes"
    assert part in repr(obj)


def test_set_name_to_hash_no_data():
    h5file = cogent3_h5seqs.open_h5_file("memory", mode="w")
    # this should not fail
    cogent3_h5seqs._set_name_to_hash_to_index(h5file=h5file, name_to_hash=None)  # noqa: SLF001


def test_set_name_to_hash_read_only(tmp_path):
    h5path = tmp_path / "test.h5"
    h5file = cogent3_h5seqs.open_h5_file(h5path, mode="w")
    h5file.close()
    # now read only
    h5file = cogent3_h5seqs.open_h5_file(h5path, mode="r")
    # this should not fail
    cogent3_h5seqs._set_name_to_hash_to_index(
        h5file=h5file, name_to_hash={"s1": "not really a hash"}
    )


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
@pytest.mark.parametrize("compression", [True, False])
def test_toggle_compression_make(suffix, compression):
    mk_cls = c3_make_funcs[suffix]
    storage = suffix
    kwargs = {"compression": compression, "storage_backend": storage}
    exp_compress = "lzf" if compression else None
    data = {"s1": "TGG--ACGG", "s2": "TGGGCAGTA", "s3": "---GCACTG"}
    seqcoll = mk_cls(data, moltype="dna", **kwargs)
    seqhash = seqcoll.storage.get_hash("s1")
    grp = seqcoll.storage._primary_grp
    dataset = f"{grp}/{seqhash}"
    record = seqcoll.storage.h5file[dataset]
    assert record.compression == exp_compress


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
@pytest.mark.parametrize("compression", [True, False])
def test_toggle_compression_load(tmp_path, suffix, compression):
    load_cls = c3_load_funcs[suffix]

    data = {"s1": "TGG--ACGG", "s2": "TGGGCAGTA", "s3": "---GCACTG"}
    aln = cogent3.make_aligned_seqs(data, moltype="dna")
    outpath = tmp_path / "test.fa"
    aln.write(outpath)

    storage = suffix
    kwargs = {"compression": compression, "storage_backend": storage}
    exp_compress = "lzf" if compression else None
    seqcoll = load_cls(outpath, moltype="dna", **kwargs)
    seqhash = seqcoll.storage.get_hash("s1")
    grp = seqcoll.storage._primary_grp
    dataset = f"{grp}/{seqhash}"
    record = seqcoll.storage.h5file[dataset]
    assert record.compression == exp_compress


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.SPARSE_SUFFIX,
    ],
)
def test_write_aligned_excludes_gaps(tmp_path, request, suffix):
    # we don't want gaps and ungapped sequences in the output
    fixtures = {
        cogent3_h5seqs.ALIGNED_SUFFIX: "small_aligned",
        cogent3_h5seqs.SPARSE_SUFFIX: "small_aligned_sparse",
    }
    outpath = tmp_path / f"test.{suffix}"
    obj = request.getfixturevalue(fixtures[suffix])
    _ = obj.get_seq_length(obj.names[0])
    obj.write(outpath)
    obj.close()
    got = c3h5_load_funcs[suffix](outpath)
    assert got is not None
    assert "gaps" not in got.h5file


@pytest.mark.parametrize(
    "suffix", [cogent3_h5seqs.ALIGNED_SUFFIX, cogent3_h5seqs.SPARSE_SUFFIX]
)
def test_write_alignment_excludes_gaps(tmp_path, raw_aligned_data, suffix):
    outpath = tmp_path / f"test.{suffix}"
    kwargs = {} if suffix.endswith("a") else {"sparse": True}
    aln = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", storage_backend="h5seqs_aligned", **kwargs
    )
    # trigger extracting gaps
    _ = aln.seqs["s1"].seq
    assert "gaps" in aln.storage.h5file
    aln.write(outpath)
    del aln
    func = (
        cogent3_h5seqs.load_seqs_data_aligned
        if suffix.endswith("a")
        else cogent3_h5seqs.load_seqs_data_sparse
    )
    got = func(outpath)
    assert got is not None
    assert "gaps" not in got.h5file


@pytest.mark.parametrize("seqid", ["s1", "s2"])
def test_sparse_set_get(raw_aligned_data, dna_alpha, seqid):
    asd = cogent3_h5seqs.SparseSeqsData.from_seqs(
        data=raw_aligned_data.copy(), alphabet=dna_alpha, sparse=True
    )
    got = asd.get_gapped_seq_str(seqid=seqid)
    expect = raw_aligned_data[seqid]
    assert got == expect


def test_make_sparse_bad_ref(raw_aligned_data, dna_alpha):
    with pytest.raises(ValueError):
        c3h5_make_funcs["c3h5s"](
            "memory",
            data=raw_aligned_data,
            alphabet=dna_alpha,
            ref_name="bad",
            in_memory=True,
        )


def test_make_sparse_no_ref(dna_alpha):
    obj = c3h5_make_funcs["c3h5s"](
        "memory",
        data=None,
        alphabet=dna_alpha,
        ref_name="bad",
        in_memory=True,
    )
    with pytest.raises(ValueError):
        obj._ref_seq


def test_sparse_set_invalid_ref(small_aligned_sparse, tmp_path, dna_alpha):
    outpath = tmp_path / f"test.{small_aligned_sparse.filename_suffix}"
    small_aligned_sparse.write(outpath)
    small_aligned_sparse.close()
    h5file = cogent3_h5seqs.open_h5_file(outpath, mode="r")
    with pytest.raises(ValueError):
        _ = cogent3_h5seqs.SparseSeqsData(
            gapped_seqs=h5file, alphabet=dna_alpha, ref_name="missing"
        )


def test_duplicate_h5_file_exclude_grp(small_aligned_sparse):
    # trigger creation of ungapped and gaps groups
    _ = small_aligned_sparse.get_seq_array(seqid="s2")
    assert "ungapped" in small_aligned_sparse.h5file
    dup = cogent3_h5seqs.duplicate_h5_file(
        h5file=small_aligned_sparse.h5file,
        path="memory",
        in_memory=True,
        exclude_groups={"gaps", "ungapped"},
    )
    assert "ungapped" not in dup


def test_get_seq_length_with_wout_gaps_present(small_aligned_sparse):
    # trigger creation of ungapped and gaps groups
    orig = small_aligned_sparse.get_seq_length(seqid="s2")
    _ = small_aligned_sparse.get_seq_array(seqid="s2")
    after = small_aligned_sparse.get_seq_length(seqid="s2")
    assert orig == after


def test_selecting_dtype():
    with pytest.raises(ValueError):
        cogent3_h5seqs._best_uint_dtype(2**64 + 10)  # noqa: SLF001


def test_make_seqs_invalid_chars():
    from cogent3.core.alphabet import AlphabetError

    data = {"seq1": "AGT1CCT", "seq2": "AGT$CCC"}
    with pytest.raises(AlphabetError):
        cogent3.make_aligned_seqs(data, moltype="dna", storage_backend="c3h5s")


@pytest.mark.parametrize(
    "suffix",
    [cogent3_h5seqs.SPARSE_SUFFIX, cogent3_h5seqs.ALIGNED_SUFFIX],
)
def test_get_pos_range_identical_seqs(suffix, dna_alpha):
    """correctly identify variable positions"""
    seq = "GCGAC"
    new_seqs = {"A": seq, "B": seq, "C": seq}
    obj = c3h5_make_funcs[suffix](
        None, data=new_seqs, alphabet=dna_alpha, in_memory=True
    )
    array = obj.get_pos_range(names=["A", "B", "C"])
    assert numpy.all(array == dna_alpha.to_indices(seq), axis=1).all()


def test_omit_bad_seqs():
    """omit_bad_seqs should return alignment w/o seqs causing most gaps"""
    data = {
        "s1": "---ACC---TT-",
        "s2": "---ACC---TT-",
        "s3": "---ACC---TT-",
        "s4": "--AACCG-GTT-",
        "s5": "--AACCGGGTTT",
        "s6": "AGAACCGGGTT-",
    }

    aln = cogent3.make_aligned_seqs(data, moltype="dna", storage_backend="c3h5s")
    # with defaults, excludes s6
    expect = data.copy()
    del expect["s6"]
    result = aln.omit_bad_seqs()
    assert result.to_dict() == expect
    # with quantile 0.5, just s1, s2, s3
    expect = data.copy()
    for key in ("s6", "s5"):
        del expect[key]
    result = aln.omit_bad_seqs(0.5)
    assert result.to_dict() == expect


@pytest.mark.parametrize(
    "suffix",
    [cogent3_h5seqs.SPARSE_SUFFIX, cogent3_h5seqs.ALIGNED_SUFFIX],
)
def test_aln_multi_add_seqs(suffix):
    data = {"name1": "AAA", "name2": "A--", "name3": "AAA", "name4": "AAA"}
    data2 = {"name5": "TTT", "name6": "---"}
    aln = cogent3.make_aligned_seqs(data, moltype="dna", storage_backend=suffix)
    out_aln = aln.add_seqs(data2)
    assert len(out_aln.names) == 6


@pytest.mark.parametrize(
    "suffix",
    [cogent3_h5seqs.SPARSE_SUFFIX, cogent3_h5seqs.ALIGNED_SUFFIX],
)
def test_counts_per_seq_bytes_moltype(suffix):
    """produce correct counts per seq with text moltypes"""
    data = {"a": "AAAA??????", "b": "CCCGGG--NN", "c": "CCGGTTCCAA"}
    coll = cogent3.make_aligned_seqs(data, moltype="bytes", storage_backend=suffix)
    got = coll.counts_per_seq(include_ambiguity=True, allow_gap=True)
    assert got.col_sum()[b"-"] == 2
    assert got.col_sum()[b"?"] == 6
    assert got.col_sum()[b"T"] == 2


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_pickling_in_memory(suffix, dna_alpha):
    # always fails
    import pickle

    raw = {"s1": "ACG-T", "s2": "ACGGT"}
    obj = c3h5_make_funcs[suffix](None, data=raw, in_memory=True, alphabet=dna_alpha)
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(obj)


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
        cogent3_h5seqs.UNALIGNED_SUFFIX,
    ],
)
def test_pickling_on_disk(suffix, dna_alpha, tmp_path):
    import pickle

    outpath = tmp_path / f"test.{suffix}"
    raw = {"s1": "ACG-T", "s2": "ACGGT"}
    obj = c3h5_make_funcs[suffix](
        outpath, data=raw.copy(), alphabet=dna_alpha, mode="w"
    )
    obj.write(outpath)
    obj.close()
    obj = c3h5_load_funcs[suffix](outpath, mode="r")
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    names = unpickled.names

    assert set(names) == set(raw.keys())
    unpickled.close()


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
def test_variable_positions(suffix, dna_alpha):
    raw = {"s1": "ACGGT", "s2": "ACGGT"}
    obj = c3h5_make_funcs[suffix](None, data=raw.copy(), alphabet=dna_alpha, mode="w")
    pos = obj.variable_positions(obj.names)
    assert pos.size == 0
    raw = {"s1": "ACGGT", "s2": "ACGAT"}
    obj = c3h5_make_funcs[suffix](None, data=raw.copy(), alphabet=dna_alpha, mode="w")
    pos = obj.variable_positions(obj.names)
    assert pos.size == 1
    assert pos[0] == 3
    cols = obj.get_positions(obj.names, pos)
    expect = numpy.array([[3], [2]], dtype=numpy.uint8)
    assert (cols == expect).all()
    pos = obj.variable_positions(obj.names, start=1)
    assert pos.size == 1
    # position values are absolute, i.e. start should have no effect
    assert pos[0] == 3
    # no variable positions within segment
    pos = obj.variable_positions(obj.names, stop=3)
    assert pos.size == 0
    # no variable positions modulo 2
    pos = obj.variable_positions(obj.names, step=2)
    assert pos.size == 0
    # no variable positions modulo 2, unless we start from 1
    pos = obj.variable_positions(obj.names, start=1, step=2)
    assert pos.size == 1
    assert pos[0] == 3


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
def test_variable_positions_one_seq(suffix, dna_alpha):
    raw = {"s1": "ACGGT", "s2": "ACGGT"}
    obj = c3h5_make_funcs[suffix](None, data=raw.copy(), alphabet=dna_alpha, mode="w")
    pos = obj.variable_positions(obj.names[:1])
    assert pos.size == 0


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
def test_variable_positions_no_variation(suffix, dna_alpha):
    raw = {"s1": "", "s2": ""}
    obj = c3h5_make_funcs[suffix](None, data=raw.copy(), alphabet=dna_alpha, mode="w")
    pos = obj.variable_positions(obj.names)
    assert pos.size == 0


@pytest.fixture
def raw_5seq_pos():
    return {
        "s1": "ACCTG",
        "s2": "-CCAG",
        "s3": "ACCTG",  # same as s1
        "s4": "ACC-A",
        "s5": "TCCTG",
    }


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
@pytest.mark.parametrize(
    "names", list(itertools.combinations(["s1", "s2", "s3", "s4", "s5"], 3))
)
def test_variable_positions_subset_seqs(raw_5seq_pos, suffix, dna_alpha, names):
    array_data = {n: dna_alpha.to_indices(s) for n, s in raw_5seq_pos.items()}
    aln = cogent3.make_aligned_seqs(array_data, moltype="dna", storage_backend=suffix)
    sub = aln.take_seqs(names)
    array_seqs = sub.array_seqs
    expect = tuple(numpy.where((array_seqs != array_seqs[0]).any(axis=0))[0])
    got = sub.variable_positions(include_gap_motif=True)
    assert got == expect


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
def test_variable_positions_subset_diff_to_whole(raw_5seq_pos, suffix, dna_alpha):
    array_data = {n: dna_alpha.to_indices(s) for n, s in raw_5seq_pos.items()}
    aln = cogent3.make_aligned_seqs(array_data, moltype="dna", storage_backend=suffix)
    sub = aln.take_seqs(["s1", "s3"])
    assert sub.storage is aln.storage
    array_seqs = sub.array_seqs
    expect = tuple(numpy.where((array_seqs != array_seqs[0]).any(axis=0))[0])
    got = sub.variable_positions(include_gap_motif=True)
    assert got == expect
    assert sub.variable_positions(include_gap_motif=True) != aln.variable_positions(
        include_gap_motif=True
    )


@pytest.mark.parametrize(
    "names",
    [["s1", "s2", "s3", "s4", "s5"], ["s2", "s3", "s4"], ["s5", "s1", "s3", "s4"]],
)
@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
def test_get_positions_name_order(raw_5seq_pos, dna_alpha, names, suffix):
    array_data = {n: dna_alpha.to_indices(s) for n, s in raw_5seq_pos.items()}
    expect = numpy.array([array_data[n] for n in names], dtype=dna_alpha.dtype)
    obj = c3h5_make_funcs[suffix](
        None, data=raw_5seq_pos.copy(), alphabet=dna_alpha, mode="w"
    )
    got = obj.get_positions(names=names, positions=numpy.arange(expect.shape[1]))
    assert (got == expect).all()


@pytest.mark.parametrize(
    "posns",
    [[0, 1, 2], [0, 3, 4], [1, 2, 4]],
)
@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
def test_get_positions_posns(raw_5seq_pos, dna_alpha, posns, suffix):
    names = sorted(raw_5seq_pos.keys())
    array_data = {n: dna_alpha.to_indices(s) for n, s in raw_5seq_pos.items()}
    expect = numpy.array([array_data[n][posns] for n in names], dtype=dna_alpha.dtype)
    obj = c3h5_make_funcs[suffix](
        None, data=raw_5seq_pos.copy(), alphabet=dna_alpha, mode="w"
    )
    got = obj.get_positions(names=names, positions=posns)
    assert (got == expect).all()


@pytest.mark.parametrize(
    "suffix",
    [
        cogent3_h5seqs.SPARSE_SUFFIX,
        cogent3_h5seqs.ALIGNED_SUFFIX,
    ],
)
def test_get_positions_no_posns(raw_5seq_pos, dna_alpha, suffix):
    names = sorted(raw_5seq_pos.keys())
    obj = c3h5_make_funcs[suffix](
        None, data=raw_5seq_pos.copy(), alphabet=dna_alpha, mode="w"
    )
    with pytest.raises(NotImplementedError):
        obj.get_positions(names=names, positions=[])


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_get_positions_invalid_name(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_positions(names=["missing"], positions=[0, 1])


@pytest.mark.parametrize("fxt", ["small_aligned", "small_aligned_sparse"])
def test_get_positions_invalid_pos_values(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(IndexError):
        obj.get_positions(names=["s1"], positions=[-1, 0, 1, 100_000])

    with pytest.raises(IndexError):
        obj.get_positions(names=["s1"], positions=[0, 1, 100_000])


def test_sparse_write_read(tmp_path, raw_5seq_pos):
    aln = cogent3.make_aligned_seqs(raw_5seq_pos.copy(), moltype="dna")
    outpath = tmp_path / f"demo.{cogent3_h5seqs.SPARSE_SUFFIX}"
    aln.write(outpath)
    ld = cogent3.load_aligned_seqs(outpath, moltype="dna")
    assert ld.to_dict() == raw_5seq_pos
