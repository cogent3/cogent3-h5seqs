import pathlib

import cogent3
import numpy
import pytest

import cogent3_h5seqs


@pytest.fixture
def dna_alpha():
    return cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()


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
    rna = cogent3.get_moltype("rna", new_type=True).most_degen_alphabet()
    mod = small.to_alphabet(rna)
    assert numpy.allclose(numpy.array(mod["s1"]), numpy.array(small["s1"]))
    assert str(mod["s2"]) == str(small["s2"]).replace("T", "U")


@pytest.mark.parametrize("fxt", ["small", "small_aligned"])
def test_dna_to_text(fxt, request):
    small = request.getfixturevalue(fxt)
    text = cogent3.get_moltype("text", new_type=True).most_degen_alphabet()
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


@pytest.mark.parametrize("fxt", ["small", "small_aligned"])
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
    "func",
    [cogent3_h5seqs.load_seqs_data_unaligned, cogent3_h5seqs.load_seqs_data_aligned],
)
def test_load_invalid(tmp_path, func):
    path = tmp_path / "wrong-suffix.h5seqs"
    with pytest.raises(ValueError):
        func(path)


@pytest.mark.parametrize(
    "func",
    [cogent3_h5seqs.make_aligned, cogent3_h5seqs.make_unaligned],
)
def test_equality(raw_aligned_data, dna_alpha, func):
    store1 = func(None, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha)
    store2 = func(None, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha)
    assert store1 == store2


@pytest.mark.parametrize(
    "func",
    [cogent3_h5seqs.make_aligned, cogent3_h5seqs.make_unaligned],
)
def test_inequality(raw_aligned_data, dna_alpha, func):
    store1 = func(None, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha)
    # wrong type
    assert store1 != "string"
    # seqnames different
    store2 = func(
        None,
        data={k: v for k, v in raw_aligned_data.items() if k != "s1"},
        in_memory=True,
        alphabet=dna_alpha,
    )
    assert store1 != store2
    # sequence different
    data_edited = raw_aligned_data.copy()
    data_edited["s1"] = data_edited["s1"][:-1] + "N"
    store2 = func(
        None,
        data=data_edited,
        in_memory=True,
        alphabet=dna_alpha,
    )
    assert store1 != store2
    # attrs different
    store2 = func(
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
    store2 = func(
        None,
        data=raw_aligned_data,
        in_memory=True,
        alphabet=dna_alpha,
        offset={"s1": 2},
    )
    store2.set_attr("test", "2")
    assert store1 != store2


def test_make_alignedseqsdata(raw_aligned_data, dna_alpha):
    asd = cogent3_h5seqs.make_aligned(
        path=None, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
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
    seqs = cogent3.load_unaligned_seqs(h5_unaligned_path, moltype="dna", new_type=True)
    assert seqs.to_dict() == raw_data


@pytest.fixture
def small_aligned(raw_aligned_data, dna_alpha):
    return cogent3_h5seqs.make_aligned(
        path=None, data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
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


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
def test_getitem_str_int(fxt, request):
    obj = request.getfixturevalue(fxt)
    seqid = "s1"
    index = obj.names.index(seqid)
    str_got = obj[seqid]
    int_got = obj[index]
    assert str(str_got) == str(int_got)


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
def test_getitem_err(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(TypeError):
        obj[20.0]


@pytest.fixture(params=[cogent3_h5seqs.make_aligned, cogent3_h5seqs.make_unaligned])
def h5seq_file(request, tmp_path, dna_alpha):
    make = request.param
    aligned = make == cogent3_h5seqs.make_aligned
    suffix = (
        cogent3_h5seqs.ALIGNED_SUFFIX if aligned else cogent3_h5seqs.UNALIGNED_SUFFIX
    )
    path = tmp_path / f"test.{suffix}"
    obj = make(path, mode="w", alphabet=dna_alpha)
    obj.close()
    yield path
    path.unlink(missing_ok=True)


def test_add_seqs_not_writeable(h5seq_file):
    load = (
        cogent3_h5seqs.load_seqs_data_aligned
        if h5seq_file.suffix.endswith(cogent3_h5seqs.ALIGNED_SUFFIX)
        else cogent3_h5seqs.load_seqs_data_unaligned
    )
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


def test_aligned_add_seqs_duplicates_disallowed(small_aligned, raw_aligned_data):
    with pytest.raises(ValueError):
        small_aligned.add_seqs(raw_aligned_data, force_unique_keys=True)


def test_aligned_add_seqs_duplicates_allowed(small_aligned, raw_aligned_data):
    num_seqs = len(small_aligned.names)
    small_aligned.add_seqs(raw_aligned_data, force_unique_keys=False)
    assert len(small_aligned.names) == num_seqs


def test_unaligned_add_seqs_duplicates_disallowed(small_unaligned, raw_data):
    with pytest.raises(ValueError):
        small_unaligned.add_seqs(raw_data, force_unique_keys=True)


def test_unaligned_add_seqs_duplicates_allowed(small_unaligned, raw_data):
    num_seqs = len(small_unaligned.names)
    small_unaligned.add_seqs(raw_data, force_unique_keys=False)
    assert len(small_unaligned.names) == num_seqs


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
def test_get_seq_length(fxt, request, raw_data):
    obj = request.getfixturevalue(fxt)
    # seq s2 is the same between the aligned and unaligned examples
    l = obj.get_seq_length(seqid="s2")
    expect = len(raw_data["s2"])
    assert l == expect


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
def test_get_seq_length_invalid_seqid(fxt, request):
    obj = request.getfixturevalue(fxt)
    # seq s2 is the same between the aligned and unaligned examples
    with pytest.raises(KeyError):
        obj.get_seq_length(seqid="missing")


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
def test_get_seq_array(fxt, request, raw_data, dna_alpha):
    obj = request.getfixturevalue(fxt)
    # seq s2 is the same between the aligned and unaligned examples
    s = obj.get_seq_array(seqid="s2")
    expect = dna_alpha.to_indices(raw_data["s2"])
    assert (s == expect).all()


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_seq_array_invalid_pos(fxt, request, arg):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(ValueError):
        obj.get_seq_array(seqid="s2", **{arg: -1})


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
def test_get_seq_array_invalid_seqid(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_seq_array(seqid="missing")


@pytest.mark.parametrize(
    "mk_obj", [cogent3.make_unaligned_seqs, cogent3.make_aligned_seqs]
)
def test_from_storage(mk_obj, raw_aligned_data):
    aligned = mk_obj == cogent3.make_aligned_seqs
    storage_backend = "h5seqs_aligned" if aligned else "h5seqs_unaligned"
    coll = mk_obj(
        raw_aligned_data,
        moltype="dna",
        new_type=True,
        storage_backend=storage_backend,
        in_memory=True,
    )
    got = coll.storage.from_storage(coll, in_memory=True)
    assert got is not coll
    assert got == coll.storage


@pytest.mark.parametrize(
    "mk_obj", [cogent3.make_unaligned_seqs, cogent3.make_aligned_seqs]
)
def test_from_storage_invalid(mk_obj, raw_aligned_data):
    aligned = mk_obj == cogent3.make_aligned_seqs
    storage_backend = "h5seqs_aligned" if aligned else "h5seqs_unaligned"
    coll = mk_obj(
        raw_aligned_data,
        moltype="dna",
        new_type=True,
        storage_backend=storage_backend,
        in_memory=True,
    )
    with pytest.raises(TypeError):
        coll.storage.from_storage({}, in_memory=False)


def test_aligned_from_names_and_array(small_aligned):
    names = small_aligned.names
    data = numpy.array(
        [small_aligned.get_gapped_seq_array(seqid=name) for name in names],
        dtype=small_aligned.alphabet.dtype,
    )
    got = small_aligned.from_names_and_array(
        names=names, data=data, alphabet=small_aligned.alphabet
    )
    assert got == small_aligned
    assert got is not small_aligned


def test_aligned_from_names_and_array_invalid(small_aligned):
    names = small_aligned.names
    data = numpy.array(
        [small_aligned.get_gapped_seq_array(seqid=name) for name in names],
        dtype=small_aligned.alphabet.dtype,
    )
    with pytest.raises(ValueError):
        small_aligned.from_names_and_array(
            names=names[:-1], data=data, alphabet=small_aligned.alphabet
        )


def test_aligned_from_names_and_array2(raw_aligned_data, dna_alpha):
    aln = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", new_type=True, storage_backend="h5seqs_aligned"
    )
    seqs = list(aln.seqs)
    gaps = {s.name: s.map.array for s in seqs}
    s = {s.name: numpy.array(s.seq) for s in seqs}
    got = cogent3_h5seqs.AlignedSeqsData.from_seqs_and_gaps(
        seqs=s, gaps=gaps, alphabet=dna_alpha
    )
    assert got == aln.storage


def test_aligned_get_ungapped(small_aligned, raw_aligned_data):
    aln = cogent3.make_aligned_seqs(small_aligned, new_type=True, moltype="dna")
    ungapped = aln.degap(storage_backend="h5seqs_unaligned")
    expect = {n: s.replace("-", "") for n, s in raw_aligned_data.items()}
    assert ungapped.to_dict() == expect
    assert isinstance(ungapped.storage, cogent3_h5seqs.UnalignedSeqsData)


@pytest.mark.parametrize("storage_backend", [None, "h5seqs_aligned"])
def test_write_aligned(raw_aligned_data, storage_backend, tmp_path):
    aln = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", new_type=True, storage_backend=storage_backend
    )
    outpath = tmp_path / f"aligned_output.{cogent3_h5seqs.ALIGNED_SUFFIX}"
    aln.write(outpath)
    assert outpath.exists()
    assert outpath.is_file()


def test_get_positions(raw_aligned_data):
    c3 = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", new_type=True, storage_backend=None
    )
    h5 = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", new_type=True, storage_backend="h5seqs_aligned"
    )
    assert (c3.array_seqs == h5.array_seqs).all()


@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_positions_invalid_coord(raw_aligned_data, arg):
    h5 = cogent3.make_aligned_seqs(
        raw_aligned_data, moltype="dna", new_type=True, storage_backend="h5seqs_aligned"
    )
    with pytest.raises(ValueError):
        h5.storage.get_positions(names=["s1", "s2"], **{arg: -1})


def test_set_as_default_drivers_unaligned(raw_aligned_data):
    cogent3.set_storage_defaults(unaligned_seqs="h5seqs_unaligned")

    coll = cogent3.make_unaligned_seqs(raw_aligned_data, moltype="dna", new_type=True)
    assert isinstance(coll.storage, cogent3_h5seqs.UnalignedSeqsData)

    cogent3.set_storage_defaults(reset=True)

    coll = cogent3.make_unaligned_seqs(raw_aligned_data, moltype="dna", new_type=True)
    assert not isinstance(coll.storage, cogent3_h5seqs.UnalignedSeqsData)


@pytest.mark.parametrize(
    "mk_obj", [cogent3_h5seqs.make_aligned, cogent3_h5seqs.make_unaligned]
)
def test_del(mk_obj, raw_aligned_data, tmp_path, dna_alpha):
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


@pytest.mark.parametrize("path_type", [pathlib.Path, str])
def test_writing_alignment(tmp_path, path_type):
    outpath = tmp_path / "alignment_output.c3h5a"
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


def test_load_unaligned_wrong_suffix(tmp_path):
    outpath = tmp_path / "alignment_output.c3h5a"
    aln = cogent3.get_dataset("brca1")
    aln.write(outpath)
    # alignment invalid for unaligned
    with pytest.raises(ValueError):
        cogent3.load_unaligned_seqs(outpath, moltype="dna", new_type=True)


def test_load_aligned_wrong_suffix(tmp_path):
    outpath = tmp_path / "alignment_output.c3h5u"
    coll = cogent3.get_dataset("brca1").degap()
    coll.write(outpath)
    # unaligned invalid for aligned
    with pytest.raises(ValueError):
        cogent3.load_aligned_seqs(outpath, moltype="dna", new_type=True)


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
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


def test_set_attr_invalid_type(small_aligned):
    with pytest.raises(TypeError):
        small_aligned.set_attr("test", numpy.array("acbgdqwertyuiop", dtype="U<15"))

    with pytest.raises(TypeError):
        small_aligned.set_attr("test", {"a": 1, "b": 2})


@pytest.mark.parametrize("fxt", ["small_aligned", "small_unaligned"])
def test_get_attr_missing(fxt, request):
    obj = request.getfixturevalue(fxt)
    with pytest.raises(KeyError):
        obj.get_attr("missing")


def test_set_attr_invalid(h5seq_file):
    load = (
        cogent3_h5seqs.load_seqs_data_aligned
        if h5seq_file.suffix.endswith(cogent3_h5seqs.ALIGNED_SUFFIX)
        else cogent3_h5seqs.load_seqs_data_unaligned
    )
    obj = load(path=h5seq_file, mode="r", check=False)
    with pytest.raises(PermissionError):
        obj.set_attr("test", "1")


@pytest.mark.parametrize("index", ["s1", 0])
def test_get_ungapped(small_aligned, index):
    ungapped = small_aligned[index]
    assert ungapped.str_value == "TGGACGG"


@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_gapped_seq_invalid_pos(small_aligned, arg):
    with pytest.raises(ValueError):
        small_aligned.get_gapped_seq_array(seqid="s1", **{arg: -1})


def test_get_gapped_seq_str(small_aligned, raw_aligned_data):
    got = small_aligned[0]
    expect = raw_aligned_data["s1"]
    assert got.gapped_str_value == expect
    s = small_aligned.get_gapped_seq_str(seqid="s1")
    assert s == expect


def test_get_gapped_seq_bytes(small_aligned, raw_aligned_data):
    got = small_aligned[0]
    expect = raw_aligned_data[got.seqid].encode("utf-8")
    assert got.gapped_bytes_value == expect
    s = small_aligned.get_gapped_seq_bytes(seqid="s1")
    assert s == expect


def test_get_positions_invalid_name(small_aligned):
    with pytest.raises(KeyError):
        small_aligned.get_positions(names=["missing"])


@pytest.mark.parametrize("arg", ["start", "stop", "step"])
def test_get_ungapped_invalid_coord(small_aligned, arg):
    with pytest.raises(ValueError):
        small_aligned.get_ungapped(name_map={"s1": "s1"}, **{arg: -1})


def test_gadd_seqs_invalid_length(small_aligned):
    with pytest.raises(ValueError):
        small_aligned.add_seqs({"s5": "ACGT"})


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


def test_get_hash(raw_aligned_data, dna_alpha):
    unaligned = cogent3_h5seqs.make_unaligned(
        "memory", data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
    )
    aligned = cogent3_h5seqs.make_aligned(
        "memory", data=raw_aligned_data, in_memory=True, alphabet=dna_alpha
    )
    seqid = "s1"
    h_u = unaligned.get_hash(seqid)
    h_a = aligned.get_hash(seqid)
    assert h_u == h_a


@pytest.mark.parametrize("fxt", ["small", "small_aligned"])
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
    "func",
    [cogent3_h5seqs.make_aligned, cogent3_h5seqs.make_unaligned],
)
def test_invalid_alphabet(func, raw_aligned_data):
    with pytest.raises(ValueError):
        func(None, data=raw_aligned_data, in_memory=True, alphabet=None, mode="w")


@pytest.mark.parametrize(
    "func",
    [cogent3_h5seqs.make_aligned, cogent3_h5seqs.make_unaligned],
)
def test_to_alphabet_invalid(func):
    prot = cogent3.get_moltype("protein", new_type=True).most_degen_alphabet()
    dna = cogent3.get_moltype("dna", new_type=True).most_degen_alphabet()
    data = {"Human": "CGTNTHASSL", "Mouse": "CGTDAHASSL", "Rhesus": "CGTNTHASSL"}

    storage = func(None, data=data, in_memory=True, alphabet=prot)
    with pytest.raises(cogent3.core.new_alphabet.AlphabetError):
        storage.to_alphabet(dna)
