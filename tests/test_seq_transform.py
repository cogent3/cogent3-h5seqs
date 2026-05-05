import cogent3 as c3
import numpy as np
import pytest

from cogent3_h5seqs.seq_transform import (
    count_bases_non_canonical_runs,
    pack_nucleic,
    unpack_nucleic,
)


@pytest.mark.parametrize(
    ("seq_str", "expected_count"),
    [
        ("ACGTAGGT", 0),  # No non-canonical characters
        ("ACGNNGTYAGGTT", 2),
        ("NNACGTN", 2),
        ("NNNN", 1),  # All non-canonical
        ("ANANANAN", 4),  # Alternating
        ("ACGTNNNNACGT", 1),  # Single run of non-canonical
        ("RYRYRY", 6),
    ],
)
def test_count_non_canonical_runs(seq_str, expected_count):
    # Test counting runs of non-canonical characters
    s = c3.make_seq(seq_str, moltype="dna")
    sarr = s.to_array()
    _, count = count_bases_non_canonical_runs(sarr)
    assert count == expected_count, f"Expected {expected_count}, got {count}"


@pytest.fixture
def seq_with_ambiguity():
    """Sequence: ACGNNGTYAGGTT (positions 0-12)
    Canonical: positions 0,1,2,5,6,8,9,10,11,12 -> packed 0-9
    Ambiguous: positions 3,4 (NN) and 7 (Y)
    """
    s = c3.make_seq("ACGNNGTYAGGTT", moltype="dna")
    sarr = s.to_array()
    packed, positions, length = pack_nucleic(sarr)
    return sarr, packed, positions, length


@pytest.fixture
def seq_no_ambiguity():
    s = c3.make_seq("ACGTAGGT", moltype="dna")
    sarr = s.to_array()
    packed, positions, length = pack_nucleic(sarr)
    return sarr, packed, positions, length


@pytest.fixture
def seq_no_ambiguity_long():
    """21-base canonical sequence for slice testing.

    Length 21 means: 6 full bytes (24 bases capacity) with 3 unused slots.
    """
    s = c3.make_seq("TCTATACCTGCGACCCGCGTC", moltype="dna")
    sarr = s.to_array()
    packed, positions, length = pack_nucleic(sarr)
    return sarr, packed, positions, length


def test_ambig_coords():
    #    012  34 56789
    s = c3.make_seq("NNNACGTNN", moltype="dna")
    sarr = s.to_array()
    _, positions, _ = pack_nucleic(sarr)
    pos, csum, _ = positions.T
    assert np.allclose(pos, [0, 4])
    assert np.allclose(csum, [3, 5])


def test_packed_and_ambig_coords():
    #    012  34 56789
    s = c3.make_seq("ACGNNGTYAGGTT", moltype="dna")
    alpha = s.moltype.most_degen_alphabet()
    n_idx = alpha.to_indices("N")[0]
    y_idx = alpha.to_indices("Y")[0]
    sarr = s.to_array()
    packed, positions, l = pack_nucleic(sarr)
    assert np.allclose(positions, [(3, 2, n_idx), (5, 3, y_idx)])
    assert l == 10

    # Verify the packed array is 2-bit encoded (4 bases per byte)
    # 10 canonical bases -> (10+3)//4 = 3 bytes
    assert len(packed) == 3, f"Expected 3 bytes, got {len(packed)}"

    # Unpack to verify canonical bases are correct
    unpacked = unpack_nucleic(packed, positions, l)
    assert (unpacked == sarr).all()


def test_unpack_nucleic_basic():
    # Test unpacking reconstructs the original sequence
    #    012  34 56789
    s = c3.make_seq("ACGNNGTYAGGTT", moltype="dna")
    sarr = s.to_array()

    # Pack the sequence
    packed, positions, length = pack_nucleic(sarr)

    # Unpack it
    unpacked = unpack_nucleic(packed, positions, length)

    # Should exactly match the original
    assert (unpacked == sarr).all()


def test_unpack_nucleic_no_ambiguity(seq_no_ambiguity):
    # Test unpacking sequence with no non-canonical characters
    sarr, packed, positions, length = seq_no_ambiguity

    unpacked = unpack_nucleic(packed, positions, length)

    # Should perfectly reconstruct the original
    assert (unpacked == sarr).all()


def test_unpack_nucleic_all_ambiguity():
    # Test unpacking sequence with only non-canonical characters
    s = c3.make_seq("NNNN", moltype="dna")
    sarr = s.to_array()

    packed, positions, length = pack_nucleic(sarr)
    unpacked = unpack_nucleic(packed, positions, length)

    # Should exactly match original
    assert (unpacked == sarr).all()


@pytest.mark.parametrize("seq_str", ["A", "N"])
def test_unpack_nucleic_single_base(seq_str):
    # Test unpacking single base sequences
    # Single canonical base
    s1 = c3.make_seq(seq_str, moltype="dna")
    sarr1 = s1.to_array()
    packed1, pos1, len1 = pack_nucleic(sarr1)
    unpacked1 = unpack_nucleic(packed1, pos1, len1)
    assert (unpacked1 == sarr1).all()


def test_adjacent_different_ambiguity_runs():
    # Test adjacent runs of different ambiguity characters (ACGGNNNYYTG)
    s = c3.make_seq("ACGGNNNYYTG", moltype="dna")
    sarr = s.to_array()

    packed, positions, length = pack_nucleic(sarr)

    # Verify positions array shape: 5 ambiguous chars (3 N's + 2 Y's)
    assert positions.shape == (2, 3), f"Expected shape (5, 3), got {positions.shape}"

    # Verify unpacking reconstructs the original sequence
    unpacked = unpack_nucleic(packed, positions, length)
    assert (unpacked == sarr).all(), "Unpacked sequence doesn't match original"


@pytest.mark.parametrize(
    "seq_str",
    [
        "ACGTACGT",  # No ambiguity
        "ACGNNGTYAGGTT",  # Multiple ambiguous runs
        "NACGT",  # Ambiguity at start
        "ACGTN",  # Ambiguity at end
        "ANANANAN",  # Alternating
        "A" * 100 + "N" * 10 + "C" * 100,  # Long sequence
        "ACGGNNNYYTG",  # Adjacent different ambiguity runs
        "NNNACGTNN",  # Ambiguity at both ends
    ],
)
def test_pack_unpack_roundtrip(seq_str):
    s = c3.make_seq(seq_str, moltype="dna")
    sarr = s.to_array()

    # Pack and unpack
    packed, positions, length = pack_nucleic(sarr)
    unpacked = unpack_nucleic(packed, positions, length)

    # Should exactly reconstruct the original
    assert (unpacked == sarr).all(), f"Failed for sequence: {seq_str}"


# seq_no_ambiguity_long fixture creates "ACGTACGTACGTACGTACGTA" (length 21)
@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (None, None),  # Full sequence (default)
        (0, 21),  # Full sequence with explicit bounds
        (0, 10),  # First half
        (10, 21),  # Second half
        (5, 15),  # Middle slice
        (0, 1),  # First element
        (20, 21),  # Last element
        (10, 11),  # Single element in middle
        (0, 0),  # Empty slice at start
        (10, 10),  # Empty slice in middle
        (21, 21),  # Empty slice at end
        (1, 20),  # Most of sequence except ends
        (0, 5),  # First byte + one base
        (3, 7),  # Starts mid-byte, ends mid-byte
        (0, 4),  # Exactly one byte
        (4, 8),  # Exactly second byte
        (0, 8),  # Exactly two bytes
        (2, 10),  # Starts mid-byte, spans multiple bytes
        (6, 18),  # Spans multiple full bytes in middle
        (17, 21),  # Last partial byte
    ],
)
def test_unpack_nucleic_slice_no_ambiguity(seq_no_ambiguity_long, start, stop):
    # Test slicing for sequences without ambiguity characters
    sarr, packed, positions, length = seq_no_ambiguity_long

    unpacked = unpack_nucleic(packed, positions, length, start=start, stop=stop)

    # Compare with direct slicing of original array
    expected = sarr[start:stop]
    assert np.array_equal(unpacked, expected)


# seq_with_ambiguity fixture creates "ACGNNGTYAGGTT" (length 13)
# Canonical bases at: 0,1,2 (ACG), 5,6 (GT), 8,9,10,11,12 (AGGTT) -> 10 packed bases
# Ambiguities at: 3,4 (NN), 7 (Y) -> 3 ambiguous chars
@pytest.mark.parametrize(
    ("start", "stop"),
    [
        # (None, None),  # Full sequence
        # (0, 13),  # Full sequence explicit
        # (0, 3),  # Before any ambiguity (ACG)
        # (0, 5),  # Through first ambiguity (ACGNN)
        (3, 5),  # Just the NN ambiguity
        (3, 8),  # From first ambig through second (NNGTY)
        (5, 13),  # After first ambiguity to end
        (7, 8),  # Just Y ambiguity
        (8, 13),  # After all ambiguities (AGGTT)
        (2, 10),  # Middle slice crossing both ambiguities
        (0, 1),  # Single canonical at start
        (3, 4),  # Single ambiguous (first N)
        (12, 13),  # Single canonical at end
        (0, 0),  # Empty slice
        (6, 6),  # Empty slice in middle
    ],
)
def test_unpack_nucleic_slice_with_ambiguity(seq_with_ambiguity, start, stop):
    # Test slicing for sequences with ambiguity characters
    sarr, packed, positions, length = seq_with_ambiguity

    unpacked = unpack_nucleic(packed, positions, length, start=start, stop=stop)

    expected = sarr[start:stop]
    assert np.array_equal(unpacked, expected)
