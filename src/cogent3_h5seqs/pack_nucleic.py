import numba
import numpy as np


@numba.jit(nopython=True, inline="always")
def count_bases_non_canonical_runs(seq) -> tuple[int, int]:
    """Counts number of bases and runs of non-canonical characters

    Parameters
    ----------
    seq
        a cogent3 encoded numpy u8 array, >= 4 are non-canonical characters

    Returns
    -------
    counts of canonical, non-canonical characters
    """
    count = 0
    num_bases = 0
    for i, base in enumerate(seq):
        if base < 4:
            num_bases += 1
            continue

        if i and base == seq[i - 1]:
            continue
        count += 1

    return num_bases, count


# @numba.jit(nopython=True)
def pack_nucleic(seq):
    """decomposes nucleic acid sequence

    Parameters
    ----------
    seq
        a cogent3 encoded numpy u8 array, >= 4 are non-canonical characters

    Notes
    -----
    packs canonical characters as 2-bit values (4 per byte) and records
    non-canonical (ambiguity code) characters

    Returns
    -------
    2-bit encoded canonical bases
    array with columns [stripped_index, cumsum, char_code] for each
    ambiguous character
    the sequence length (number of canonical bases)
    """
    num_canon, num_ambig = count_bases_non_canonical_runs(seq)

    # pack 4 bases per byte
    num_packed_bytes = (num_canon + 3) // 4
    packed = np.zeros(num_packed_bytes, dtype=np.uint8)
    # packed_position, cumulative_count, char_code
    ambig_posns = np.zeros((num_ambig, 3), dtype=np.int64)

    canon_idx = 0
    ambig_idx = -1
    byte_idx = 0
    bit_offset = 0
    cumsum_ambig = 0
    last = 255

    for i in range(len(seq)):
        base = seq[i]
        if base < 4:
            # we have a canonical base
            packed[byte_idx] |= np.uint8(base << bit_offset)
            canon_idx += 1
            bit_offset += 2
            if bit_offset == 8:
                bit_offset = 0
                byte_idx += 1
            last = base
            continue

        # we have an ambiguity code
        cumsum_ambig += 1
        if base != last:
            ambig_idx += 1

        ambig_posns[ambig_idx, 0] = canon_idx  # packed position before this ambig
        ambig_posns[ambig_idx, 1] = cumsum_ambig
        ambig_posns[ambig_idx, 2] = base
        last = base

    return packed, ambig_posns, num_canon


@numba.jit(nopython=True)
def stripped_to_natural_pos(ambig_positions, pos):
    """Map a packed (canonical character) sequence `pos` back to the original sequence.

    `ambig_positions` has columns [packed_pos, cumulative_count, char_code].
    We need to count how many ambiguous characters have packed_pos <= pos,
    then add that count to `pos`.
    """
    if ambig_positions.size == 0 or pos < ambig_positions[0, 0]:
        return pos

    if pos >= ambig_positions[-1, 0]:
        return pos + int(ambig_positions[-1, 1])

    # Extract the sorted packed positions and find the rightmost one <= pos
    n = len(ambig_positions)
    starts = np.empty(n, dtype=np.int64)
    for i in range(n):
        starts[i] = int(ambig_positions[i, 0])

    # Use searchsorted with side='right' to find insertion point
    # (equivalent to bisect_right)
    i = np.searchsorted(starts, pos, side="right") - 1

    return pos + int(ambig_positions[i, 1])


def unpack_packed(packed, length):
    """Decode 2-bit packed canonical bases into their original indices.

    Parameters
    ----------
    packed
        numpy uint8 array, each byte contains four 2-bit encoded bases
    length
        number of canonical bases to recover (may be < len(packed) * 4)
    """

    if length == 0:
        return np.zeros(0, dtype=np.uint8)

    result = np.zeros(length, dtype=np.uint8)
    for idx in range(length):
        byte_idx = idx // 4
        bit_offset = (idx % 4) * 2
        result[idx] = (packed[byte_idx] >> bit_offset) & np.uint8(3)

    return result


def unpack_packed_slice(packed, start, stop):
    """Decode 2-bit packed canonical bases for a slice [start:stop].

    Parameters
    ----------
    packed
        numpy uint8 array, each byte contains four 2-bit encoded bases
    start
        start index in the unpacked sequence
    stop
        stop index (exclusive) in the unpacked sequence

    Returns
    -------
    numpy uint8 array of decoded bases for the slice
    """
    length = stop - start
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)

    result = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        idx = start + i
        byte_idx = idx // 4
        bit_offset = (idx % 4) * 2
        result[i] = (packed[byte_idx] >> bit_offset) & np.uint8(3)

    return result


# @numba.jit(nopython=True)
def compose_seq(stripped, ambig_posns, total_len):
    """composes a sequence from its canonical and ambiguous bases."""
    result = np.zeros(total_len, dtype=np.uint8)

    # inject the canonical bases
    for stripped_pos in range(stripped.size):
        nat_pos = stripped_to_natural_pos(ambig_posns, stripped_pos)
        result[nat_pos] = stripped[stripped_pos]

    # inject ambiguity codes
    last = 0
    for stripped_pos, cumsum, char_code in ambig_posns:
        run_length = cumsum - last
        for j in range(stripped_pos + last, stripped_pos + last + run_length):
            result[j] = char_code
        last = cumsum

    return result


def unpack_nucleic(
    packed,
    ambig_posns,
    num_canon,
    start=None,
    stop=None,
):
    """Reverses 2-bit encoding to reconstruct the original sequence or a slice.

    Parameters
    ----------
    packed
        2-bit encoded nucleotides (4 bases per uint8 byte)
    ambig_posns
        array with columns [stripped_index, cumsum, char_code] for each
        ambiguous character
    num_canon
        number of canonical bases (the sequence length after stripping
        ambiguities)
    start
        start index in natural (original) sequence coordinates, defaults to 0
    stop
        stop index (exclusive) in natural sequence coordinates, defaults to
        sequence length

    Returns
    -------
    fully reconstructed original sequence (or slice) with actual ambiguity codes
    """
    total_ambig = int(ambig_posns[-1, 1]) if ambig_posns.size > 0 else 0
    total_len = num_canon + total_ambig

    if start is None:
        start = 0
    if stop is None:
        stop = total_len

    # For no ambiguity case, natural coords == packed coords
    if ambig_posns.size == 0:
        return unpack_packed_slice(packed, start, stop)

    # For sequences with ambiguities, fall back to full unpack + slice
    stripped = unpack_packed(packed, num_canon)
    full_seq = compose_seq(stripped, ambig_posns, total_len)
    return full_seq[start:stop]
