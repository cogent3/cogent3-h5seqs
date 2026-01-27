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


@numba.jit(nopython=True, inline="always")
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


@numba.jit(nopython=True, inline="always")
def decode_single_base(packed, packed_idx):
    """Decode a single base from packed data at the given packed index.

    Parameters
    ----------
    packed
        numpy uint8 array of packed bases
    packed_idx
        index in the canonical (packed) sequence

    Returns
    -------
    The decoded base value (0-3)
    """
    byte_idx = packed_idx // 4
    bit_offset = (packed_idx % 4) * 2
    return (packed[byte_idx] >> bit_offset) & np.uint8(3)


@numba.jit(nopython=True, inline="always")
def natural_to_packed_info(ambig_posns, nat_pos):
    """Convert natural position to packed position info.

    Parameters
    ----------
    ambig_posns
        array with columns [packed_pos, cumsum, char_code] for each
        ambiguous run
    nat_pos
        position in natural (original) sequence coordinates

    Returns
    -------
    tuple of (packed_pos, is_ambiguous, char_code)
    - If canonical: (packed_index, False, 255)
    - If ambiguous: (255, True, char_code)
    """
    prev_cumsum = 0

    for i in range(len(ambig_posns)):
        packed_pos = ambig_posns[i, 0]
        cumsum = ambig_posns[i, 1]
        char_code = ambig_posns[i, 2]

        # Natural position where this ambiguity run starts
        nat_start = packed_pos + prev_cumsum
        # Length of this ambiguity run
        run_length = cumsum - prev_cumsum

        if nat_pos < nat_start:
            # Position is before this ambiguity run, so it's canonical
            return nat_pos - prev_cumsum, False, 255

        if nat_pos < nat_start + run_length:
            # Position falls within this ambiguity run
            return 255, True, char_code

        prev_cumsum = cumsum

    # After all ambiguity runs, position is canonical
    return nat_pos - prev_cumsum, False, 255


@numba.jit(nopython=True, inline="always")
def compose_seq_slice(packed, ambig_posns, start, stop):
    """Compose a slice [start:stop) directly from packed data.

    Parameters
    ----------
    packed
        numpy uint8 array of 2-bit packed canonical bases
    ambig_posns
        array with columns [packed_pos, cumsum, char_code] for each
        ambiguous run
    start
        start index in natural sequence coordinates
    stop
        stop index (exclusive) in natural sequence coordinates

    Returns
    -------
    numpy uint8 array of the slice
    """
    length = stop - start
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)

    result = np.zeros(length, dtype=np.uint8)

    for i in range(length):
        nat_pos = start + i
        packed_idx, is_ambig, char_code = natural_to_packed_info(ambig_posns, nat_pos)
        result[i] = char_code if is_ambig else decode_single_base(packed, packed_idx)

    return result


@numba.jit(nopython=True, inline="always")
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

    return compose_seq_slice(packed, ambig_posns, start, stop)
