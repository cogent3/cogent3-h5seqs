import numba
import numpy as np


@numba.jit(nopython=True, cache=True, inline="always")
def count_bases_non_canonical_runs(seq) -> tuple[int, int]:  # pragma: no cover
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


@numba.jit(nopython=True, cache=True, inline="always")
def pack_nucleic(seq):  # pragma: no cover
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


@numba.jit(nopython=True, cache=True, inline="always")
def decode_single_base(packed, packed_idx):  # pragma: no cover
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


@numba.jit(nopython=True, cache=True, inline="always")
def natural_to_packed_info(ambig_posns, nat_pos):  # pragma: no cover
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


@numba.jit(nopython=True, cache=True, inline="always")
def find_stripped_range(ambig_posns, start, stop, num_canon):  # pragma: no cover
    """Find the first and last stripped (canonical) indices in [start, stop).

    Parameters
    ----------
    ambig_posns
        array with columns [packed_pos, cumsum, char_code] for each ambiguous run
    start
        start index in natural sequence coordinates
    stop
        stop index (exclusive) in natural sequence coordinates
    num_canon
        number of canonical bases

    Returns
    -------
    tuple of (first_stripped, last_stripped), or (None, None) if no canonical
    bases exist in the range
    """
    if start >= stop or num_canon == 0:
        return None, None

    # No ambiguity - natural = stripped
    if ambig_posns.size == 0:
        first = start if start < num_canon else None
        last = min(stop - 1, num_canon - 1) if start < num_canon else None
        return first, last

    nat_ends = ambig_posns[:, 0] + ambig_posns[:, 1]

    # Find first stripped index
    idx_start = np.searchsorted(nat_ends, start, side="right")

    if idx_start == len(ambig_posns):
        # start is after all runs
        first_stripped = start - int(ambig_posns[-1, 1])
        if first_stripped >= num_canon:
            return None, None
    else:
        prev_cumsum_start = int(ambig_posns[idx_start - 1, 1]) if idx_start > 0 else 0
        nat_start_run = int(ambig_posns[idx_start, 0]) + prev_cumsum_start

        if start >= nat_start_run:
            # In ambiguity run - first canonical is at end of run
            if nat_ends[idx_start] >= stop:
                return None, None  # No canonical in range
            first_stripped = int(ambig_posns[idx_start, 0])
        else:
            # Canonical before run
            first_stripped = start - prev_cumsum_start

    # Find last stripped index
    pos = stop - 1
    idx_stop = np.searchsorted(nat_ends, pos, side="right")

    if idx_stop == len(ambig_posns):
        # pos is after all runs
        last_stripped = pos - int(ambig_posns[-1, 1])
    else:
        prev_cumsum_stop = int(ambig_posns[idx_stop - 1, 1]) if idx_stop > 0 else 0
        nat_start_run = int(ambig_posns[idx_stop, 0]) + prev_cumsum_stop

        if pos >= nat_start_run:
            # In ambiguity run - last canonical is before this run
            if nat_start_run <= start:
                return None, None  # No canonical before this run within range
            last_stripped = int(ambig_posns[idx_stop, 0]) - 1
        else:
            # Canonical before run
            last_stripped = pos - prev_cumsum_stop

    return first_stripped, last_stripped


@numba.jit(nopython=True, cache=True, inline="always")
def get_packed_byte_range(ambig_posns, start, stop, num_canon):  # pragma: no cover
    """Get the minimal byte range from packed array needed for a slice.

    Parameters
    ----------
    ambig_posns
        array with columns [packed_pos, cumsum, char_code] for each ambiguous run
    start
        start index in natural sequence coordinates
    stop
        stop index (exclusive) in natural sequence coordinates
    num_canon
        number of canonical bases

    Returns
    -------
    tuple of (start_byte, end_byte, packed_start) for slicing packed array
    """
    min_stripped, max_stripped = find_stripped_range(
        ambig_posns, start, stop, num_canon
    )

    if min_stripped is None or max_stripped is None:
        return 0, 0, 0  # No canonical bases in range

    start_byte = min_stripped // 4
    end_byte = max_stripped // 4 + 1
    packed_start = start_byte * 4

    return start_byte, end_byte, packed_start


@numba.jit(nopython=True, cache=True, inline="always")
def compose_seq_slice(
    packed, ambig_posns, start, stop, packed_start=0
):  # pragma: no cover
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
    packed_start
        offset to adjust packed indices (used when packed is a slice)

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
        if is_ambig:
            result[i] = char_code
        else:
            # Adjust packed_idx by the offset
            result[i] = decode_single_base(packed, packed_idx - packed_start)

    return result


# we do not numba.jit this function as packed will be a h5py dataset
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

    # Get minimal byte range needed
    start_byte, end_byte, packed_start = get_packed_byte_range(
        ambig_posns, start, stop, num_canon
    )

    # Extract minimal packed segment
    packed_slice = packed[start_byte:end_byte]

    return compose_seq_slice(packed_slice, ambig_posns, start, stop, packed_start)
