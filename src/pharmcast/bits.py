"""Bit-level view of a pharmacophore fingerprint.

**A fingerprint is 10,549 pharmacophores.** It is *stored* in 330 unsigned
32-bit words. A word count is storage and is never a fingerprint width; eleven
of the packed positions carry no pharmacophore at all.

`unpack` returns one entry per pharmacophore, with **pharmacophore j at index
j**, so a set bit is traceable to a specific typed triangle. `pack` is its exact
inverse.

THE ORDERING RULE, which is what makes this non-obvious.

`pfpkey.c` stores pharmacophore *i* as::

    fingerprint[i / 32] |= 1 << (31 - i % 32);

so within each word the pharmacophores run from the most significant bit *down*.
Unpacking a little-endian word LSB-first reverses them inside the word, putting
pharmacophore *i* at packed position ``(i//32)*32 + 31 - (i%32)``. Word 329
therefore carries pharmacophores 10528 to 10548 in its **top** 21 bits, which
means:

* the eleven positions carrying no pharmacophore are **10528 to 10538**, not the
  last eleven;
* packed positions **10539 to 10559 are real pharmacophores**.

**A ``[:10549]`` slice is therefore wrong**: it keeps the eleven meaningless
positions and discards eleven real pharmacophores. Real records do set them:
pharmacophores 10529 and 10531 occur in the ChEMBL corpus. `PACK_POS` is the
mapping and must be used instead of any slice. Pinned by
`tests/test_fingerprint_width.py`; see `PHARMCAST_FINGERPRINT_WIDTH.md`.
"""
from __future__ import annotations

import numpy as np

N_INTS = 330
N_BITS = N_INTS * 32     # packed word slots, storage only
N_PHARM = 10549          # the fingerprint width

# pfpkey.c stores pharmacophore i with fingerprint[i/32] |= 1 << (31 - i%32),
# so within each word the pharmacophores run from the most significant bit
# down. Unpacking a little endian word LSB first reverses them inside the word,
# which means the 11 positions carrying no pharmacophore are 10528 to 10538,
# not the top 11. PACK_POS gives the packed position of each pharmacophore.
_J = np.arange(N_PHARM)
PACK_POS = ((_J // 32) * 32 + 31 - (_J % 32)).astype(np.int64)



def unpack(words):
    """330 integers -> bool array of N_PHARM bits, pharmacophore j at index j."""
    w = np.asarray(words, dtype="<u4")
    return np.unpackbits(w.view(np.uint8),
                         bitorder="little")[PACK_POS].astype(bool)


def pack(bits):
    """Bool array of up to N_PHARM pharmacophore bits -> list of 330 integers.

    The exact inverse of `unpack`. Only positions that carry a pharmacophore
    are ever written, so a caller cannot set a bit that means nothing.
    """
    a = np.asarray(bits, dtype=bool).ravel()
    if a.size > N_PHARM:
        raise ValueError(
            "expected at most %d pharmacophore bits, got %d. A 10,560-long "
            "array is packed word slots, not a fingerprint; map it through "
            "PACK_POS first." % (N_PHARM, a.size))
    v = np.zeros(N_PHARM, dtype=bool)
    v[:a.size] = a
    b = np.zeros(N_BITS, dtype=bool)
    b[PACK_POS] = v
    return [int(v) for v in np.packbits(b, bitorder="little").view("<u4")]


def native_index(i):
    """Convert between a packed position and a pharmacophore number.

    ``i -> 32 * (i // 32) + 31 - (i % 32)``: the word is unchanged and the
    offset within the word is mirrored. The mapping is its own inverse, so one
    function converts in both directions.

    **You almost certainly do not need this.** `unpack` and `set_bits` already
    return pharmacophore numbers, so applying `native_index` to their output
    mirrors a second time and produces nonsense. It is exported only for code
    that has a raw packed position in hand.
    """
    i = np.asarray(i)
    return 32 * (i // 32) + 31 - (i % 32)


def set_bits(words):
    """-> sorted array of the PHARMACOPHORE NUMBERS that are on.

    Not packed positions. Column j is pharmacophore j, which is what makes a
    set bit traceable back to a specific typed triangle.
    """
    return np.flatnonzero(unpack(words))


def popcount(words):
    """-> number of set bits. The cheap route; no unpacking."""
    return int(sum(bin(int(v)).count("1") for v in words))


def to_bitstring(words, width=None):
    """-> a '0101...' string, one character per pharmacophore slot.

    `width` truncates to the first N slots, which is the only thing that makes
    a 10,549-character string readable in a terminal. It is a display control
    and nothing computed from it should be reported as a property of the whole
    fingerprint.
    """
    b = unpack(words)
    if width is not None:
        b = b[:width]
    return "".join("1" if v else "0" for v in b)


def from_bitstring(text):
    """'0101...' -> list of 330 integers. The inverse of `to_bitstring` at
    full width."""
    return pack(np.frombuffer(text.encode(), dtype=np.uint8) == ord("1"))
