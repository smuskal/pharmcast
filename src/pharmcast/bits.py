"""Bit-level view of a pharmacophore fingerprint.

A fingerprint is carried as 330 unsigned 32-bit words -- 10,560 slots, of which
the first 10,549 are real pharmacophores and the remainder is padding that is
never set. These helpers expand that packing into the 0/1 form people actually
want to look at, and back again.

THE BIT CONVENTION, which has caused confusion and must not be "fixed".

There are two orderings in play and they are both correct in their own frame:

  native     `pfpall` places pharmacophore *i* at word ``i // 32``, bit
             ``31 - (i % 32)`` counting from the least significant end.
  packed     this library unpacks the same words little-endian without a
             byteswap, which lands pharmacophore *i* at a different position.

Everything here is self-consistent because the same convention is used on both
sides: `unpack` is the exact inverse of `pack`, and `pack` reproduces the
native word representation byte for byte. Only when you need to talk about a
specific *pharmacophore number* -- to line a bit up against `pfpall`'s own
numbering -- does the difference matter, and `native_index` converts.

Do not change one side to match the other. The round trip is verified by test.
"""
from __future__ import annotations

import numpy as np

N_INTS = 330
N_BITS = N_INTS * 32     # 10,560 slots
N_PHARM = 10549          # real pharmacophores; the rest is padding


def unpack(words):
    """330 integers -> bool array of N_PHARM pharmacophore bits (packed order)."""
    w = np.asarray(words, dtype=np.uint32)
    return np.unpackbits(w.view(np.uint8),
                         bitorder="little")[:N_PHARM].astype(bool)


def pack(bits):
    """Bool array of up to N_BITS bits -> list of 330 integers.

    The exact inverse of `unpack`. Padding slots are forced off, so a caller
    cannot accidentally set a bit that no pharmacophore corresponds to.
    """
    b = np.zeros(N_BITS, dtype=bool)
    b[:len(bits)] = np.asarray(bits, dtype=bool)[:N_BITS]
    b[N_PHARM:] = False
    return [int(v) for v in np.packbits(b, bitorder="little").view(np.uint32)]


def native_index(i):
    """Convert between packed position and `pfpall` pharmacophore number.

    ``i -> 32 * (i // 32) + 31 - (i % 32)``: the word is unchanged and the
    offset within the word is mirrored. The mapping is its own inverse, so the
    one function converts in both directions.
    """
    i = np.asarray(i)
    return 32 * (i // 32) + 31 - (i % 32)


def set_bits(words):
    """-> sorted array of the packed positions that are on."""
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
