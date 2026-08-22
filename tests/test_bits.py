"""The bit convention is the part most likely to be broken by a well-meaning
edit, so it is pinned here. See src/pharmcast/bits.py for why there are two
orderings and why neither may be "fixed" to match the other."""
import numpy as np
import pytest

from pharmcast import bits as B


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    b = rng.random(B.N_PHARM) < 0.05
    assert np.array_equal(B.unpack(B.pack(b)), b)


def test_pack_returns_330_words():
    w = B.pack(np.zeros(B.N_PHARM, dtype=bool))
    assert len(w) == B.N_INTS
    assert all(v == 0 for v in w)


def test_padding_slots_cannot_be_set():
    b = np.ones(B.N_BITS, dtype=bool)
    assert B.popcount(B.pack(b)) == B.N_PHARM


def test_native_index_is_its_own_inverse():
    i = np.arange(B.N_BITS)
    assert np.array_equal(B.native_index(B.native_index(i)), i)


def test_native_index_keeps_the_word_and_mirrors_the_offset():
    assert B.native_index(0) == 31
    assert B.native_index(31) == 0
    assert B.native_index(32) == 63
    assert all(B.native_index(i) // 32 == i // 32 for i in range(0, 400))


def test_popcount_agrees_with_unpack():
    rng = np.random.default_rng(1)
    w = B.pack(rng.random(B.N_PHARM) < 0.05)
    assert B.popcount(w) == int(B.unpack(w).sum()) == len(B.set_bits(w))


def test_bitstring_roundtrip():
    rng = np.random.default_rng(2)
    w = B.pack(rng.random(B.N_PHARM) < 0.05)
    assert B.from_bitstring(B.to_bitstring(w)) == w
