"""Guards for the two things that would silently differ between machines.

Neither of these fails on the developer's own laptop, which is exactly why they
need a test: an Apple Silicon Mac, an Intel PC and an x86_64 Linux box are all
little-endian and all ship NumPy 2.x, so the bugs only surface on someone
else's install.
"""
import numpy as np
import pytest

from pharmcast import bits as B
from pharmcast import pharmtan, pharmtan_matrix
from pharmcast import similarity as S


def test_packing_is_little_endian_regardless_of_host():
    """`pack` must produce the same integers on any machine.

    Bit 0 set means the low bit of the first byte, which in little-endian word
    order is the low bit of word 0 -- value 1, never 0x01000000.
    """
    b = np.zeros(B.N_PHARM, dtype=bool)
    b[0] = True
    assert B.pack(b)[0] == 1
    b[:] = False
    b[31] = True
    assert B.pack(b)[0] == 1 << 31


def test_word_dtype_is_explicitly_little_endian():
    """A native-order view would be a latent cross-platform bug."""
    import re
    for path in ("bits.py", "model.py", "similarity.py"):
        src = open(__file__.replace("tests/test_portability.py",
                                    "src/pharmcast/" + path)).read()
        assert not re.search(r"\.view\(np\.uint32\)", src), path


def test_popcount_fallback_matches_numpy():
    """The NumPy 1.x fallback must agree with np.bitwise_count exactly."""
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    a = np.arange(256, dtype=np.uint8)
    assert np.array_equal(table[a], np.vectorize(lambda x: bin(x).count("1"))(a))
    if hasattr(np, "bitwise_count"):
        assert np.array_equal(table[a], np.bitwise_count(a))


def test_matrix_agrees_with_pairwise_under_the_fallback(monkeypatch):
    """Force the lookup-table path and prove it gives identical answers."""
    rng = np.random.default_rng(7)
    fps = [B.pack(rng.random(B.N_PHARM) < 0.05) for _ in range(4)]
    real = pharmtan_matrix(fps)

    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    monkeypatch.setattr(S, "_popcount", lambda a: table[a])
    fallback = pharmtan_matrix(fps)

    assert np.allclose(real, fallback)
    for i in range(4):
        for j in range(4):
            assert abs(fallback[i, j] - pharmtan(fps[i], fps[j])) < 1e-9


def test_no_absolute_developer_paths_in_shipped_code():
    """The working-tree copy hardcoded /Users/smuskal/...; the release must not."""
    import os
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "pharmcast")
    for fn in os.listdir(src):
        if fn.endswith(".py"):
            body = open(os.path.join(src, fn)).read()
            assert "/Users/" not in body, fn
            assert "\\\\Users\\\\" not in body, fn
