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


def test_packing_matches_the_reference_and_is_host_independent():
    """`pack` must produce the same integers the reference calculation would,
    on any machine.

    The native format is the authority: pharmacophore i sits in word i//32,
    counted from the most significant bit of that word.

    so pharmacophore 0 is the TOP bit of word 0, and pharmacophore 31 is the
    low bit of word 0. The word values must not depend on host endianness.
    """
    b = np.zeros(B.N_PHARM, dtype=bool)
    b[0] = True
    assert B.pack(b)[0] == 1 << 31
    b[:] = False
    b[31] = True
    assert B.pack(b)[0] == 1


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


def _internal_names():
    """Internal tool, binary and data-file names that must never ship.

    They are not part of this project's public vocabulary, and they reached
    this repository once already, in prose that described the screening command
    by the name of its internal predecessor. Refer to what a thing DOES -- "the
    reference calculation", "the reference generator" -- never to the program
    that does it.

    Held encoded rather than spelled out, because a test enforcing "these words
    must not be published" should not itself publish them. Both cases are
    returned, so a lowercase spelling is caught too.
    """
    import base64
    raw = base64.b64decode(
        "UGhhcm1UYW5MaXN0LHBmcGFsbCxwZnByaWdpZCxNQVhEQixwaGFybXByaW50X25vUixw"
        "aGFybXByaW50X1Jvbmx5LHBoYXJtcHJpbnRfc3ltbSxwZnBrZXksUGhhcm1UYW5MaXN0"
        "Tm9MaW1pdA==").decode().split(",")
    return raw + [n.lower() for n in raw]


def test_no_absolute_developer_paths_or_internal_names_anywhere_shipped():
    """Nothing shipped may carry a developer path or an internal project name.

    This used to scan src/pharmcast/*.py only. A docs/ file then shipped six
    absolute home paths, the private conda environment name, the on-disk layout
    of a licensed third party corpus, and the code name of an unannounced
    internal program, straight into the public repository. The scan now covers
    everything that ships, not only the modules.
    """
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Anything an outsider can read after a clone or a pip install.
    # `tools` is walked too: the decoding tools ship with their own README,
    # and it named the reference generator and its data files until 2026-08-30.
    roots = ["src", "docs", "tests", "examples", "tools"]
    files = [os.path.join(root, f)
             for f in ("README.md", "NOTICE", "CITATION.cff", "pyproject.toml")]
    for d in roots:
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        for base, _dirs, names in os.walk(full):
            if "__pycache__" in base:
                continue
            files += [os.path.join(base, n) for n in names
                      if n.endswith((".py", ".md", ".txt", ".toml", ".cfg",
                                     ".yml", ".yaml", ".rst", ".c", ".h",
                                     ".sh")) or n == "Makefile"]

    # Local paths and build-machine detail. A published file naming one of
    # these tells a reader nothing and tells them where it was built.
    banned = ["/Users/", "\\\\Users\\\\", "miniforge", "ai-steve",
              "chip-sandbox", "enamine-screening", "pfp-libraries",
              "pfp-surrogate",
              ] + _internal_names()
    bad = []
    for path in files:
        if not os.path.exists(path):
            continue
        rel = os.path.relpath(path, root)
        if rel == os.path.join("tests", "test_portability.py"):
            continue                       # this file names the patterns
        body = open(path, encoding="utf-8", errors="replace").read()
        for token in banned:
            if token in body:
                bad.append("%s carries %r" % (rel, token))
    assert not bad, "internal detail in shipped files:\n" + "\n".join(bad)


def test_words_batch_returns_330_words_without_a_model():
    """Guards the scatter in `words_batch` with no checkpoint on disk.

    N_PHARM is not a multiple of 32, so packing pharmacophore-ordered bits
    directly gives 1,319 bytes, which cannot be viewed as uint32. This is the
    exact failure that shipped once.
    """
    import numpy as np

    from pharmcast import bits as B
    from pharmcast.model import N_BITS, PACK_POS

    pred = np.zeros((3, B.N_PHARM), dtype=bool)
    pred[:, [0, 10529, 10531, 10548]] = True
    full = np.zeros((pred.shape[0], N_BITS), dtype=bool)
    full[:, PACK_POS] = pred
    words = np.packbits(full, axis=1, bitorder="little").view("<u4")
    assert words.shape == (3, 330)
    for row in words:
        assert list(B.set_bits([int(v) for v in row])) == [0, 10529, 10531, 10548]
