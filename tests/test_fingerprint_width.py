"""The fingerprint is 10,549 pharmacophores. 10,560 is a storage count.

These tests exist because a fix for exactly this confusion shipped a second bug:
`_bits` was changed to return pharmacophore-ordered columns without updating
`words_batch`, which then packed 10,549 bits into 1,319 bytes and could not view
them as uint32. `words_batch` raised on every call and no test caught it, because
nothing asserted the width of what it returns.

See PHARMCAST_FINGERPRINT_WIDTH.md for the contract these check.
"""
import numpy as np
import pytest

from pharmcast import bits as B


def test_fingerprint_width_is_the_pharmacophore_count():
    assert B.N_PHARM == 10549
    assert B.N_INTS == 330
    assert B.N_BITS == 10560          # storage only, never a fingerprint width


def test_unpack_returns_one_entry_per_pharmacophore():
    assert len(B.unpack([0] * B.N_INTS)) == B.N_PHARM


def test_the_eleven_dead_positions_are_10528_to_10538():
    """Not the top 11. This is the whole bug in one assertion."""
    live = set(int(v) for v in B.PACK_POS)
    dead = sorted(set(range(B.N_BITS)) - live)
    assert dead == list(range(10528, 10539))


def test_pack_never_writes_a_dead_position():
    w = B.pack(np.ones(B.N_PHARM, dtype=bool))
    b = np.unpackbits(np.asarray(w, dtype="<u4").view(np.uint8), bitorder="little")
    assert b[10528:10539].sum() == 0
    assert b.sum() == B.N_PHARM


def test_top_pharmacophores_survive_a_round_trip():
    """Pharmacophores 10528-10548 live at packed 10559 down to 10539.

    A `[:10549]` slice silently drops exactly these. Real pfpall records do set
    them, rarely: pharmacophores 10529 and 10531 appear in the ChEMBL corpus.
    """
    for j in (10528, 10529, 10531, 10540, 10548):
        v = np.zeros(B.N_PHARM, dtype=bool)
        v[j] = True
        w = B.pack(v)
        assert B.unpack(w)[j], f"pharmacophore {j} lost in the round trip"
        assert list(B.set_bits(w)) == [j]
        assert B.popcount(w) == 1


def test_pharmacophore_lands_where_pfpkey_puts_it():
    """pfpkey.c: fingerprint[i/32] |= 1 << (31 - i%32)."""
    for j in (0, 31, 32, 10529, 10531, 10548):
        v = np.zeros(B.N_PHARM, dtype=bool); v[j] = True
        w = np.asarray(B.pack(v), dtype="<u4")
        assert int(w[j // 32]) == (1 << (31 - j % 32)), f"pharmacophore {j} misplaced"


def test_pack_unpack_are_inverses_on_random_fingerprints():
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.random(B.N_PHARM) < 0.05
        assert np.array_equal(B.unpack(B.pack(v)), v)


def test_native_index_is_its_own_inverse():
    i = np.arange(B.N_BITS)
    assert np.array_equal(B.native_index(B.native_index(i)), i)


def test_no_source_file_slices_to_the_pharmacophore_count():
    """`[:N_PHARM]` on a packed array is the original defect. It must not return.

    Scans real code only: string literals and comments are tokenized away, so
    the docstrings that *describe* the bug do not trip the guard.
    """
    import io
    import os
    import re
    import tokenize

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "src", "pharmcast")
    pattern = re.compile(r"\[\s*:\s*(N_PHARM|10549)\s*\]"
                         r"|\[\s*:\s*,\s*:\s*(N_PHARM|10549)\s*\]")
    bad = []
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(src, fn)
        with open(path, "rb") as fh:
            toks = list(tokenize.tokenize(fh.readline))
        code_lines = {}
        for t in toks:
            if t.type in (tokenize.STRING, tokenize.COMMENT, tokenize.NL,
                          tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                continue
            code_lines.setdefault(t.start[0], []).append(t.string)
        for n, parts in code_lines.items():
            joined = " ".join(parts)
            if pattern.search(joined) or pattern.search(joined.replace(" ", "")):
                bad.append("%s:%d: %s" % (fn, n, joined))
    assert not bad, "packed-order truncation reintroduced:\n" + "\n".join(bad)
