"""PharmSim / PharmTan: pharmacophoric similarity between fingerprints.

`pharmtan` is the Tanimoto coefficient over the set bits of two ensemble
fingerprints -- the same number the native comparison tools produce, computed
here so callers need not shell out.

`pharmsim` is the same comparison with its working shown: the shared, only-in-A
and only-in-B bit sets, which is what you look at when you want to know *why*
two molecules scored the way they did rather than merely that they did.

A note on interpreting the number. These are ensemble fingerprints: each is the
bitwise OR over a conformer ensemble, so a bit means "this molecule can present
this pharmacophore triplet in some accessible conformation", not "it does".
Values are therefore lower than 2D fingerprint Tanimotos on the same pairs and
must not be compared against them on a shared scale. Typical set-bit counts are
around 530 for catalogue-sized compounds and around 2,000 for peptides, so a
large size mismatch depresses the coefficient on its own.
"""
from __future__ import annotations

import numpy as np

from .bits import N_BITS, N_PHARM, unpack

# np.bitwise_count landed in NumPy 2.0. Plenty of working environments are
# still pinned to NumPy 1.x by some other dependency, and this library has no
# other reason to demand 2.x, so fall back to a 256-entry lookup table rather
# than raising AttributeError deep inside a screen. The table is the standard
# byte-popcount trick and gives identical results.
if hasattr(np, "bitwise_count"):
    _popcount = np.bitwise_count
else:                                                    # pragma: no cover
    _POPCOUNT_TABLE = np.array(
        [bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(a):
        return _POPCOUNT_TABLE[a]


def pharmtan(a, b):
    """Tanimoto between two 330-integer fingerprints."""
    inter = sum(bin(int(x) & int(y)).count("1") for x, y in zip(a, b))
    na = sum(bin(int(x)).count("1") for x in a)
    nb = sum(bin(int(y)).count("1") for y in b)
    return inter / max(na + nb - inter, 1e-9)


# Kept as an alias: the coefficient has been called both things in this
# codebase's history and in the native tools, and they are one function.
tanimoto = pharmtan


def _matrix(fps):
    """-> uint8 array [n, N_BITS/8], the packed bytes of each fingerprint."""
    # '<u4' rather than np.uint32: the byte order of the packed view must not
    # depend on the host's endianness, or two machines would disagree about
    # what the same fingerprint is. Every mainstream platform is little-endian,
    # which is exactly why this would go unnoticed until it did not.
    w = np.asarray(list(fps), dtype="<u4")
    return w.view(np.uint8).reshape(len(w), -1)


def pharmtan_matrix(queries, targets=None):
    """All-against-all Tanimoto -> float array [n_queries, n_targets].

    Vectorised over packed bytes with a popcount, which is roughly two
    orders of magnitude faster than looping `pharmtan`. This is the routine to
    use for a screen; `pharmtan` is for one pair.

    With `targets` omitted the queries are compared against themselves, giving
    a symmetric matrix with 1.0 on the diagonal.
    """
    q = _matrix(queries)
    t = q if targets is None else _matrix(targets)
    nq = _popcount(q).sum(axis=1).astype(np.float64)
    nt = _popcount(t).sum(axis=1).astype(np.float64)
    out = np.empty((len(q), len(t)), dtype=np.float64)
    for i in range(len(q)):
        inter = _popcount(q[i] & t).sum(axis=1).astype(np.float64)
        out[i] = inter / np.maximum(nq[i] + nt - inter, 1e-9)
    return out


def pharmsim(a, b):
    """Compare two fingerprints and show the working.

    -> dict with the coefficient, the three set sizes, and the actual bit
    positions of the shared and exclusive sets in packed order. Positions are
    what you feed to `native_index` if you need `pfpall`'s own pharmacophore
    numbering.
    """
    ba, bb = unpack(a), unpack(b)
    shared = ba & bb
    only_a = ba & ~bb
    only_b = bb & ~ba
    union = int(shared.sum()) + int(only_a.sum()) + int(only_b.sum())
    return {
        "tanimoto": int(shared.sum()) / max(union, 1e-9),
        "bits_a": int(ba.sum()),
        "bits_b": int(bb.sum()),
        "shared": int(shared.sum()),
        "only_a": int(only_a.sum()),
        "only_b": int(only_b.sum()),
        "union": union,
        "shared_positions": np.flatnonzero(shared),
        "only_a_positions": np.flatnonzero(only_a),
        "only_b_positions": np.flatnonzero(only_b),
    }
