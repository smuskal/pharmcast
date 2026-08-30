"""The streaming screen must agree with the pairwise coefficient, exactly.

`screen` exists because `pharmtan_matrix` cannot hold a four-million-target
matrix. That is a performance change and it must not be a numerical one: the
whole value of the command is that its ranking is the ranking `pharmtan` would
have produced one pair at a time.

These fixtures are synthetic bit patterns rather than model output, so the
tests need no checkpoint and run anywhere.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

from pharmcast import pharmtan
from pharmcast.bits import N_INTS
from pharmcast.screen import screen, write_tsv


def _fp(seed, density=0.05):
    """A deterministic pseudo-random fingerprint as N_INTS packed words."""
    rng = np.random.default_rng(seed)
    bits = rng.random(N_INTS * 32) < density
    packed = np.packbits(bits, bitorder="little").view("<u4")
    return [int(v) for v in packed]


def _corpus(n, start=0):
    return [("mol%04d" % i, _fp(i), "C" * (1 + i % 5)) for i in range(start, start + n)]


def test_streaming_matches_pairwise_exactly():
    queries = _corpus(4)
    targets = _corpus(60, start=100)

    got = screen(queries, iter(list(targets)), top=None, cutoff=0.0)

    for qi, (_qn, qw, _qs) in enumerate(queries):
        want = {t[0]: pharmtan(qw, t[1]) for t in targets}
        assert len(got[qi]) == len(targets)
        for score, name, _smiles, _bits in got[qi]:
            # Exactly equal, not approximately: same popcount, same words.
            assert score == pytest.approx(want[name], abs=0.0, rel=0.0), name


def test_top_n_is_the_best_n_and_is_ordered():
    queries = _corpus(3)
    targets = _corpus(200, start=1000)

    got = screen(queries, iter(list(targets)), top=5)

    for qi, (_qn, qw, _qs) in enumerate(queries):
        rows = got[qi]
        assert len(rows) == 5
        scores = [r[0] for r in rows]
        assert scores == sorted(scores, reverse=True)
        best = sorted((pharmtan(qw, t[1]) for t in targets), reverse=True)[:5]
        assert scores == pytest.approx(best, abs=0.0, rel=0.0)


def test_cutoff_and_top_agree_on_shared_rows():
    """A cutoff run and a top-N run must not disagree about what they share."""
    queries = _corpus(3)
    targets = _corpus(300, start=2000)

    every = screen(queries, iter(list(targets)), top=None, cutoff=0.0)
    for qi in range(len(queries)):
        # A cutoff that admits at least 10 hits, so the comparison is real.
        scores = sorted((r[0] for r in every[qi]), reverse=True)
        cut = scores[9]

        by_cut = screen([queries[qi]], iter(list(targets)), top=None, cutoff=cut)
        by_top = screen([queries[qi]], iter(list(targets)), top=10, cutoff=cut)

        cut_map = {r[1]: r[0] for r in by_cut[0]}
        assert len(by_top[0]) <= 10
        for score, name, _s, _b in by_top[0]:
            assert name in cut_map, "top-N returned a hit the cutoff run missed"
            assert score == cut_map[name]


def test_chunking_does_not_change_the_answer():
    queries = _corpus(2)
    targets = _corpus(150, start=3000)
    a = screen(queries, iter(list(targets)), top=20, chunk=7)
    b = screen(queries, iter(list(targets)), top=20, chunk=1000)
    for qi in range(len(queries)):
        assert [(r[0], r[1]) for r in a[qi]] == [(r[0], r[1]) for r in b[qi]]


def test_self_screen_excludes_only_itself():
    queries = _corpus(6)
    got = screen(queries, iter(list(queries)), top=None, cutoff=0.0,
                 exclude_self=True)
    for qi, (name, _w, _s) in enumerate(queries):
        names = [r[1] for r in got[qi]]
        assert name not in names
        assert len(names) == len(queries) - 1


def test_ties_are_broken_deterministically():
    """Identical fingerprints all score 1.0; the order must still be stable."""
    same = _fp(42)
    queries = [("q", same, "C")]
    targets = [("t%02d" % i, list(same), "C") for i in range(10)]
    a = screen(queries, iter(list(targets)), top=None, cutoff=0.0)
    b = screen(queries, iter(list(reversed(targets))), top=None, cutoff=0.0)
    assert [r[1] for r in a[0]] == [r[1] for r in b[0]]
    assert [r[1] for r in a[0]] == sorted(r[1] for r in a[0])


def test_tsv_has_a_header_and_full_identifiers():
    """The original truncated names to 20 characters. This one must not."""
    long_name = "a-deliberately-long-identifier-well-past-twenty-characters"
    long_smiles = "C" * 120
    queries = [(long_name, _fp(7), long_smiles)]
    targets = [(long_name + "-target", _fp(8), long_smiles)]
    got = screen(queries, iter(list(targets)), top=1)

    buf = io.StringIO()
    write_tsv(buf, queries, got)
    lines = buf.getvalue().splitlines()

    assert lines[0].split("\t") == [
        "query_name", "query_smiles", "rank", "target_name", "target_smiles",
        "pharmtan", "query_bits", "target_bits"]
    row = lines[1].split("\t")
    assert row[0] == long_name
    assert row[1] == long_smiles
    assert row[3] == long_name + "-target"
    assert row[2] == "1"
    assert len(row[5].split(".")[1]) == 3          # three decimal places


# ---------------------------------------------------------------------------
# Provenance independence.
#
# A .pfp is a FORMAT, not a producer. Fingerprints from the reference
# calculation and fingerprints predicted by PharmCast are equally valid input,
# and all four combinations of query and target origin must work -- with no
# model loaded, because a .pfp already holds fingerprints.
# ---------------------------------------------------------------------------
import subprocess
import sys as _sys

from pharmcast.io import write_pfp
from pharmcast.screen import Provenance, iter_fingerprints, warn_on_conformer_mismatch


def _write(tmp_path, name, records, nconf):
    p = tmp_path / name
    write_pfp(str(p), records, nconf=nconf)
    return p


@pytest.mark.parametrize("q_conf,t_conf", [(1, 1), (100, 100), (1, 100), (100, 1)])
def test_all_four_provenance_combinations_need_no_model(tmp_path, q_conf, t_conf):
    """reference/reference, predicted/predicted, and both mixed directions."""
    q = _write(tmp_path, "q.pfp",
               [("q%d" % i, _fp(i), "C") for i in range(3)], nconf=q_conf)
    t = _write(tmp_path, "t.pfp",
               [("t%d" % i, _fp(100 + i), "CC") for i in range(20)], nconf=t_conf)

    # model=None throughout: nothing about a .pfp needs a checkpoint.
    queries = list(iter_fingerprints(q, model=None))
    targets = iter_fingerprints(t, model=None)
    got = screen(queries, targets, top=5)

    want = {name: dict((tn, pharmtan(w, tw))
                       for tn, tw, _ in iter_fingerprints(t, model=None))
            for name, w, _ in queries}
    for qi, (name, _w, _s) in enumerate(queries):
        assert len(got[qi]) == 5
        for score, t_name, _sm, _b in got[qi]:
            assert score == pytest.approx(want[name][t_name], abs=0.0, rel=0.0)


def test_cli_screens_two_pfp_files_with_no_model_argument(tmp_path):
    """The command itself must run without --model when both sides are .pfp."""
    q = _write(tmp_path, "q.pfp", [("q0", _fp(1), "C")], nconf=1)
    t = _write(tmp_path, "t.pfp",
               [("t%d" % i, _fp(50 + i), "CC") for i in range(10)], nconf=1)

    r = subprocess.run(
        [_sys.executable, "-m", "pharmcast.cli", "screen",
         "--queries", str(q), "--targets", str(t), "--top", "3"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.splitlines() if not l.startswith("#")]
    assert lines[0].split("\t")[0] == "query_name"
    assert len(lines) == 4                      # header + top 3

    # and the scores are the pairwise ones
    qw = next(iter_fingerprints(q, model=None))[1]
    by_name = {n: w for n, w, _ in iter_fingerprints(t, model=None)}
    for row in lines[1:]:
        f = row.split("\t")
        assert float(f[5]) == pytest.approx(pharmtan(qw, by_name[f[3]]), abs=5e-4)


def test_output_header_names_both_inputs(tmp_path):
    q = _write(tmp_path, "real.pfp", [("q0", _fp(1), "C")], nconf=1)
    t = _write(tmp_path, "pred.pfp", [("t0", _fp(2), "CC")], nconf=100)
    q_prov, t_prov = Provenance(q), Provenance(t)
    queries = list(iter_fingerprints(q, provenance=q_prov))
    targets = list(iter_fingerprints(t, provenance=t_prov))
    got = screen(queries, iter(targets), top=1)

    buf = io.StringIO()
    write_tsv(buf, queries, got, q_prov=q_prov, t_prov=t_prov)
    head = buf.getvalue().splitlines()
    assert head[0].startswith("# queries:") and "real.pfp" in head[0]
    assert head[1].startswith("# targets:") and "pred.pfp" in head[1]
    assert "nconf 1" in head[0] and "nconf 100" in head[1]


def test_conformer_mismatch_warns_but_does_not_stop(tmp_path):
    q_prov, t_prov = Provenance("q.pfp"), Provenance("t.pfp")
    q_prov["nconf"], t_prov["nconf"] = {1}, {100}
    buf = io.StringIO()
    assert warn_on_conformer_mismatch(q_prov, t_prov, stream=buf) is True
    assert "nconf 1" in buf.getvalue() and "nconf 100" in buf.getvalue()

    same = Provenance("s.pfp"); same["nconf"] = {100}
    buf2 = io.StringIO()
    assert warn_on_conformer_mismatch(same, same, stream=buf2) is False
    assert buf2.getvalue() == ""


def test_a_record_of_the_wrong_width_is_rejected(tmp_path):
    """Validate the format contract, never the producer."""
    bad = tmp_path / "bad.pfp"
    bad.write_text("mol " + " ".join(["0"] * (N_INTS - 1)) + "\n")
    with pytest.raises(SystemExit):
        list(iter_fingerprints(bad, model=None))
