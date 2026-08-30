"""Nearest-neighbour screening: many queries against a large target set.

This is a port of the original PharmPrint C tool, `PharmTanList.x`, which is the
reference for behaviour:

    PharmTanList.x cutoff qry.bits db.bits
    returns list of Tanimoto scores >= cutoff in db.bits
    using 1st fingerprint in qry.bits, ordered by score

Three deliberate differences, and nothing else:

* **Every** fingerprint in the query file is used, not only the first.
* Top-N as well as a cutoff, and the two compose.
* No fixed database cap. The original preallocates `MAXDB 100000` entries and
  aborts past it, which is why a `PharmTanListNoLimit.c` had to exist at all.

WHY THIS IS NOT `pharmtan_matrix`. That function is correct and vectorised and
stays the right tool for a few thousand fingerprints, but it materialises one
dense queries-by-targets matrix. At four million targets a single query row is
32 MB and the full matrix is out of reach, so a screen has to stream. Here the
targets pass through in chunks and only a bounded amount of state per query --
a top-N heap, or nothing at all when a bare cutoff is asked for -- is ever held.
Memory is therefore a function of the QUERY count and N, never of how many
targets there are.

EXACTNESS. The coefficient computed here is the same one `pharmtan` computes,
bit for bit: the same popcount over the same packed words, only vectorised
across a chunk. `tests/test_screen.py` asserts that against the pairwise path
rather than trusting this paragraph.

PROVENANCE INDEPENDENCE. A `.pfp` is a format, not a producer. Real `pfpall`
output, PharmCast predictions and files from the original PharmPrint C tools are
all valid input, and all four combinations of query and target origin are
legitimate -- including a real query screening a predicted library, which is how
you ask "would this screen have found the same thing". So nothing here infers or
requires a producer. What IS enforced is the format contract: exactly N_INTS
words per record, and a declared set-bit count that cannot exceed N_PHARM.

The one honest hazard in mixing them is the conformer count. A 100-conformer
ensemble compared against a single-conformer fingerprint scores lower for that
reason alone, and the descriptor block records `nconf`, so a mismatch is warned
about on stderr and the file names and counts are written into the output
header. Warned, not blocked: the comparison is meaningful, it just must not be
read as if both sides had the same conformational coverage.

Individual pharmacophores are never touched. A Tanimoto is a popcount over the
packed words, so `PACK_POS` does not enter into it -- and must not, because
introducing a per-pharmacophore step here could only make the result differ from
`pharmtan`. `PACK_POS` is for code that reports WHICH pharmacophore is set;
`pharmsim` is that code.
"""
from __future__ import annotations

import heapq
import itertools
import os
import sys
import time

import numpy as np

from .bits import N_INTS, N_PHARM
from .io import PFP_FIELDS, _open, read_smiles
from .similarity import _popcount

# Targets are read and scored this many at a time. Big enough that the
# vectorised popcount dominates the per-chunk overhead, small enough that one
# chunk of packed bytes is a couple of megabytes rather than a couple of
# gigabytes.
CHUNK = 20_000

COLUMNS = ("query_name", "query_smiles", "rank", "target_name",
           "target_smiles", "pharmtan", "query_bits", "target_bits")


def _packed(words_iterable):
    """-> uint8 array [n, N_INTS*4], the packed bytes of each fingerprint.

    '<u4' rather than np.uint32 for the same reason `similarity._matrix` does
    it: the byte order of the packed view must not depend on the host, or two
    machines disagree about what one fingerprint is.
    """
    w = np.asarray(list(words_iterable), dtype="<u4")
    if w.ndim == 1:
        w = w.reshape(1, -1)
    return w.view(np.uint8).reshape(len(w), -1)


class Provenance(dict):
    """What an input file turned out to be, filled in as it is streamed.

    Deliberately descriptive, never prescriptive: it records the format
    version and conformer counts a file declared so the output can say where
    each side came from. It is not used to accept or reject anything.
    """

    def __init__(self, path):
        super().__init__(path=os.fspath(path), records=0,
                         versions=set(), nconf=set(), kind="")

    def summary(self):
        def one(key):
            vals = sorted(v for v in self[key] if v not in (None, ""))
            return ",".join(str(v) for v in vals) if vals else "unknown"
        return ("%s (%s, %s records, version %s, nconf %s)"
                % (self["path"], self["kind"] or "unknown",
                   format(self["records"], ","), one("versions"), one("nconf")))


def iter_fingerprints(path, model=None, batch=2000, provenance=None):
    """Stream (name, words, smiles) from a `.pfp`/`.pfp.gz` or a `.smi`.

    A `.pfp` is read straight through and NEEDS NO MODEL, whatever produced it.
    A `.smi` is fingerprinted in batches as it is consumed, so a large SMILES
    file never has to be held in memory in full either.

    Pass a `Provenance` to have the file's declared format version and
    conformer counts collected while it streams.
    """
    p = os.fspath(path)
    if p.endswith(".pfp") or p.endswith(".pfp.gz"):
        if provenance is not None:
            provenance["kind"] = "native .pfp"
        # A STRICTER READER THAN `read_pfp_records`. That one skips any short
        # line, because these files are written incrementally by long jobs and
        # a half-written FINAL line is normal. That tolerance is right for it
        # and wrong here: a screen silently dropping records would report a
        # ranking over a corpus that is quietly not the one you asked for.
        #
        # So: a short line is an error, EXCEPT as the last line of the file,
        # which keeps the original tolerance exactly where it was earned.
        with _open(p) as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < N_INTS + 1:
                    raise SystemExit(
                        "%s: line %d carries %d words, not %d -- this is not a "
                        "%d-pharmacophore fingerprint record.\n"
                        "If this file is still being written by a running job, "
                        "the last line is half-written: screen a finished copy, "
                        "or drop the final line. A screen must not silently "
                        "skip records, or the ranking is over a corpus you did "
                        "not ask for."
                        % (p, lineno, len(parts) - 1, N_INTS, N_PHARM))
                name = parts[0]
                try:
                    words = [int(v) for v in parts[1:N_INTS + 1]]
                except ValueError:
                    raise SystemExit(
                        "%s: line %d does not parse as %d unsigned integers."
                        % (p, lineno, N_INTS))
                meta = {}
                for key, val in zip(PFP_FIELDS, parts[N_INTS + 1:]):
                    if key == "version":
                        meta[key] = val
                    else:
                        try:
                            meta[key] = (float(val) if key == "mw" else int(val))
                        except ValueError:
                            pass
                declared = meta.get("bits")
                if declared is not None and declared > N_PHARM:
                    raise SystemExit(
                        "%s: record %r declares %d set bits, more than the %d "
                        "pharmacophores this format defines."
                        % (p, name, declared, N_PHARM))
                if provenance is not None:
                    provenance["records"] += 1
                    provenance["versions"].add(meta.get("version"))
                    provenance["nconf"].add(meta.get("nconf"))
                yield name, words, ""
        return
    if provenance is not None:
        provenance["kind"] = "SMILES, fingerprinted by the model"

    if model is None:
        raise SystemExit(
            "%s is not a .pfp, so it has to be fingerprinted: pass --model" % p)
    pairs = read_smiles(p)
    while True:
        block = list(itertools.islice(pairs, batch))
        if not block:
            return
        words = model.words_batch([s for s, _ in block])
        for (smiles, name), w in zip(block, words):
            if provenance is not None:
                provenance["records"] += 1
                provenance["nconf"].add(100)      # what the model predicts
            yield name, w, smiles


def screen(queries, targets, top=10, cutoff=None, exclude_self=False,
           chunk=CHUNK, progress=None):
    """Screen queries against a stream of targets.

    `queries` is a materialised list of (name, words, smiles): there are few of
    them and each is compared against everything, so they are held. `targets`
    is any iterable of the same shape and is consumed once, in chunks.

    `top` bounds what is kept per query; `cutoff` filters on score. With both,
    the answer is the best `top` of those at or above `cutoff`. With `top=None`
    and a cutoff, every hit above the cutoff is kept -- which is the original
    tool's behaviour, and the only mode whose memory grows with the number of
    hits rather than with N.

    -> dict {query index: [(score, target_name, target_smiles, target_bits)]},
    each list sorted by descending score, ties broken by target name so the
    ordering is deterministic across runs and platforms.
    """
    q_names = [q[0] for q in queries]
    q_packed = _packed([q[1] for q in queries])
    q_bits = _popcount(q_packed).sum(axis=1).astype(np.int64)

    # A heap per query. heapq is a MIN-heap, so the smallest kept score sits at
    # [0] and is what a new hit has to beat -- which is exactly the test a
    # bounded top-N needs.
    heaps = [[] for _ in queries]
    kept_all = [[] for _ in queries]
    counter = itertools.count()          # tie-break inside the heap only
    compared = 0
    t0 = time.time()

    for block in iter(lambda: list(itertools.islice(targets, chunk)), []):
        t_names = [t[0] for t in block]
        t_smiles = [t[2] if len(t) > 2 else "" for t in block]
        t_packed = _packed([t[1] for t in block])
        t_bits = _popcount(t_packed).sum(axis=1).astype(np.int64)

        for qi in range(len(queries)):
            inter = _popcount(q_packed[qi] & t_packed).sum(axis=1).astype(np.int64)
            union = q_bits[qi] + t_bits - inter
            scores = inter / np.maximum(union, 1e-9)

            idx = np.arange(len(block))
            if cutoff is not None:
                idx = idx[scores[idx] >= cutoff]
            if exclude_self:
                # Identity by NAME, not by score: two distinct molecules can
                # legitimately score 1.0, and dropping them would be a lie.
                idx = idx[[t_names[i] != q_names[qi] for i in idx]]

            if top is None:
                for i in idx:
                    kept_all[qi].append(
                        (float(scores[i]), t_names[i], t_smiles[i], int(t_bits[i])))
                continue

            h = heaps[qi]
            for i in idx:
                s = float(scores[i])
                if len(h) < top:
                    heapq.heappush(
                        h, (s, next(counter), t_names[i], t_smiles[i], int(t_bits[i])))
                elif s > h[0][0]:
                    heapq.heapreplace(
                        h, (s, next(counter), t_names[i], t_smiles[i], int(t_bits[i])))

        compared += len(block) * len(queries)
        if progress:
            progress(compared, time.time() - t0)

    out = {}
    for qi in range(len(queries)):
        rows = ([(s, n, sm, b) for s, _c, n, sm, b in heaps[qi]]
                if top is not None else kept_all[qi])
        # Descending score, then target name ascending. The name tie-break is
        # what makes the output reproducible: scores are printed to three
        # decimal places and ties at that precision are common, so without it
        # two runs could order the same hits differently.
        rows.sort(key=lambda r: (-r[0], r[1]))
        out[qi] = rows
    out["_compared"] = compared
    out["_seconds"] = time.time() - t0
    return out


def warn_on_conformer_mismatch(q_prov, t_prov, stream=sys.stderr):
    """Warn once if the two sides declare different conformer counts.

    Not an error. An ensemble fingerprint over 100 conformers records what a
    molecule CAN present; a single-conformer fingerprint records what one pose
    does. Comparing them is legitimate and sometimes exactly the question --
    but the second scores lower for that reason alone, so the reader has to be
    told rather than left to wonder.
    """
    qn = {v for v in q_prov.get("nconf", ()) if v}
    tn = {v for v in t_prov.get("nconf", ()) if v}
    if qn and tn and qn != tn:
        print("warning: conformer counts differ -- queries declare nconf %s, "
              "targets declare nconf %s. Scores across a mismatch are depressed "
              "by the difference in conformational coverage alone."
              % (",".join(str(v) for v in sorted(qn)),
                 ",".join(str(v) for v in sorted(tn))), file=stream)
        return True
    return False


def write_tsv(fh, queries, results, q_prov=None, t_prov=None):
    """Write the ranked hits as TSV, with a provenance header and a header row.

    The provenance lines are comments. A screen that mixes predicted and real
    fingerprints is legitimate and common, and the output must never leave a
    reader guessing which side was which.
    """
    if q_prov is not None:
        print("# queries: %s" % q_prov.summary(), file=fh)
    if t_prov is not None:
        print("# targets: %s" % t_prov.summary(), file=fh)
    print("\t".join(COLUMNS), file=fh)
    q_packed = _packed([q[1] for q in queries])
    q_bits = _popcount(q_packed).sum(axis=1).astype(np.int64)
    for qi, (name, _w, smiles) in enumerate(queries):
        for rank, (score, t_name, t_smiles, t_bits) in enumerate(results[qi], 1):
            # Names and SMILES go out IN FULL. The original truncated names to
            # MAXLINE 20 characters; a clipped identifier is not an identifier
            # and a clipped SMILES is not a structure.
            print("\t".join((name, smiles, str(rank), t_name, t_smiles,
                             "%.3f" % score, str(int(q_bits[qi])), str(t_bits))),
                  file=fh)
