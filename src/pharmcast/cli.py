"""Command line entry points.

    pharmcast fp       --model M --in in.smi --out out.pfp
    pharmcast sim      --model M "SMILES_A" "SMILES_B"
    pharmcast bits     --model M "SMILES"  [--width N]
    pharmcast card     --model M
    pharmcast pfp2bits FILE.pfp                 (no model needed)

`pfp2bits` is also installed as its own command, so a `.pfp` written by
`pharmcast fp` can be expanded to ones and zeros without loading a model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import bits as B
from .io import read_pfp, read_pfp_records, read_smiles, write_pfp
from .model import PharmCast
from .similarity import pharmsim


def _load(a):
    # --model is optional at the parser level because pfp2bits reads a file
    # that already holds fingerprints and has no use for a network.
    if not getattr(a, "model", None):
        raise SystemExit("this command needs --model, a path to a .pt checkpoint")
    return PharmCast.load(a.model, threads=a.threads)


def _words(pc, spec):
    """A SMILES on the command line, or name:index into a .pfp -- both give a
    fingerprint, so `sim` can compare a prediction against a real record."""
    if spec.endswith(".pfp") or spec.endswith(".pfp.gz"):
        _, w = next(read_pfp(spec))
        return w
    return pc.words_batch([spec])[0]


def cmd_fp(a):
    pairs = list(read_smiles(a.infile))
    t0 = time.time()
    words = pc_words = _load(a).words_batch([s for s, _ in pairs])
    dt = time.time() - t0
    n = write_pfp(a.out, [(nm, w, s) for (s, nm), w in zip(pairs, pc_words)])
    print("%d fingerprints -> %s  (%.3f s, %.1f mol/s)"
          % (n, a.out, dt, len(pairs) / max(dt, 1e-9)), file=sys.stderr)


def cmd_sim(a):
    pc = _load(a)
    x, y = _words(pc, a.a), _words(pc, a.b)
    r = pharmsim(x, y)
    if a.json:
        r = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
        print(json.dumps(r, indent=2))
        return
    print("PharmSim   %.4f" % r["tanimoto"])
    print("bits A     %d" % r["bits_a"])
    print("bits B     %d" % r["bits_b"])
    print("shared     %d" % r["shared"])
    print("only A     %d" % r["only_a"])
    print("only B     %d" % r["only_b"])
    print("union      %d" % r["union"])


def cmd_bits(a):
    pc = _load(a)
    w = _words(pc, a.smiles)
    if a.format == "words":
        print(" ".join(str(v) for v in w))
    elif a.format == "positions":
        print(" ".join(str(int(v)) for v in B.set_bits(w)))
    else:
        print(B.to_bitstring(w, width=a.width))
    print("%d bits set of %d" % (B.popcount(w), B.N_PHARM), file=sys.stderr)


def cmd_pfp2bits(a):
    """Expand a native .pfp into ones and zeros, record by record.

    A .pfp stores each fingerprint as 330 unsigned 32-bit integers, which is
    compact and completely unreadable. This prints the same fingerprint as
    explicit bits so it can be eyeballed, diffed, or piped into anything that
    expects a bit vector.

    The descriptor block each record carries -- format version, molecular
    weight, heavy atoms, set bits, rotatable bonds, conformer count -- is
    printed alongside, because a bit vector with no provenance is not much use.
    No model is loaded: the fingerprints are already in the file.
    """
    n = 0
    for name, words, meta in read_pfp_records(a.infile):
        n += 1
        if a.limit and n > a.limit:
            break
        if a.format == "words":
            body = " ".join(str(v) for v in words)
        elif a.format == "positions":
            body = " ".join(str(int(v)) for v in B.set_bits(words))
        else:
            body = B.to_bitstring(words, width=a.width)
        if a.json:
            print(json.dumps({"name": name, "metadata": meta,
                              "set_bits": int(B.popcount(words)),
                              "n_pharmacophores": B.N_PHARM,
                              a.format: body}))
            continue
        desc = "  ".join("%s=%s" % (k, meta[k]) for k in
                         ("version", "mw", "heavy", "bits", "rotb", "nconf")
                         if k in meta)
        print(">%s  set=%d/%d  %s" % (name, B.popcount(words), B.N_PHARM, desc))
        print(body)
    if not n:
        raise SystemExit("no fingerprint records found in %s" % a.infile)
    print("%d record%s from %s" % (n if not a.limit else min(n, a.limit),
                                   "" if n == 1 else "s", a.infile),
          file=sys.stderr)


def cmd_card(a):
    print(json.dumps(_load(a).card(), indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="pharmcast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="path to a .pt checkpoint; required by "
                                   "every command except pfp2bits")
    p.add_argument("--threads", type=int, default=2)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fp", help="SMILES file -> native .pfp")
    s.add_argument("--in", dest="infile", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_fp)

    s = sub.add_parser("sim", help="compare two molecules")
    s.add_argument("a"); s.add_argument("b")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_sim)

    s = sub.add_parser("bits", help="show the bit set of one molecule")
    s.add_argument("smiles")
    s.add_argument("--format", choices=("binary", "positions", "words"),
                   default="binary")
    s.add_argument("--width", type=int, default=256,
                   help="binary format only: slots to show (0 for all)")
    s.set_defaults(fn=cmd_bits)

    s = sub.add_parser("pfp2bits",
                       help="expand a .pfp file into ones and zeros (no model)")
    s.add_argument("infile", help="a .pfp or .pfp.gz written by `pharmcast fp`")
    s.add_argument("--format", choices=("binary", "positions", "words"),
                   default="binary")
    s.add_argument("--width", type=int, default=0,
                   help="binary format: slots to show, 0 for all")
    s.add_argument("--limit", type=int, default=0,
                   help="stop after this many records, 0 for all")
    s.add_argument("--json", action="store_true",
                   help="one JSON object per record, metadata included")
    s.set_defaults(fn=cmd_pfp2bits)

    s = sub.add_parser("card", help="what the checkpoint says about itself")
    s.set_defaults(fn=cmd_card)

    a = p.parse_args(argv)
    if getattr(a, "width", None) == 0:
        a.width = None
    return a.fn(a)


if __name__ == "__main__":
    main()


def pfp2bits_main(argv=None):
    """Standalone `pfp2bits` entry point.

    The same expansion as `pharmcast pfp2bits`, installed under its own name
    because reading a fingerprint file has nothing to do with the model and
    should not look as though it does.
    """
    p = argparse.ArgumentParser(
        prog="pfp2bits",
        description="Expand a native .pfp fingerprint file into ones and zeros.")
    p.add_argument("infile", help="a .pfp or .pfp.gz")
    p.add_argument("--format", choices=("binary", "positions", "words"),
                   default="binary")
    p.add_argument("--width", type=int, default=0,
                   help="binary format: slots to show, 0 for all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.width == 0:
        a.width = None
    return cmd_pfp2bits(a)
