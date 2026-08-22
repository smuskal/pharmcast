"""Command line entry points.

    pharmcast fp     --model M --in in.smi --out out.pfp
    pharmcast sim    --model M "SMILES_A" "SMILES_B"
    pharmcast bits   --model M "SMILES"  [--width N]
    pharmcast card   --model M
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import bits as B
from .io import read_pfp, read_smiles, write_pfp
from .model import PharmCast
from .similarity import pharmsim


def _load(a):
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


def cmd_card(a):
    print(json.dumps(_load(a).card(), indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="pharmcast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to a .pt checkpoint")
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

    s = sub.add_parser("card", help="what the checkpoint says about itself")
    s.set_defaults(fn=cmd_card)

    a = p.parse_args(argv)
    if getattr(a, "width", None) == 0:
        a.width = None
    return a.fn(a)


if __name__ == "__main__":
    main()
