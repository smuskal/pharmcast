#!/usr/bin/env python
"""End to end in one file: SMILES in, PharmSim out, bits shown.

    python examples/quickstart.py /path/to/PharmCastSP.21August2026.pt
"""
import sys
import time

from pharmcast import (PharmCast, pharmsim, pharmtan, pharmtan_matrix,
                       popcount, to_bitstring)

MOLECULES = {
    "aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
    "salicylate":  "OC(=O)c1ccccc1O",
    "caffeine":    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
}


def main(model_path):
    pc = PharmCast.load(model_path)
    print("model    %s" % pc.version)
    for k, v in pc.card().items():
        print("  %-26s %s" % (k, v))

    names = list(MOLECULES)
    t0 = time.time()
    fps = pc.words_batch([MOLECULES[n] for n in names])
    dt = time.time() - t0
    print("\n%d molecules in %.4f s (%.2f ms each)" % (len(fps), dt,
                                                       1000 * dt / len(fps)))

    print("\nset bits")
    for n, w in zip(names, fps):
        print("  %-12s %4d of 10,549" % (n, popcount(w)))

    print("\nPharmSim matrix")
    m = pharmtan_matrix(fps)
    print("  %-12s %s" % ("", " ".join("%6.6s" % n for n in names)))
    for n, row in zip(names, m):
        print("  %-12s %s" % (n, " ".join("%6.3f" % v for v in row)))

    # Aspirin and salicylate differ by an acetyl. The decomposition shows
    # where that lands: mostly bits aspirin has and salicylate does not.
    r = pharmsim(fps[0], fps[1])
    print("\naspirin vs salicylate")
    for k in ("tanimoto", "bits_a", "bits_b", "shared", "only_a", "only_b"):
        print("  %-12s %s" % (k, r[k]))
    print("\nfirst 96 slots of aspirin\n  %s" % to_bitstring(fps[0], width=96))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
