"""Prove a PharmCast install reproduces the fingerprints the site publishes.

    pharmcast --model pharmcast_scp_v10.pt verify

RDKit computes the network's whole input: a Morgan block plus descriptors. A
different RDKit changes that input, so the fingerprint changes and nothing else
notices. This is not hypothetical here. Aromaticity perception changed between
2024.03 and 2025.09 for a ring bearing an exocyclic carbonyl, and caffeine moved
from 15 set bits to 11 on the same checkpoint. From 2025.09.6 a fully deuterated
methyl also stopped counting as a rotatable bond, which moves the descriptor
block for isotope-labeled compounds.

The reference below was recorded with the RDKit the published service runs, so a
match here means your fingerprints are the published ones. Regenerate it only
when the served environment changes:

    pharmcast --model <checkpoint> verify --write
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "data", "reference_fingerprints.json")

# Chosen to exercise what actually moved between RDKit releases: an aromatic
# ring with an exocyclic carbonyl, explicit hydrogen isotopes, and their
# protiated analogue, alongside ordinary drug-like structures.
SMILES = [
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "OC(=O)c1ccccc1O",
    "C1CCC(CC1)N2CCN(CC2)c3ncccn3",
    "N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(=O)O)C(=O)O",
    "[2H]C([2H])([2H])N(C)C(=O)c1ccccc1",
    "[2H]C([2H])([2H])CO",
    "[3H]CC(=O)Nc1ccccc1",
    "CC(=O)Nc1cc(-c2cc(C(=O)N(C)c3ccccc3)nn2C(C)C)ccc1N1CCCCC1",
]


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprints(model):
    """-> (per-molecule word arrays, digest over all of them)."""
    words, h = [], hashlib.sha256()
    for smi in SMILES:
        w = np.asarray(model.words(smi), dtype=np.uint32)
        words.append(w)
        h.update(w.tobytes())
    return words, h.hexdigest()


def environment():
    import rdkit.rdBase
    import torch
    return {"python": sys.version.split()[0],
            "rdkit": rdkit.rdBase.rdkitVersion,
            "numpy": np.__version__,
            "torch": torch.__version__}


def run(model, model_path=None, write=False):
    words, digest = fingerprints(model)
    counts = [int(sum(bin(int(x)).count("1") for x in w)) for w in words]
    env = environment()
    release = model.version

    if write:
        json.dump({
            "what": "PharmCast fingerprints for the SMILES in pharmcast/verify.py, "
                    "recorded in the environment the published service runs",
            "model_release": release,
            "model_sha256": _sha256_file(model_path) if model_path else None,
            "smiles": SMILES,
            "set_bits": counts,
            "sha256": digest,
            "reference_environment": env,
            "how_to_verify": "pharmcast --model <checkpoint> verify",
        }, open(REFERENCE, "w"), indent=1)
        print("wrote %s" % REFERENCE)
        print("sha256 %s" % digest)
        for k, v in env.items():
            print("  %-8s %s" % (k, v))
        return 0

    if not os.path.exists(REFERENCE):
        print("no reference recorded at %s" % REFERENCE, file=sys.stderr)
        return 1
    ref = json.load(open(REFERENCE))
    print("reference recorded with:")
    for k, v in ref["reference_environment"].items():
        print("  %-8s %s" % (k, v))
    print("this environment:")
    for k, v in env.items():
        print("  %-8s %s" % (k, v))

    if model_path and ref.get("model_sha256"):
        got = _sha256_file(model_path)
        if got != ref["model_sha256"]:
            print("\nDIFFERENT CHECKPOINT. This file is not the one the reference "
                  "was recorded from, so the fingerprints below are not comparable.")
            print("  this  %s\n  ref   %s" % (got[:32], ref["model_sha256"][:32]))
            return 1
        print("checkpoint sha256 matches the reference")

    if release != ref.get("model_release"):
        print("\nmodel release %r, reference recorded from %r"
              % (release, ref.get("model_release")))

    if digest == ref["sha256"]:
        print("\nMATCH. This install reproduces the published fingerprints.")
        return 0

    print("\nMISMATCH. sha256 %s, expected %s" % (digest, ref["sha256"]))
    print("molecules that differ (set bits here, in the reference):")
    for smi, got, want in zip(SMILES, counts, ref["set_bits"]):
        if got != want:
            print("  %5d  %5d  %s" % (got, want, smi))
    print("\nRDKit computes the network's input, so a different RDKit changes "
          "every fingerprint it touches. Install the version in "
          "pyproject.toml rather than one already on this machine.")
    return 1
