"""The published fingerprints, replayed.

RDKit computes the network's input, so the library version decides the
fingerprint. This runs the same check `pharmcast verify` runs, and skips when no
checkpoint is at hand, because the weights are served from the site rather than
committed here.

    PHARMCAST_MODEL=/path/to/pharmcast_scp_v10.pt pytest tests/ -q
"""
import json
import os

import pytest

from pharmcast import PharmCast
from pharmcast import verify as V


def _checkpoint():
    p = os.environ.get("PHARMCAST_MODEL")
    if p and os.path.exists(p):
        return p
    for name in ("pharmcast_scp_v10.pt",):
        if os.path.exists(name):
            return name
    return None


def test_reference_file_is_shipped_and_readable():
    ref = json.load(open(V.REFERENCE))
    assert len(ref["smiles"]) == len(V.SMILES) == len(ref["set_bits"])
    assert ref["reference_environment"]["rdkit"] >= "2025.09.6"


def test_this_environment_reproduces_the_published_fingerprints():
    path = _checkpoint()
    if path is None:
        pytest.skip("no checkpoint; set PHARMCAST_MODEL to run this")
    model = PharmCast.load(path)
    ref = json.load(open(V.REFERENCE))
    if ref.get("model_sha256") and V._sha256_file(path) != ref["model_sha256"]:
        pytest.skip("a different checkpoint from the one the reference records")
    _, digest = V.fingerprints(model)
    assert digest == ref["sha256"], (
        "this environment computes different fingerprints from the published "
        "ones; run `pharmcast --model <checkpoint> verify` for the molecules "
        "that differ")
