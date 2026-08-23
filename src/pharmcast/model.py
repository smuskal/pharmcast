"""PharmCast: predict a PolyPharmPrint ensemble fingerprint from a SMILES.

The normal route from a structure to a 3D pharmacophore fingerprint is SMILES,
then a 100-conformer ensemble, then `pfpall` over that ensemble -- about 2.5
seconds per catalogue molecule, of which almost all is conformer generation.
PharmCast goes straight from the SMILES to the predicted fingerprint in well
under a millisecond.

THE OUTPUT IS THE STANDARD NATIVE FORMAT, not an internal representation.
`words_batch` returns the same 330 unsigned 32-bit integers that `pfpall`
emits, in the same bit convention, so any tool that consumes a native `.pfp`
consumes PharmCast output unchanged.

TWO RULES FOR USING THE SCORES
  1. Never predict the reference of a campaign. Fingerprint it once with the
     real calculation. There is no reason to accept model error on the one
     molecule the work is aimed at.
  2. PharmCast scores rank candidates; they are not quoted as the similarity.
     The predicted fingerprint carries a small systematic offset in bit count
     -- harmless for ordering, misleading in a report. Published numbers come
     from a real rescore of the survivors.

See docs/APPLICABILITY_DOMAIN.md before trusting a score on anything above
about 600 Da.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .features import featurize, valid_mask
from .net import Net

N_INTS = 330                 # 32-bit words per fingerprint
N_BITS = N_INTS * 32         # 10,560 slots
N_PHARM = 10549              # real pharmacophores; the rest is padding
FORMAT_VERSION = "v1.6"      # the version token written into a .pfp record


class PharmCast:
    """A loaded PharmCast model."""

    def __init__(self, blob, model, name):
        self.blob = blob
        self.model = model
        self.name = name
        self.mean = blob["mean"]
        self.std = blob["std"]
        self.features = blob["features"]

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path, threads=2):
        """Load a PharmCast checkpoint from a path.

        `threads` bounds torch's intra-op pool. The default of 2 is deliberate:
        the model is small enough that more threads mostly add contention, and
        an unbounded pool starves anything else sharing the machine.
        """
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        torch.set_num_threads(threads)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = Net(len(blob["mean"]), blob["hidden"], blob["n_bits"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return cls(blob, model, os.path.basename(path))

    # ------------------------------------------------------------- properties
    @property
    def version(self):
        """The model's own release label.

        Falls back to the filename when a checkpoint predates release
        stamping. A checkpoint with no `release` key has not been through
        `pharmcast.stamp`, so treat the name as informal.
        """
        return self.blob.get("release", self.name)

    def card(self):
        """-> flat dict of what the checkpoint knows about its own training.

        Read this rather than trusting a filename. The training-snapshot
        boundary is written into the blob precisely so evaluation cannot guess
        wrong about which molecules the model has seen.

        `snapshot` and `report` are nested sub-dicts in the stored blob and are
        flattened here, minus the bulk membership lists -- `peptide_names` and
        the per-chunk records run to tens of thousands of entries and are not
        summary information.
        """
        BULK = {"peptide_names", "chunk_names", "history", "taken_names",
                "chembl_names", "tail_names"}
        out = {}
        for k in ("release", "features", "hidden", "n_bits"):
            if k in self.blob:
                out[k] = self.blob[k]
        for section in ("snapshot", "report"):
            sub = self.blob.get(section)
            if isinstance(sub, dict):
                for k, v in sub.items():
                    if k in BULK:
                        out[k + "_n"] = len(v)
                    elif not isinstance(v, (list, dict)):
                        out[k] = v

        # APPLICABILITY DOMAIN. A card that reports only how many molecules a
        # model saw invites the reader to assume it saw them evenly. It did not:
        # the screening collection is filtered at MW 600 on ingest, so every
        # training molecule above that weight comes from another corpus. State
        # the range and where the model is extrapolating, in the card itself,
        # because that is where someone deciding whether to trust a prediction
        # will look.
        dom = self.blob.get("applicability")
        if isinstance(dom, dict):
            out["applicability"] = dom
        else:
            out["applicability"] = self._infer_domain(out)
        return out

    @staticmethod
    def _infer_domain(out):
        """Domain from corpus composition when the blob does not record one.

        Measured on the corpora themselves, not assumed:
          screening collection  MW 142-598, median 344, p99 498 (hard cap 600)
          loop peptides         MW 116-988, median 576, p99 843
          large ChEMBL          MW 600-1000, activity-backed, ascending build
        """
        coll = out.get("train_collection") or 0
        pep = out.get("train_peptides") or 0
        chem = out.get("train_chembl") or 0
        tail = out.get("train_tail") or 0
        parts = []
        if coll:
            parts.append({"corpus": "Enamine screening collection",
                          "molecules": coll, "mw_range": [142, 598],
                          "mw_median": 344, "mw_p99": 498,
                          "note": "filtered at MW 600 on ingest"})
        if tail:
            parts.append({"corpus": "Enamine collection, large tail",
                          "molecules": tail, "mw_range": [557, 917],
                          "mw_median": 643,
                          "note": "the compounds the MW 600 ingest filter "
                                  "skipped, fingerprinted separately"})
        if pep:
            parts.append({"corpus": "loop peptides", "molecules": pep,
                          "mw_range": [116, 988], "mw_median": 576,
                          "mw_p99": 843})
        if chem:
            parts.append({"corpus": "large ChEMBL, activity-backed",
                          "molecules": chem, "mw_range": [600, 1000],
                          "note": "built in ascending molecular weight; "
                                  "evaluation compounds held out by identifier "
                                  "and by canonical structure"})
        total = coll + tail + pep + chem
        d = {"corpora": parts, "train_molecules": total}
        if not chem and total:
            # Without ChEMBL, everything above MW 600 is peptidic.
            d["calibrated_for"] = ("catalogue-like chemistry to about MW 600, "
                                   "plus peptides to about MW 900")
            d["extrapolating_above"] = 600
            d["caveat"] = ("about 1.4% of training sits above MW 600 and all "
                           "of it is peptidic, so predictions for large "
                           "non-peptidic molecules are extrapolation: measured "
                           "error rises from 0.02 to 0.07 and r falls from "
                           "0.97 to 0.63")
        elif total:
            d["calibrated_for"] = ("catalogue-like chemistry to about MW 600, "
                                   "peptides to about MW 900, and "
                                   "activity-backed ChEMBL chemistry in the "
                                   "band the corpus has reached")
            d["caveat"] = ("the ChEMBL corpus is built in ascending molecular "
                           "weight, so the upper bound of reliable "
                           "extrapolation moves with it; check "
                           "train_chembl and the corpus band before trusting "
                           "a prediction near the top of the range")
        return d

    # ------------------------------------------------------------- prediction
    def _bits(self, smiles):
        """-> bool array [n, N_BITS] in the training bit order."""
        x = featurize(list(smiles), self.features)
        x -= self.mean
        x /= self.std
        out = []
        with torch.no_grad():
            for i in range(0, x.shape[0], 512):
                p = torch.sigmoid(self.model(torch.from_numpy(x[i:i + 512])))
                out.append(p.numpy() >= 0.5)
        return np.vstack(out) if out else np.zeros((0, N_BITS), dtype=bool)

    def words_batch(self, smiles):
        """-> list of 330-integer lists, native representation.

        THIS IS THE API THAT MATTERS. The model earns its speed in a batch: a
        single call is dominated by featurisation overhead, while a batch of a
        thousand costs almost nothing extra per molecule. Never loop this one
        molecule at a time.

        The packing is the exact inverse of how the training targets were
        unpacked, which is why the result lands in the native integer
        representation rather than a private one. Verified by round-tripping a
        real record: unpack, repack, bytes identical.
        """
        smiles = list(smiles)
        bits = self._bits(smiles)
        bits[:, N_PHARM:] = False          # padding slots are never real
        packed = np.packbits(bits, axis=1, bitorder="little")
        words = packed.view("<u4")      # explicit LE; see bits.py on byte order
        return [[int(v) for v in row] for row in words]

    def words(self, smi, *_ignored, **__ignored):
        """One molecule, or None if it does not parse.

        Extra positional arguments are accepted and ignored so this drops in
        for a real-calculation call site without touching the caller:
        PharmCast always predicts the 100-conformer ensemble, so there is no
        conformer count to honour.
        """
        try:
            if not valid_mask([smi])[0]:
                return None
            return self.words_batch([smi])[0]
        except Exception:
            return None
