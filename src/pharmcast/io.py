"""Reading and writing native `.pfp` files.

Record layout, matching the reference tool's own output exactly:

    name  <330 ints>  version  MW  heavy  bits  rotb  nconf  X A D H N P R

Readers take a name then exactly 330 integers and ignore the rest, so the
trailing fields exist for compatibility. Molecular weight, heavy-atom count,
rotatable bonds and the set-bit count are real. The seven per-type feature
counts require the reference tool's own atom typing and are written as 0, which
no comparison tool reads.
"""
from __future__ import annotations

import gzip
import os

from .bits import N_INTS, popcount

FORMAT_VERSION = "v1.6"


def _open(path, mode="rt"):
    return (gzip.open if os.fspath(path).endswith(".gz") else open)(path, mode)


def read_pfp(path):
    """Iterate a `.pfp` (or `.pfp.gz`) -> (name, [330 ints]) pairs.

    Malformed lines are skipped rather than raised on: these files are written
    incrementally by long jobs and a half-written final line is normal.
    """
    with _open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < N_INTS + 1:
                continue
            try:
                yield parts[0], [int(v) for v in parts[1:N_INTS + 1]]
            except ValueError:
                continue


# The trailing descriptor fields, in the order the record carries them.
PFP_FIELDS = ("version", "mw", "heavy", "bits", "rotb", "nconf",
              "X", "A", "D", "H", "N", "P", "R")


def read_pfp_records(path):
    """Iterate a `.pfp` -> (name, [330 ints], metadata dict).

    `read_pfp` deliberately ignores everything after the 330 integers, because
    that is what every comparison tool does. This one keeps the trailing
    descriptor block as well, so a reader can see the format version, molecular
    weight, heavy-atom count, set-bit count, rotatable bonds and conformer count
    that the record was written with.

    Missing or short trailing blocks give an empty dict rather than raising: a
    record is still a valid fingerprint without them.
    """
    with _open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < N_INTS + 1:
                continue
            try:
                words = [int(v) for v in parts[1:N_INTS + 1]]
            except ValueError:
                continue
            tail = parts[N_INTS + 1:]
            meta = {}
            for key, val in zip(PFP_FIELDS, tail):
                if key == "version":
                    meta[key] = val
                elif key == "mw":
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        pass
                else:
                    try:
                        meta[key] = int(val)
                    except ValueError:
                        pass
            yield parts[0], words, meta


def write_pfp(path, records, nconf=100):
    """Write a native `.pfp` from (name, words) pairs, or (name, words, smiles).

    When a SMILES is supplied the real descriptor fields are filled in from it;
    otherwise they are written as 0, which readers ignore. Returns the number
    of records written.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdMolDescriptors as rdMD
    RDLogger.DisableLog("rdApp.*")

    n = 0
    with _open(path, "wt") as fh:
        for rec in records:
            name, words = rec[0], rec[1]
            smi = rec[2] if len(rec) > 2 else None
            mol = Chem.MolFromSmiles(smi) if smi else None
            mw = Descriptors.MolWt(mol) if mol else 0.0
            heavy = mol.GetNumHeavyAtoms() if mol else 0
            rotb = rdMD.CalcNumRotatableBonds(mol) if mol else 0
            fh.write("%s %s %s %.1f %d %d %d %d 0 0 0 0 0 0 0\n"
                     % (str(name).replace(" ", "_"),
                        " ".join(str(int(v)) for v in words),
                        FORMAT_VERSION, mw, heavy, popcount(words), rotb,
                        nconf))
            n += 1
    return n


def read_smiles(path):
    """Iterate a whitespace-delimited .smi -> (smiles, name) pairs.

    A line with no second column gets a generated name, so an unnamed SMILES
    list works without preprocessing.
    """
    i = 0
    with _open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            i += 1
            yield parts[0], (parts[1] if len(parts) > 1 else "PC%07d" % i)
