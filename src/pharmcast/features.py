"""Molecular featurisation, vendored from the trainer.

The feature block is part of the checkpoint contract: a model blob records
which set it was trained with under `blob["features"]`, and inference must
reproduce that block bit for bit. Both descriptor functions and the
`FEATURE_SETS` table are therefore copied here verbatim rather than imported,
so a checkpoint stays loadable independently of the training tree.

`binary2048` is the shipped block. `count1024` is retained because earlier
checkpoints name it and would otherwise fail to load.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors as rdMD

RDLogger.DisableLog("rdApp.*")

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]


def descriptors_14(mol):
    """Size, shape and composition. No logP, no TPSA, no aromatic counts.

    These are all extensive quantities: they grow with the molecule. That is
    deliberate, because PFP bit count grows with molecular size too, and the
    network needs a direct handle on scale rather than having to infer it from
    a hashed substructure histogram.
    """
    counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return ([Descriptors.MolWt(mol),
             float(mol.GetNumHeavyAtoms()),
             float(mol.GetNumBonds()),
             float(rdMD.CalcNumRotatableBonds(mol)),
             float(rdMD.CalcNumRings(mol))]
            + [float(counts.get(sym, 0)) for sym in ELEMENTS])


def descriptors_11(mol):
    return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol), rdMD.CalcNumRotatableBonds(mol),
            rdMD.CalcNumRings(mol), rdMD.CalcNumAromaticRings(mol),
            rdMD.CalcNumHBD(mol), rdMD.CalcNumHBA(mol), mol.GetNumHeavyAtoms(),
            rdMD.CalcFractionCSP3(mol), rdMD.CalcLabuteASA(mol)]


FEATURE_SETS = {
    "count1024": {"dim": 1024 + 14, "n_bits": 1024, "counts": True,
                  "desc": descriptors_14,
                  "label": "count Morgan 1024 r2 + 14 descriptors"},
    "binary2048": {"dim": 2048 + 11, "n_bits": 2048, "counts": False,
                   "desc": descriptors_11,
                   "label": "binary Morgan 2048 r2 + 11 descriptors"},
}


def featurize(smiles_list, kind: str):
    """-> float32 array [n, dim].

    A SMILES RDKit cannot parse yields an all-zero row rather than an
    exception, so a batch is never lost to one bad record. Callers that care
    which molecules failed should check with `Chem.MolFromSmiles` themselves;
    `PharmCast.words_batch` reports them through `valid_mask`.
    """
    spec = FEATURE_SETS[kind]
    n_bits, use_counts, desc_fn = spec["n_bits"], spec["counts"], spec["desc"]
    out = np.zeros((len(smiles_list), spec["dim"]), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        if use_counts:
            fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=n_bits)
            for idx, count in fp.GetNonzeroElements().items():
                out[i, idx] = float(count)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            out[i, list(fp.GetOnBits())] = 1.0
        out[i, n_bits:] = desc_fn(mol)
    return out


def valid_mask(smiles_list):
    """-> bool array, True where RDKit parsed the SMILES."""
    return np.array([bool(s) and Chem.MolFromSmiles(s) is not None
                     for s in smiles_list], dtype=bool)
