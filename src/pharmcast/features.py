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

import sys

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


def _largest(fragments):
    """The fragment the trainer would keep, chosen deterministically.

    ONE IMPLEMENTATION, called by both public helpers below, so they cannot
    drift apart about which fragment is "largest".

    THE TIE-BREAK IS NOT DECORATION. `max` returns the FIRST maximum, so
    heavy-atom count alone makes the answer depend on the order the fragments
    were typed: measured, `"CCO.CCN"` kept `CCO` and `"CCN.CCO"` kept `CCN`.
    Two spellings of one 1:1 co-former salt would score two different
    molecules and return two different results. The canonical SMILES breaks
    the tie, so the choice depends only on the chemistry.
    """
    return max(fragments,
               key=lambda f: (f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))


def largest_fragment_smiles(smiles):
    """Apply the same salt handling the training corpus builders used.

    Returns a canonical SMILES string, never a mol, and that is deliberate. A
    fragment taken with `sanitizeFrags=False` carries no ring information, so
    handing it straight to a Morgan generator raises "RingInfo not
    initialized". The round trip through `MolToSmiles` is what makes the
    fragment usable; it is not a step to be optimised away.

    Raises ValueError on a SMILES RDKit cannot parse, matching the trainer.
    `prepare` is what the services call; this is the trainer-shaped entry
    point, kept because the published tests pin it and because it documents
    the rule. Both go through `_largest`, so there is still one predicate.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("unparseable SMILES")
    if "." in smiles:
        mol = _largest(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    return Chem.MolToSmiles(mol)


def fragment_note(smiles):
    """-> None for a single fragment, else what largest-fragment keeps and drops.

    Reporting only; the caller decides how to word it. Returns None for an
    unparseable SMILES, because that is a different problem with its own
    message, and None rather than raising if any fragment will not write back
    out as SMILES: this feeds a public response, and a note nobody asked for
    must never be the thing that fails a request.
    """
    if not smiles or "." not in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(fragments) < 2:
            return None
        kept = _largest(fragments)
        return {"n_fragments": len(fragments),
                "kept": Chem.MolToSmiles(kept),
                "dropped": [Chem.MolToSmiles(f)
                            for f in fragments if f is not kept]}
    except Exception as exc:                                   # noqa: BLE001
        # THIS HANDLER MUST NOT RAISE. It is the fall-back for a systematic
        # break in the fragment path, and anything thrown here escapes
        # `fragment_note` into `prepare`, which does not guard this call, and
        # turns a documented fall-back into a 500 on a public endpoint and a
        # lost `featurize` batch.
        #
        # stderr, not RDLogger. `RDLogger.logger()` has no `message()` -- it
        # carries debug/info/warning/error/critical -- so the previous call
        # raised AttributeError every time this handler ran, which is exactly
        # the failure it existed to prevent. `DisableLog("rdApp.*")` above
        # would have swallowed a real RDKit log line anyway.
        #
        # Deliberately quiet about the molecule, which may be a visitor's, but
        # not quiet about the fault.
        try:
            sys.stderr.write("pharmcast: fragment_note failed: %s\n"
                             % str(exc)[:200])
        except Exception:                                      # noqa: BLE001
            pass
        return None


def prepare(smiles, strip_salts=True):
    """-> (smiles_that_will_be_featurised_or_None, note_or_None).

    ONE PREDICATE, because two of them disagreed. The feature path decided to
    strip on `"." in smiles` while the reporting path decided on what
    `fragment_note` returned, so a record could be scored as its parent and
    reported as the salt, or validated as readable and then featurised into an
    all-zero row. Every caller that needs to know what was scored, or whether
    it can be scored at all, asks this.

    A None first element means there is nothing to featurise: unparseable
    input, or a fragment that will not read back.
    """
    if not smiles:
        return None, None
    note = fragment_note(smiles) if strip_salts else None
    candidate = note["kept"] if note else smiles
    try:
        if Chem.MolFromSmiles(candidate) is None:
            return None, note
    except Exception:                                          # noqa: BLE001
        return None, note
    return candidate, note


def featurize(smiles_list, kind: str, strip_salts: bool = True):
    """-> float32 array [n, dim].

    A SMILES RDKit cannot parse yields an all-zero row rather than an
    exception, so a batch is never lost to one bad record. Callers that care
    which molecules failed should check with `Chem.MolFromSmiles` themselves;
    `PharmCast.words_batch` reports them through `valid_mask`.

    `strip_salts` reduces each input to its largest fragment first, which is
    what training did: `chembl_scp_fingerprint.py` wrote the stripped SMILES
    into the index column the trainer reads its SMILES from, so a counterion
    left in place presents the network with a molecule it never saw.

    A single-fragment input is left alone entirely, not merely unchanged: the
    dot test below skips the reduction, because routing every molecule through
    an extra parse and rewrite cost 24% of featurisation throughput for a
    result measured to be bit-identical. Only a SMILES carrying a dot pays.

    Leave it on unless the caller has its own desalting policy. CHIA keeps
    metal-bearing fragments deliberately, because a largest-fragment rule
    discards the metal, and it passes False for that reason.
    """
    spec = FEATURE_SETS[kind]
    n_bits, use_counts, desc_fn = spec["n_bits"], spec["counts"], spec["desc"]
    out = np.zeros((len(smiles_list), spec["dim"]), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if strip_salts and smi and "." in smi:
            smi, _note = prepare(smi, strip_salts=True)
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
