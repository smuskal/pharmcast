"""Inference must reduce an input to its largest fragment, as training did.

`pharmcast verify` cannot cover this: all ten of its reference molecules are
single-fragment, so it prints MATCH either way. These are the assertions that
would actually fail if the behaviour reverted.
"""
import numpy as np
import pytest
from rdkit import Chem, RDLogger

from pharmcast.features import (featurize, fragment_note,
                                largest_fragment_smiles, prepare)

RDLogger.DisableLog("rdApp.*")

PROPRANOLOL = "CC(C)NCC(O)COc1cccc2ccccc12"
PROPRANOLOL_HCL = "Cl.CC(C)NCC(O)COc1cccc2ccccc12"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def feats(smis, **kw):
    return featurize(list(smis), "binary2048", **kw)


def test_a_salt_features_identically_to_its_parent():
    a, b = feats([PROPRANOLOL]), feats([PROPRANOLOL_HCL])
    assert np.array_equal(a, b)


def test_a_single_fragment_molecule_is_untouched():
    """The whole safety argument for this change: no-op on one fragment."""
    for smi in (ASPIRIN, CAFFEINE, PROPRANOLOL, "C(C(=O)[O-])[NH3+]"):
        assert np.array_equal(feats([smi]), feats([smi], strip_salts=False))


def test_the_opt_out_really_opts_out():
    on, off = feats([PROPRANOLOL_HCL]), feats([PROPRANOLOL_HCL],
                                              strip_salts=False)
    assert not np.array_equal(on, off)
    assert np.array_equal(on, feats([PROPRANOLOL]))


def test_largest_fragment_matches_the_trainer_including_the_round_trip():
    """A sanitizeFrags=False fragment has no ring info.

    Returning the mol instead of a SMILES raises "RingInfo not initialized"
    the moment a Morgan generator touches it, so the round trip is the
    contract, not an incidental detail.
    """
    assert largest_fragment_smiles(PROPRANOLOL_HCL) == \
        Chem.MolToSmiles(Chem.MolFromSmiles(PROPRANOLOL))
    with pytest.raises(ValueError):
        largest_fragment_smiles("this_is_not_a_smiles")


def test_prepare_and_featurize_agree_on_a_heavy_atom_tie():
    """The defect that shipped once: two predicates, two different fragments.

    `max` returns the first maximum and `MolToSmiles` reorders atoms, so
    choosing from the submitted string and from the canonical one can differ.
    Benzoic acid and phenethylamine are nine heavy atoms each.
    """
    tie = "O=C(O)c1ccccc1.NCCc1ccccc1"
    kept, note = prepare(tie)
    assert note is not None
    assert kept == note["kept"]
    # What prepare reports is what featurize builds from.
    assert np.array_equal(feats([tie]), feats([kept]))


def test_fragment_note_reports_what_was_set_aside():
    note = fragment_note(PROPRANOLOL_HCL)
    assert note["n_fragments"] == 2
    assert note["dropped"] == ["Cl"]
    assert note["kept"] == Chem.MolToSmiles(Chem.MolFromSmiles(PROPRANOLOL))
    assert fragment_note(ASPIRIN) is None


def test_nothing_degenerate_raises():
    """A public endpoint feeds this arbitrary text."""
    for smi in (".", "..", "C.", ".C", "", " ", "[", "C(", "[*].[*]",
                "O.O.O.O.O", "[Na+].[Cl-]"):
        row = feats([smi])
        assert row.shape[0] == 1
        fragment_note(smi)                      # must not raise
        prepare(smi)                            # must not raise


def test_an_unparseable_input_stays_an_all_zero_row():
    assert not feats(["this_is_not_a_smiles"]).any()
    assert prepare("this_is_not_a_smiles")[0] is None


def test_read_smiles_skips_comment_lines(tmp_path):
    """count_records has always skipped them; read_smiles did not.

    Once a screen refuses an unreadable SMILES, that disagreement turns a
    commented catalogue file into an aborted run.
    """
    from pharmcast.io import read_smiles
    from pharmcast.screen import count_records
    f = tmp_path / "t.smi"
    f.write_text("CCO\ta\n\n# a comment\nc1ccccc1\tb\n")
    assert [s for s, _ in read_smiles(f)] == ["CCO", "c1ccccc1"]
    assert count_records(f) == len(list(read_smiles(f))) == 2
