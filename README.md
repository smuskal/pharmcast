# PharmCast

**Predict a complete 3D pharmacophore fingerprint directly from a 2D structure.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-pharmcast.ai-8a5a2b.svg)](https://pharmcast.ai)

A pharmacophore fingerprint records the three-dimensional arrangement of binding
features a molecule can present, so two compounds from completely different
scaffolds can be compared on the thing a protein actually reads. Its cost has
never been the fingerprint — of the **2.48 s** needed to fingerprint one
catalogue molecule over 100 conformers, **2.44 s is conformer generation** and
only 0.043 s is the fingerprint itself.

PharmCast reads a SMILES and predicts all 10,560 bits directly. It removes the
conformational stage rather than accelerating it, which is why the speed-up is
of a different order to what optimisation normally buys: **a complete comparison
of two molecules, from structure alone, takes 0.034 ms against 5.0 s.**

---

## The fingerprint

Every pharmacophore in the scheme is a triangle of three typed features at three
measured distances. Enumerate every combination of feature type and distance
range and you have a fixed vocabulary of triangles; each one is a bit, and a
molecule sets the bit for every triangle it can form.

![A three-point pharmacophore and the enumeration that turns it into one bit](assets/pharmacophore-triplet.png)

Because the vocabulary is geometric rather than substructural, two molecules
with nothing in common on paper score highly if they present their features in
the same places. That is exactly the comparison a scaffold hop needs, and it is
not what a 2D fingerprint measures.

![The same typed triangle located in two unrelated three-dimensional structures](assets/pharmacophore-3d.png)

---

## Install

```bash
pip install pharmcast          # once published
# or, from a clone:
pip install -e .
```

Requires Python ≥ 3.9, `numpy`, `torch` and `rdkit`. Model weights are attached
to [releases](https://github.com/smuskal/pharmcast/releases) rather than
committed, and carry the same Apache-2.0 licence as the code.

## Quick start

```python
from pharmcast import PharmCast, pharmtan, pharmsim

pc = PharmCast.load("PharmCastSP.21August2026.pt")

# words_batch IS THE API THAT MATTERS. The model earns its speed in a batch;
# a single call is dominated by featurisation overhead.
a, b = pc.words_batch(["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"])

pharmtan(a, b)      # 0.0957 — the PharmSim coefficient
pharmsim(a, b)      # ...with the shared and exclusive bit sets shown
```

Each fingerprint is **330 unsigned 32-bit words** in the *native* format — the
same words the reference calculation emits, in the same bit convention — so
existing PharmSim tooling consumes PharmCast output unchanged.

### Screening a collection

```python
from pharmcast import pharmtan_matrix, read_pfp

library = [w for _, w in read_pfp("library.pfp")]
query   = pc.words_batch(["c1ccc2c(c1)nc(n2)N1CCNCC1"])
scores  = pharmtan_matrix(query, library)[0]     # vectorised popcount
```

### Looking at the bits

```python
from pharmcast import to_bitstring, set_bits, popcount, native_index

popcount(a)                    # 87 bits set of 10,549
set_bits(a)[:5]                # the positions that are on
to_bitstring(a, width=80)      # '01010001001011000001011011011011...'
native_index(set_bits(a))      # ...renumbered into pfpall's own ordering
```

## Command line

```bash
pharmcast --model M.pt fp   --in library.smi --out library.pfp
pharmcast --model M.pt sim  "CCO" "c1ccccc1O"
pharmcast --model M.pt bits "CC(=O)Oc1ccccc1C(=O)O" --format positions
pharmcast --model M.pt card                    # what the checkpoint says of itself
```

---

## Accuracy

Measured on molecules the model has never seen, on a validation set held fixed
across every release so versions stay comparable.

![PharmCast similarity against the real calculation across three chemistries](assets/three-chemistries.png)

| Regime | Median error | Correlation *r* | Pairwise ranking |
|---|---:|---:|---:|
| Catalogue chemistry | 0.02 | 0.97 | 92% |
| Loop peptides | 0.02 | 0.97 | — |
| Large compounds, above 600 Da | 0.07 | 0.63 | 71% |
| *The real calculation against itself* | *0.006* | *0.995* | *ceiling* |

That last row sets the scale. Rebuilding the same molecules with a different
embedding seed reproduces pair similarity to 0.006 at *r* 0.995 — the reference
agrees with itself an order of magnitude more tightly than PharmCast agrees with
it. **The ground truth is not the problem**, and 0.006 is the floor no surrogate
can beat.

![30,000 pairs: surrogate similarity against real similarity](assets/surrogate-vs-real.png)

## Applicability domain — read this before trusting a score

> **PharmCast SP is calibrated for catalogue-like chemistry up to about 600 Da,
> plus peptides to about 900 Da.** Outside that range it is extrapolating.

The reason is entirely in the training corpus. The screening collection is
filtered at 600 Da on ingest, so it contributes nothing above that line *by
construction*. Only about **37,060 training molecules — 1.43% of the corpus —
sit above 600 Da, and every one of them is a peptide.** Above 800 Da it is
0.10%. So when PharmCast is asked about a large non-peptide drug it is
extrapolating from a few thousand peptides, and the error rises from 0.02 to
0.07 while *r* falls from 0.97 to 0.63. This is a genuine size effect and not
merely a harder task: a matched-spread catalogue control gives 92% on the
identical protocol.

A model card that hides its failure mode is worse than no model card.

## Training corpora

| Corpus | Molecules | Share | Mass range |
|---|---:|---:|---|
| Screening collection | 2,511,440 | 96.69% | 142–598 Da, median 344 |
| Loop peptides, ensemble enhanced | 86,039 | 3.31% | 116–988 Da, median 576 |
| **Total** | **2,597,479** | 100% | |

Naming: **S** is the screening collection, **P** is peptides. There is no
peptide-only model, and "composite" means SP rather than a third thing.

Loops of two to six residues were extracted from crystal structures at 2.5 Å
resolution and R-free 0.22, capped with their real flanking atoms and checked
for backbone continuity. Each distinct peptide gets **one** fingerprint that ORs
together every observed crystal conformation along with 100 computed conformers.
Repeats are kept rather than deduplicated: how often a loop is observed in a
given shape is signal about how it occupies space.

![One loop sequence, seven deposited conformations, and the similarity between them](assets/loop-ensembles.png)

---

## Two rules for using the scores

1. **Never predict the reference of a campaign.** Fingerprint it once with the
   real calculation. There is no reason to accept model error on the one
   molecule the work is aimed at.
2. **PharmCast scores rank; they are not quoted.** The predicted fingerprint
   carries a small systematic offset in bit count — harmless for ordering and
   misleading in a report. Published numbers come from a real rescore of the
   survivors.

There is a third, which matters under optimisation pressure: **a search pointed
at PharmCast will eventually optimise its error rather than the molecule.**
Measured on a design campaign, seventeen times the search budget raised the
median surrogate score by 0.20 and the median real score by 0.02, while the
surrogate's own error over survivors went from +0.06 to +0.24. Rescore before
believing anything, and carry many survivors into that rescore rather than a top
handful.

## What is *not* here

This repository ships the model and the utilities for using it. It does **not**
ship the reference pharmacophore fingerprint generator (`pfpall` / `pfprigid`)
or any of the fingerprint-generating toolchain, and it does not redistribute any
training corpus.

## Licence

**[Apache-2.0](LICENSE)** — code and model weights alike. Use it commercially,
fork it, embed it; keep the `LICENSE` and `NOTICE` files with it.

`NOTICE` credits the sources the model was trained on: the RCSB Protein Data
Bank (CC0), ChEMBL (CC BY-SA 3.0) and the Enamine screening collection. No
training corpus is redistributed here — only the trained parameters — and
keeping `NOTICE` intact, which Apache-2.0 already requires, satisfies the
attribution those sources ask for.

## Citation

See [CITATION.cff](CITATION.cff). A preprint is in preparation; the method it
builds on is McGregor & Muskal, *J. Chem. Inf. Comput. Sci.*,
[1999](https://www.eidogen.com/pdfs/pharmprintpaper1.pdf) and
[2000](https://www.eidogen.com/pdfs/pharmprintpaper2.pdf).

---

PharmCast™, PharmPrint™, PolyPharmPrint™, PharmSim™ and ChIP™ are trademarks of
[Eidogen-Sertanty, Inc.](https://eidogen-sertanty.com)
