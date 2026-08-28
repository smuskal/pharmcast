# PharmCast

**Predict a complete 3D pharmacophore fingerprint directly from a 2D structure.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![White paper](https://img.shields.io/badge/white%20paper-read-8a5a2b.svg)](https://pharmcast.ai/whitepaper/)
[![Site](https://img.shields.io/badge/site-pharmcast.ai-8a5a2b.svg)](https://pharmcast.ai)
[![Try it now](https://img.shields.io/badge/try%20it%20now-pharmcast.ai-2d7d46.svg)](https://pharmcast.ai/)

A pharmacophore fingerprint records the three-dimensional arrangement of binding
features a molecule can present, so two compounds from completely different
scaffolds can be compared on the thing a protein actually reads. Its cost has
never been the fingerprint. Almost all of the cost of fingerprinting a
catalogue molecule is generating its conformer ensemble; the bit calculation
itself is a small fraction of it.

PharmCast reads a SMILES and predicts all 10,549 pharmacophores directly. It
removes the conformational stage rather than accelerating it, which is why the
speed-up is of a different order to what optimisation normally buys.

**[Try it now at pharmcast.ai](https://pharmcast.ai/)** — no install, no email.

- **[Fingerprint a molecule](https://pharmcast.ai/fingerprint.html)** — draw or
  paste one structure and get its complete predicted fingerprint: which of the
  10,549 pharmacophores are set, what each is as a feature triplet, and the
  fingerprint itself ready to copy in both packed 32-bit and `0`/`1` form.
- **[Compare two molecules](https://pharmcast.ai/compare.html)** — PharmSim
  similarity between two structures.

To decode a `.pfp` locally, see [`tools/pfp-decode`](tools/pfp-decode).

---

## The fingerprint

Every pharmacophore in the scheme is a triangle of three typed features at three
measured distances. Enumerate every combination of feature type and distance
range and you have a fixed vocabulary of triangles; each one is a bit, and a
molecule sets the bit for every triangle it can form.

![A three point pharmacophore: three typed features p1, p2, p3 joined by three measured distances, beside the seven feature types and six distance bins that enumerate 10,549 pharmacophores](assets/pharmacophore-bit.jpg)

*What one bit is. A three point pharmacophore is three typed features and the
three binned distances between them. Enumerating every triangle that satisfies
the triangle inequality on the bin bounds gives 10,549 distinct pharmacophores.*

Because the vocabulary is geometric rather than substructural, two molecules
with nothing in common on paper score highly if they present their features in
the same places. That is exactly the comparison a scaffold hop needs, and it is
not what a 2D fingerprint measures.

![Estradiol and diethylstilbestrol, each presenting an acceptor, a donor and an aromatic ring at distances that fall in the same three bins](assets/scaffold-hop.jpg)

*The same pharmacophore on two unrelated scaffolds. Estradiol (above) and
diethylstilbestrol (below) each present an acceptor (p1), a donor (p2) and an
aromatic ring (p3). The distances differ, 2.7, 7.8 and 10.4 Å against 2.7, 9.2
and 11.9, but all three fall in the same bins, so both set the same bit. That is
what scaffold hopping looks like from inside the descriptor.*

## How it works

![Two molecules through 2D features, the network, the predicted fingerprint against the real one, and the resulting similarity](assets/how-it-works.jpg)

*The two routes to a fingerprint. The conventional route generates a conformer
ensemble and runs the reference calculation over it. PharmCast predicts the same
ensemble record from the two dimensional structure and skips the ensemble
entirely. Panel E is saquinavir against indinavir, two HIV-1 protease
inhibitors, scored by PharmCast SCP v7: the reference calculation gives a
pharmacophoric similarity of 0.84 and PharmCast predicts 0.86, where a two
dimensional Morgan Tanimoto puts the same pair at 0.30. That gap is the whole
point of the descriptor. The network is 2,059 inputs, hidden layers of 1,024 and
512, and one output unit per pharmacophore, 10,549 in total, for 8,045,877
parameters.*

`saquinavir`

    CC(C)(C)NC(=O)[C@@H]1C[C@@H]2CCCC[C@@H]2CN1C[C@@H](O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(N)=O)NC(=O)c1ccc2ccccc2n1

`indinavir`

    CC(C)(C)NC(=O)[C@@H]1CN(Cc2cccnc2)CCN1C[C@@H](O)C[C@@H](Cc1ccccc1)C(=O)N[C@H]1c2ccccc2C[C@H]1O

---

## Install

```bash
pip install git+https://github.com/smuskal/pharmcast.git
# or, from a clone:
pip install -e .
```

Python ≥ 3.9 with `numpy`, `torch` and `rdkit`. **CPU only**: an
8-million-parameter network gains nothing from a GPU, so there is no CUDA build
to match and no accelerator to configure.

Runs unmodified on **Apple Silicon, Intel macOS, Windows and Linux** (x86_64 and
aarch64). The package is pure Python: no compiled extension, no architecture
build step, and nothing that assumes a byte order. Word packing uses an explicit
little-endian dtype so a fingerprint means the same thing on every machine.
NumPy 2.x is used when present and 1.x falls back to a lookup table, so a stack
pinned to NumPy 1.x is not forced to upgrade.

## Get the weights

**The weights are not in this repository.** They are served, versioned and
checksummed, from **<https://pharmcast.ai/models>**.

| Model | Endpoint | SHA-256 |
|---|---|---|
| **PharmCast SCP v7** (current) | [`/models/pharmcast_scp_v7.pt`](https://pharmcast.ai/models/pharmcast_scp_v7.pt) | `ad5aee5c…0b72d4aa8` |
| *(all releases, append-only)* | [`/models/SHA256SUMS`](https://pharmcast.ai/models/SHA256SUMS) | n/a |

```bash
curl -O https://pharmcast.ai/models/pharmcast_scp_v7.pt
curl -O https://pharmcast.ai/models/SHA256SUMS
shasum -a 256 -c SHA256SUMS                      # macOS / Linux
certutil -hashfile pharmcast_scp_v7.pt SHA256    # Windows
```

**SCP v7 is the served and published model.** Earlier checkpoints are not
distributed. Their checksums remain in `SHA256SUMS`, which is append-only, so a
copy someone already holds still verifies.

Everything measured below is a measurement of **SCP v7**, on molecules
fingerprinted after its training corpus was sealed, which the model therefore
cannot have seen.

On **25 approved-drug pairs with full reference pfpall ensembles** (100 ETKDGv3
conformers, UFF relaxed, native binary):

| | mean abs. error | median abs. error | mean signed error |
|---|---:|---:|---:|
| **SCP v7** | **0.043** | **0.031** | +0.030 |

The model reads slightly high. On formoterol/olodaterol the reference is 0.805
and v7 says 0.799.

**A published checkpoint is never replaced in place.** A new release is added
beside the old ones and `SHA256SUMS` is append-only, so a script pinned to a URL
keeps returning the same bytes. The weights carry the same Apache-2.0 licence as
the code.

## Quick start

Everything below is the command line. Install, download a model, run.

```bash
pip install -e .
# weights are served from the site, not committed; verify what you got
curl -LO https://pharmcast.ai/models/pharmcast_scp_v7.pt
curl -LO https://pharmcast.ai/models/SHA256SUMS
shasum -a 256 -c SHA256SUMS 2>/dev/null | grep pharmcast_scp_v7
```

Then fingerprint a file of SMILES:

```bash
pharmcast --model pharmcast_scp_v7.pt fp --in examples/data/molecules.smi --out molecules.pfp
```

Compare two molecules:

```bash
pharmcast --model pharmcast_scp_v7.pt sim \
  "C[C@]12CC[C@H]3[C@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O" \
  'CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1'
```

See the bits:

```bash
pharmcast --model pharmcast_scp_v7.pt bits "CC(=O)Nc1ccc(O)cc1" --width 120
pfp2bits molecules.pfp --format positions
```

And ask the model what it is:

```bash
pharmcast --model pharmcast_scp_v7.pt card
```

## Worked examples

`examples/` holds runnable scripts and the data they use. Each takes a model
path and prints its own output; nothing needs editing first.

```bash
cd examples
./01_fingerprint.sh  /path/to/pharmcast_scp_v7.pt   # SMILES file -> native .pfp
./02_similarity.sh   /path/to/pharmcast_scp_v7.pt   # 3D vs 2D similarity, three pairs
./03_bits.sh         /path/to/pharmcast_scp_v7.pt   # fingerprints -> ones and zeros
./04_model_card.sh   /path/to/pharmcast_scp_v7.pt   # corpus and applicability domain
```

`02_similarity.sh` is the one to run first. It scores three pairs **both ways**,
by pharmacophore and by ordinary 2D similarity, because the pharmacophore score
alone does not tell you anything:

```
                                      3D (PharmCast)   2D (Morgan)
  estradiol / diethylstilbestrol      pharmacophore 0.479    2D 0.163
  aspirin / salicylate                pharmacophore 0.379    2D 0.448
  caffeine / ibuprofen                pharmacophore 0.000    2D 0.087
```

*Numbers above are real output from `02_similarity.sh` on **SCP v7**, run
27 August 2026. They move with the model: the same three pairs on the withdrawn
SP v4 gave 0.5062, 0.4775 and 0.0000, and that stale transcript sat in this
README until it was re-run. If your output differs, check the checkpoint's
sha256 and your RDKit version before anything else.*

Estradiol and diethylstilbestrol are unrelated in 2D and bind the same receptor.
That gap between the columns is the scaffold hop, and it is the whole reason to
compute a pharmacophore fingerprint. Aspirin and salicylate score similarly in
3D but are obviously related in 2D already, so nothing was discovered. Caffeine
and ibuprofen are unrelated both ways.

## Command line reference

```bash
pharmcast --model M.pt fp       --in library.smi --out library.pfp
pharmcast --model M.pt sim      "SMILES_A" "SMILES_B"
pharmcast --model M.pt bits     "SMILES" [--format binary|positions|words] [--width N]
pharmcast --model M.pt card
pharmcast --model M.pt pfp2bits FILE.pfp

pfp2bits FILE.pfp                              # standalone, no model needed
pfp2bits FILE.pfp --format positions --json
```

`fp` is the one that matters for throughput: it batches, and PharmCast earns its
speed in a batch rather than one molecule at a time.

Each fingerprint is **10,549 pharmacophores**, stored as **330 unsigned 32-bit words** in the *native* format, the
same words the reference calculation emits, in the same bit convention, so
existing PharmSim tooling consumes PharmCast output unchanged.

### Writing and reading `.pfp` files

`pharmcast fp` writes the native `.pfp` layout, the same one the reference
comparison tools read:

```
name  <330 unsigned 32-bit ints>  version  MW  heavy  bits  rotb  nconf  X A D H N P R
```

Everything a 2D structure can supply is filled in, so a `.pfp` written by
PharmCast is self-describing rather than an anonymous block of integers:

```
ethanol      ... v1.6  46.1   3   3  0  100 ...
phenol       ... v1.6  94.1   7  17  0  100 ...
paracetamol  ... v1.6 151.2  11  46  1  100 ...
```

`version` is the format version, `MW` the molecular weight, `heavy` the
heavy-atom count, `bits` the number of pharmacophores set, `rotb` the rotatable
bonds and `nconf` the conformer count the fingerprint represents. The seven
per-feature-type counts (`X A D H N P R`) require the reference tool's own atom
typing and are written as `0`; no comparison tool reads them, and writing a
number derived from different typing would be worse than writing none.

`pfp2bits` turns any of that back into something a human can look at:

```
$ pfp2bits library.pfp --width 120
>paracetamol  set=46/10549  version=v1.6  mw=151.2  heavy=11  bits=46  rotb=1  nconf=100
111000101000100001000000000000000000001000000000000000000000000000000000000000000000111000110100011000001000000000000000
```

It loads no model, because the fingerprints are already in the file.

*Bit counts above are **SCP v7**, 27 August 2026. `--width 120` prints the first
120 of 10,549 pharmacophores, so it is a prefix and not a short fingerprint —
the same molecule on <https://pharmcast.ai/fingerprint.html> prints all 10,549
and begins with these same 120 characters. The counts move with the model: the
withdrawn SP v4 set 43 pharmacophores here rather than 46.*

---

## Accuracy

**PharmCast SCP v7**, measured on molecules fingerprinted after its training
corpus was sealed, so the model cannot have seen them. 900 molecules and 1,200
pairs per population.

| Population | Pairs | Median error | Pearson *r* | Within 0.05 |
|---|---:|---:|---:|---:|
| Catalogue chemistry | 1,200 | 0.017 | 0.974 | 89.3% |
| Loop peptides | 1,200 | 0.019 | 0.972 | 84.8% |
| ChEMBL, activity backed | 1,200 | 0.024 | 0.935 | 78.7% |
| **All three combined** | **3,600** | **0.020** | **0.968** | **84.2%** |
| *Reference against itself* | | *0.006* | *0.995* | *ceiling* |

Per fingerprint rather than per pair, over 30,000 held-out catalogue molecules:
**median MCC 0.909**, mean 0.898, against a ligand-blind baseline of 0.602. The
model sets 543.6 pharmacophores on average where the reference sets 540.5.

That ceiling row sets the scale. Rebuilding the same molecules with a different
embedding seed reproduces pair similarity to 0.006 at *r* 0.995, so the
reference calculation agrees with itself an order of magnitude more tightly than
PharmCast agrees with it. **The ground truth is not the problem**, and 0.006 is
the floor no surrogate can beat.

![Predicted against real pairwise PFP Tanimoto for PharmCast SCP v7, coloured by population: catalogue chemistry, ChEMBL activity backed, and loop peptides, all following the diagonal](assets/fidelity-by-population.jpg)

*PharmCast SCP v7. Predicted against real pairwise similarity on held out
molecules, by population. The dashed line is exact agreement. Catalogue
chemistry and loop peptides sit tight to it; ChEMBL compounds are the widest of
the three.*

![Per-molecule Matthews correlation coefficient between the predicted and the real fingerprint for PharmCast SCP v7](assets/per-molecule-mcc.jpg)

*How closely each individual predicted fingerprint matches the real one, as the
Matthews correlation coefficient per molecule.*

### It is not re-deriving 2D similarity

![The same pairs coloured by 2D Morgan similarity, with Pearson essentially unchanged across every 2D similarity band](assets/not-2d-similarity.jpg)

*Predicted against real pairwise similarity, coloured by two dimensional Morgan
similarity, green for the most dissimilar pairs through red for the least.
Agreement does not depend on 2D similarity, so the model is predicting three
dimensional feature geometry rather than restating its own input. PharmCast SCP
v7 on 5,000 held-out screening collection molecules, 24,664 pairs stratified
into five 2D bands. Pearson within the bands is 0.972, 0.973, 0.976, 0.978 and
0.973 from the least 2D similar to the most. That flatness is the claim the
panel makes.*

## Applicability domain: read this before trusting a score

> **PharmCast is calibrated for catalogue-like chemistry to about 600 Da,
> peptides to about 900 Da, and activity-backed ChEMBL chemistry in the band the
> corpus has reached.** Outside that range it is extrapolating.

**The upper bound moves between releases.** The ChEMBL corpus is built in
ascending molecular weight, so read `card()["applicability"]` out of the
checkpoint rather than assuming a fixed number.

The reason is in the training corpus. The screening collection is filtered at
600 Da on ingest, so it contributes nothing above that line *by construction*.
In the peptide-only SP models that left **only 1.43% of training above 600 Da,
and every molecule of it a peptide**, so a large non-peptidic drug was answered
by extrapolating from a few thousand peptides: error rose from 0.02 to 0.07 and
*r* fell from 0.97 to 0.63.

**ChEMBL exists to close exactly that gap, and it is working.** Activity-backed
ChEMBL chemistry scores 0.024 median error at *r* 0.935, against 0.017 and 0.974
for catalogue chemistry. The gap is now small. It is not closed: the corpus is
built in ascending molecular weight, so competence extends upward as it fills
rather than covering the whole range at once. ChEMBL remains the widest of the
three populations and is the number to watch.

A model card that hides its failure mode is worse than no model card.

## Training corpora

Where the three corpora stand today:

| Corpus | Fingerprinted today | Expected total | Complete |
|---|---:|---:|---:|
| Screening collection | 3,296,466 | 4,617,292 | 71.4% |
| ChEMBL | 435,314 | 2,737,190 † | 15.9% |
| Loop peptides | 30,800 | 132,878 | 23.2% |
| **All three** | **3,762,580** | **7,487,360** | **50.3%** |

![Bar chart of each corpus, fingerprinted today against the expected total: screening collection 71.4%, ChEMBL 15.9%, loop peptides 23.2%](assets/corpus-progress.jpg)

**The finished training set is expected to be 7,487,360 molecules**, about twice
the size of the corpus SCP v7 was trained on. **Half of it is fingerprinted
today.** The growth is not evenly distributed: the screening
collection is most of the mass, while ChEMBL is the least complete and grows the
most in proportion. Every molecule is fingerprinted with
the real 100-conformer calculation, which is what makes the corpus slow to build
and worth building.

† **ChEMBL is the only projected figure.** The screening collection and the
peptide corpus are enumerated sets whose totals are known, with only the
fingerprinting left to do.

![Distribution of each molecule's highest 2D Tanimoto to any other molecule in the same corpus, for the screening collection, ChEMBL and the loop peptides](assets/corpus-chemistry.jpg)

*How much distinct chemistry each corpus holds. For every molecule, the highest
2D Tanimoto to any other molecule in the same corpus, so a distribution sitting
high means the corpus is largely analog series. Medians are 0.386 for the
screening collection, 0.425 for ChEMBL and 0.862 for the loop peptides, on the
same 6,000-molecule sample from each, because nearest-neighbour similarity rises
with sample size and unequal samples cannot be compared.* ChEMBL selection is still running, so its expected
total is an estimate and will move.

SCP v7 trained on a 3,724,667-molecule snapshot:

| Corpus | Molecules | Share |
|---|---:|---:|
| Screening collection | 3,263,245 | 87.61% |
| ChEMBL, activity backed | 430,915 | 11.57% |
| Protein loop peptides | 30,507 | 0.82% |

Its output layer is 10,549 wide, one per pharmacophore.

### The training set is still being built

**This is work in progress, not a finished system.** All three corpora are still
being fingerprinted: 3,326,264 of an expected 7,487,360 molecules, **44%**.
SCP v7 is the current model, trained on a 3,724,667-molecule snapshot taken
along the way: 3,263,245 screening collection, 430,915 ChEMBL, 30,507 loop
peptides.

**Work continues toward v8 and beyond.** No model here is a release
candidate, and none will be until the corpora are complete. A preprint
describing the finished work is being written.

Each version is published beside its predecessors and never replaces one, so a
result computed against a given version stays reproducible.

**SCP v7 reserves a holdout split** of 37,623 molecules, used only for early
stopping, and the released weights are the epoch-57 ones restored after the run.
Accuracy is then measured on a separate population: molecules fingerprinted
*after* the training corpus was sealed, which the model cannot have seen.

Every release is published beside its predecessors and never replaces one.

Naming: **S** is the screening collection, **C** is ChEMBL, **P** is peptides.
There is no peptide-only model, and "composite" means SP rather than a third
thing.

The ChEMBL corpus exists to repair a specific, measured weakness. The screening
collection is filtered at MW 600 on ingest, so in the SP models **only 1.4% of
training sat above MW 600 and every molecule of it was a peptide**. Asked about a
large non-peptidic drug, SP was extrapolating, and its error rose from 0.02 to
0.07 with r falling from 0.97 to 0.63. The ChEMBL corpus is real, activity-backed
chemistry in exactly that band, built in ascending molecular weight so the
model's competence extends upward from what it already knows rather than
jumping into a gap.

**The evaluation compounds are held out of training** by ChEMBL identifier *and*
by canonical structure, because the training corpus and the large-compound
evaluation set are drawn from the same ChEMBL band and would otherwise overlap.
Training on them would improve the reported large-compound error for an entirely
artificial reason.

Loops of two to six residues were extracted from crystal structures at 2.5 Å
resolution and R-free 0.22, capped with their real flanking atoms and checked
for backbone continuity. Each distinct peptide gets **one** fingerprint that ORs
together every observed crystal conformation along with 100 computed conformers.
Repeats are kept rather than deduplicated: how often a loop is observed in a
given shape is signal about how it occupies space.

![One loop sequence in several deposited conformations, with the similarity between them](assets/loop-ensembles.jpg)

*One loop sequence, every deposited conformation of it, and the similarity
between them. The sequence LGGK appears 217 times across 61 Protein Data Bank
entries; 201 share the same 25 heavy atoms and all 201 are drawn. Across 25,000
loop sequences the median is 0.878, which is what the ensemble record is built to
capture. Regenerated at each release, so it grows as more structures are
extracted.*

---

## Two rules for using the scores

1. **Never predict the reference of a campaign.** Fingerprint it once with the
   real calculation. There is no reason to accept model error on the one
   molecule the work is aimed at.
2. **PharmCast scores rank; they are not quoted.** The predicted fingerprint
   carries a small systematic offset in bit count, harmless for ordering and
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

**[Apache-2.0](LICENSE)** covers code and model weights alike. Use it commercially,
fork it, embed it; keep the `LICENSE` and `NOTICE` files with it.

`NOTICE` credits the sources the model was trained on: the RCSB Protein Data
Bank (CC0), ChEMBL (CC BY-SA 3.0) and the Enamine screening collection. No
training corpus is redistributed here, only the trained parameters, and
keeping `NOTICE` intact, which Apache-2.0 already requires, satisfies the
attribution those sources ask for.

## White paper

The full method description, with the corpus construction, the evaluation
protocol and every figure, is current on SCP v7.

**Read it at <https://pharmcast.ai/whitepaper/>**, or open
[`docs/whitepaper.pdf`](docs/whitepaper.pdf), which GitHub renders inline.

`docs/whitepaper.html` is the same document and is kept here so the repository
is self-contained, but **GitHub shows HTML as source rather than rendering it**,
so a browser is the wrong way to read that file. Download it, or use one of the
two links above.

Note the evaluation protocol it describes: the model is scored on collection
molecules fingerprinted *after* its training corpus was sealed, so the
evaluation population is one the model cannot have seen.

## Citation

See [CITATION.cff](CITATION.cff). A preprint is in preparation; the method it
builds on is McGregor & Muskal, *J. Chem. Inf. Comput. Sci.*,
[1999](https://www.eidogen.com/pdfs/pharmprintpaper1.pdf) and
[2000](https://www.eidogen.com/pdfs/pharmprintpaper2.pdf).

---

PharmCast™, PharmPrint™, PolyPharmPrint™, PharmSim™ and ChIP™ are trademarks of
[Eidogen-Sertanty, Inc.](https://eidogen-sertanty.com)
