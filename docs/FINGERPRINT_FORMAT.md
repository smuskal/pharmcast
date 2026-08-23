# The fingerprint format

## On the wire

A fingerprint is **330 unsigned 32-bit words**, so 10,560 slots, of which the
first **10,549** are real pharmacophores and the remainder is padding that is
never set.

A native `.pfp` record is whitespace-delimited, one molecule per line:

```
<name> <330 unsigned 32-bit words> <version> <MW> <heavy> <bits> <rotb> <nconf> X A D H N P R
```

Readers take a name then exactly 330 integers and ignore the rest, so the
trailing fields exist for compatibility. `MW`, `heavy`, `bits` and `rotb` are
real. The seven per-type feature counts require the reference tool's own atom
typing and are written as `0` by this library, which no comparison tool reads.

## What a bit means

Every bit is one **three-point pharmacophore**: three typed features at three
binned distances.

- **7 feature types**: acceptor, donor, negative, positive, hydrophobic,
  aromatic, other.
- **6 distance bins**: 2.0 to 4.5, 4.5 to 7.0, 7.0 to 10.0, 10.0 to 14.0,
  14.0 to 19.0, 19.0 to 24.0 Angstrom.

Every legal triangle of three typed features at three binned distances gets one
bit.

These are **ensemble** fingerprints: the bitwise OR over a conformer ensemble.
A set bit therefore means *this molecule can present this triplet in some
accessible conformation*, not that it does. Typical set-bit counts are around
**534** for catalogue-sized compounds and around **2,015** for peptides, so a
large size mismatch depresses a Tanimoto on its own, and these values are not
on a shared scale with 2D fingerprint Tanimotos.

## How the reference is generated

100 conformers, RDKit ETKDGv3 with a fixed seed plus UFF, largest fragment,
hydrogens added, from isomeric SMILES; per-conformer fingerprints combined with
a bitwise OR into one ensemble record.

## The bit convention, and why not to "fix" one side

Two orderings are in play and both are correct in their own frame:

| | Placement of pharmacophore *i* |
|---|---|
| **native** | word `i // 32`, bit `31 - (i % 32)` from the least significant end |
| **packed** | this library unpacks the same words little-endian with no byteswap |

Everything is self-consistent because the same convention is used on both sides:
`pack` is the exact inverse of `unpack`, and `pack` reproduces the native word
representation byte for byte. The difference only matters when you need to talk
about a specific *pharmacophore number*, to line a bit up against the reference
tool's own numbering, and `native_index` converts. It is its own inverse, so
one function converts both ways.

This is pinned by `tests/test_bits.py`. Changing one side to match the other
breaks every existing fingerprint.
