# PharmCast fingerprint decoding tools

## Scope — decoders only, by design

These tools **read** a `.pfp` fingerprint and make it human readable. They
**cannot produce one**. Fingerprint generation is performed by `pfpall`, which
is not distributed here and is not part of this repository.

Deliberately excluded, and not to be added:

- `pfpall`, `pfpall_arm64`, `pfpall.c` — the fingerprint generator
- `pharmprint_noR1.6.data`, `pharmprint_Ronly.data`, `pharmprint_symm.data` —
  the pharmacophore definition files the generator consumes
- compiled binaries of any kind — build from source with the command below

`pharm10549.list` **is** included: it is the bit-index-to-pharmacophore lookup
table the decoders need in order to label a hit. It is a decode table and confers
no ability to generate a fingerprint.

The intended workflow is that fingerprints come from PharmCast output, and these
tools make that output readable.


Turns a `.pfp` fingerprint into something a person can read.

## Format

A `.pfp` line is `<name>` followed by 330 unsigned 32 bit words. Bits are
written **MSB first inside each word**, carrying 10,549 three point
pharmacophores. The low 11 bits of word 330 are padding and are always zero.
Authority: `Handoff/reference/PHARMCAST_FINGERPRINT_WIDTH.md`.

Each set bit is one triangle: three pharmacophore features at three binned
edge distances in angstroms.

```
         p1
        /  \
   d2  /    \  d3
      /      \
    p3 ------ p2
        d1
```

| Code | Feature | | Distance bins (A) |
|---|---|---|---|
| `A` | H-bond acceptor | | 2-4.5 |
| `D` | H-bond donor | | 4.5-7 |
| `H` | Hydrophobic | | 7-10 |
| `N` | Negative charge | | 10-14 |
| `P` | Positive charge | | 14-19 |
| `R` | Aromatic ring | | 19-24 |
| `X` | No feature | | |

Feature codes are taken from the pharmacophore definition files
(`pharmprint_noR1.6.data`, `pharmprint_Ronly.data`), not inferred.

## Tools

| Tool | Purpose |
|---|---|
| `pfp2bits_arm64`, `pfp2bits_x86_64` | `.pfp` to a 10,549 character `0`/`1` string |
| `pharmstat_arm64`, `pharmstat_x86_64` | text summary plus the decoded hit list |
| `pfp_report.py` | HTML report: stats, feature triplets, distance bins, hit list |

```bash
./pfp2bits_arm64 molecules.pfp > molecules.bits
./pharmstat_arm64 molecules.pfp pharm10549.list
python3 pfp_report.py molecules.pfp -o report.html
python3 pfp_report.py chunk_06663.pfp.gz --limit 50 -o report.html
python3 pfp_report.py molecules.pfp --bits          # same output as pfp2bits
```

`pfp_report.py` reads `.pfp` and `.pfp.gz`, uses only the standard library,
and writes one self contained HTML file.

The report ends with a **Copyable fingerprint** section holding two blocks with
copy buttons: the packed `.pfp` row exactly as it was read (name plus 330
unsigned 32 bit words) and the same fingerprint expanded to a 10,549 character
`0`/`1` string, byte identical to `pfp2bits` output.

## Verification

Bit order is proven, not assumed. On real PharmCast rows all four agree
exactly: the rebuilt `pfp2bits` against the original binary, arm64 against
x86_64, and `pfp_report.py --bits` against `pfp2bits`. The Python decoder in
`pfp-surrogate` (little endian unpack then `PACK_POS`) produces the identical
bit set.

Example: `examples/CHEMBL163631.pfp` and `examples/CHEMBL163631_report.html`.

## Source changes from the originals

Both `.c` files derive from the original 2002-era PharmPrint sources, which
are unchanged upstream.
Two defects were fixed here:

1. `pharmstat.c` decremented its record count after EOF, so a single molecule
   file reported `0 = number of bitstrings processed` and an infinite average.
2. `pharmstat.c` opened `pharm10549.list` only from the current directory and
   did not check for `NULL`, so it crashed when run from anywhere else. The
   path now comes from `argv[2]`, then `$PHARM_LIST`, then the current
   directory.

Both build with `clang -std=gnu89` (the sources are K&R era and declare
`main` without a return type).

```bash
for arch in arm64 x86_64; do
  for prog in pfp2bits pharmstat; do
    clang -std=gnu89 -arch $arch -O2 -w -o ${prog}_${arch} ${prog}.c -lm
  done
done
```

These tools are decoders. They do not generate fingerprints and are outside
the `control_expectation.json` binary pin, which governs `pfpall` only.
