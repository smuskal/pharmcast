# PharmCast fingerprint decoding tools

Turns a `.pfp` fingerprint into something a person can read.

[![Fingerprint a molecule](https://img.shields.io/badge/▶%20Fingerprint%20a%20molecule-pharmcast.ai-2d7d46?style=for-the-badge)](https://pharmcast.ai/fingerprint.html)
[![Compare two molecules](https://img.shields.io/badge/▶%20Compare%20two%20molecules-pharmcast.ai-4a7c1f?style=for-the-badge)](https://pharmcast.ai/compare.html)

## Scope

These tools read a `.pfp` fingerprint and make it human readable.
Fingerprints come from PharmCast output, and these tools make that output
readable.

`pharm10549.list` is included: it is the bit-index-to-pharmacophore lookup
table the decoders need in order to label a hit.

## Try it now — the same report, in the browser

**Fingerprint a molecule** does in a browser what `pfp_report.py` does locally.
Draw or paste one structure and it returns the complete predicted fingerprint:
which of the 10,549 pharmacophores are set, each one named as a feature triplet,
and the fingerprint itself ready to copy in both packed 32-bit and `0`/`1` form.
Nothing to install.

**Compare two molecules** returns the PharmSim similarity between two structures.

Use those to see the output; use the tools here when you have `.pfp` files of
your own to decode in bulk, offline, or in a pipeline.

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
the generator's own pharmacophore definition files, not inferred.

## Tools

| Tool | Purpose |
|---|---|
| `pfp2bits` | `.pfp` to a 10,549 character `0`/`1` string |
| `pharmstat` | text summary plus the decoded hit list |
| `pfp_report.py` | HTML report: stats, feature triplets, distance bins, hit list |
| `Makefile`, `run_tests.py` | build and the bit-order regression suite |

## Build and test

No binaries are shipped. There is a Makefile:

```bash
make            # build pfp2bits and pharmstat for this machine
make all-archs  # also build arm64 and x86_64 (macOS cross-compile)
make test       # run the bit-order regression suite
make clean
```

`make test` pins the contract that matters. Over `examples/CHEMBL163631.pfp` it
asserts the bit string is exactly 10,549 characters with exactly 173 set, that
`pfp_report.py --bits` is byte identical to `pfp2bits`, that `pharmstat` reports
1 record and a 173.0 average with 173 hit rows, and that both copy blocks in the
committed HTML unescape to their exact sources. With `make all-archs` first it
also proves arm64 and x86_64 agree byte for byte; otherwise it says it skipped
that, rather than passing quietly.

Bit order is the whole contract — a silent change corrupts every downstream
consumer and nothing else would notice.

Building by hand, if you prefer:

```bash
clang -std=gnu89 -O2 -w -o pfp2bits  pfp2bits.c  -lm
clang -std=gnu89 -O2 -w -o pharmstat pharmstat.c -lm
``` The sources are K&R era
and declare `main` without a return type, hence `-std=gnu89 -w`. `pfp_report.py`
needs no build — Python 3, standard library only.

## Worked example

Everything below runs against `examples/CHEMBL163631.pfp`, one molecule, and
every output file it produces is committed next to it so you can diff.

### 1. `pfp2bits` — the raw bit string

```bash
./pfp2bits examples/CHEMBL163631.pfp > CHEMBL163631.bits
```

Writes `<name> <10549 characters of 0/1>`. Progress goes to stderr
(`CHEMBL163631 numbits: 173`), so redirecting stdout keeps the data clean.
Committed as `examples/CHEMBL163631.bits` — 173 bits set.

### 2. `pharmstat` — summary and decoded hit list

```bash
./pharmstat examples/CHEMBL163631.pfp pharm10549.list
```

```
     1 = number of bitstrings processed
 173.0 = average number of pharmacophores hit per molecule
   1.6 = percentage of pharmacophores hit at least once (out of 10549)

Pharmacophore hit list:
    ...
 3725 | 4.5-7 | 4.5-7 |  7-10 | H | H | X |
 3746 | 4.5-7 | 4.5-7 |  7-10 | H | R | X |
```

Each row is `bit index | d1 | d2 | d3 | p1 | p2 | p3` — the three edge lengths
of the triangle, then the three features at its corners. The row above says:
bit 3746 is an aromatic ring and two hydrophobes at 4.5-7, 4.5-7 and 7-10 A.

The second argument is optional; `pharmstat` also accepts `$PHARM_LIST` or a
`pharm10549.list` in the working directory. Full output is committed as
`examples/CHEMBL163631_pharmstat.txt`.

### 3. `pfp_report.py` — the HTML report

```bash
python3 pfp_report.py examples/CHEMBL163631.pfp -o report.html
```

The readable one. Same information as `pharmstat` plus feature-triplet and
distance-bin rollups, a per-molecule table, and a copyable packed row. Committed
as `examples/CHEMBL163631_report.html`.

**GitHub will not render it.** Viewing an `.html` blob on github.com shows the
source, not the page — GitHub never executes repository HTML. To see it:

```bash
open examples/CHEMBL163631_report.html     # macOS; or just double-click it
```

Cloning and opening locally is the reliable route. GitHub Pages would render it,
but only once this repository is public.

Other forms:

```bash
python3 pfp_report.py chunk_06663.pfp.gz --limit 50 -o report.html
python3 pfp_report.py examples/CHEMBL163631.pfp --bits    # identical to pfp2bits
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
the `pharmcast` package (little endian unpack then `PACK_POS`) produces the
identical bit set.

Example: `examples/CHEMBL163631.pfp` and `examples/CHEMBL163631_report.html`.

## Source changes from the originals

Both `.c` files derive from the original 2002-era sources, which
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
the `control_expectation.json` binary pin, which governs the reference
generator only.
