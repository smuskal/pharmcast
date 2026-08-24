# The fingerprint width contract

**A PharmPrint fingerprint is 10,549 pharmacophores.** Everything in this
repository reads, writes, predicts and scores 10,549. Nothing reports any other
width.

Written 2026-08-23. This document is the specification a reviewer should check
the code against, and it carries the commands to check it with.

---

## 1. The authority

`pfpkey.c`, in `FoodWellness/pfp`, is the reference implementation.

```c
#define NPHARM 10549
#define NINT   330
...
fingerprint[tempi/32] |= (1 << 31-tempi%32);      /* line 817 */
```

Three facts follow, and all three matter.

1. **There are 10,549 pharmacophores.** `pharm10549.list` in the same
   directory has exactly 10,549 lines, one per pharmacophore. The independent
   enumeration in `chip-sandbox-orforglipron-v4/code/pfp_bits.py`, which
   mirrors `SetPharmNum`, also produces 10,549.

2. **The file format is 330 unsigned 32 bit words.** 330 words is what it
   takes to carry 10,549 bits. The `.pfp` files on disk are correct and no
   corpus needs regenerating. 330 is a storage fact, never a fingerprint width.

3. **Pharmacophore `i` is stored at bit `31 - i%32` of word `i/32`.** Within a
   word the pharmacophores run from the most significant bit DOWN.

## 2. The consequence that is easy to get wrong

Unpacking a little endian word LSB first, which is what every reader here does,
puts pharmacophore `i` at array position

```
    (i//32)*32 + 31 - (i%32)
```

so the order is reversed inside each word. Word 329 carries pharmacophores
10528 to 10548 in its top 21 bits, which means:

- the 11 positions that carry **no** pharmacophore are array positions
  **10528 to 10538**;
- array positions **10539 to 10559 are real pharmacophores**.

**A `[:10549]` slice is therefore wrong.** It keeps the 11 positions that mean
nothing and discards 11 real pharmacophores. The mapping must be explicit.

## 3. Where the mapping lives

One definition, in `pfp-surrogate/code/train_surrogate_1038.py`:

```python
N_INTS   = 330
N_PHARM  = 10549
N_BITS   = N_PHARM                     # the width of a fingerprint

_k = np.arange(N_INTS * 32)
_p = (_k // 32) * 32 + 31 - (_k % 32)
PHARM_COL = np.where(_p < N_PHARM, _p, -1)   # packed position -> pharmacophore
_j = np.arange(N_PHARM)
PACK_POS  = (_j // 32) * 32 + 31 - (_j % 32) # pharmacophore -> packed position
```

`PHARM_COL` is applied when reading a `.pfp`. `PACK_POS` is applied to a model
output that was produced at the packed width, and to `unpack`/`pack` in the
libraries. The two are inverses; a test asserts it.

Column `j` is pharmacophore `j` everywhere downstream. That is what makes
`pfp_bits.py` able to decode a column back to a pharmacophore.

## 4. Verify it

```bash
cd /Users/smuskal/pfp-libraries/pfp-surrogate
PY=/Users/smuskal/miniforge3-arm64/envs/ai-steve-face/bin/python

# the map is self consistent
$PY -c "import sys;sys.path.insert(0,'code')
from train_surrogate_1038 import N_BITS,PHARM_COL,PACK_POS
import numpy as np
assert N_BITS==10549
assert list(np.nonzero(PHARM_COL<0)[0])==list(range(10528,10539))
assert np.all(PHARM_COL[PACK_POS]==np.arange(10549))
print('map OK')"

# the library round trips real pfpall words
$PY -c "import sys,gzip,glob
sys.path.insert(0,'/Users/smuskal/pfp-libraries/pharmcast-release/src')
from pharmcast import bits as B
f=sorted(glob.glob('/Users/smuskal/pfp-libraries/enamine-screening-202606/chunks/chunk_*.pfp.gz'))[0]
w=[int(v) for v in next(l.split() for l in gzip.open(f,'rt') if len(l.split())>=331)[1:331]]
assert B.pack(B.unpack(w))==w
print('round trip OK')"

# the released package
cd /Users/smuskal/pfp-libraries/pharmcast-release && $PY -m pytest tests -q

# nothing in live source mentions the packed width as a fingerprint width
cd /Users/smuskal/pfp-libraries
grep -rnE "330 ?\* ?32|10560|10,560" . | grep -E "\.(py|sh)(:|$)" \
  | grep -v "\.bak_|/archive/|\.retired_|__pycache__"
```

The last command should return nothing but deliberate references to the
storage word count.

## 5. Current statistics, on this basis

Recomputed on the fixed 9,975 molecule validation split with
`code/agreement_10549.py`. Median per molecule MCC:

| Model | Training molecules | Median MCC |
|---|---|---|
| PharmCast SP v2 | 1,724,833 | 0.891 |
| PharmCast SP v3 | 2,114,521 | 0.893 |
| PharmCast SP v4 | 2,597,479 | 0.895 |
| PharmCast SCP v5 | 2,888,503 | 0.894 |

Similarity statistics are unchanged by the width, because Tanimoto is an
intersection over a union and is invariant to a permutation of positions and to
positions that are zero in both fingerprints.

## 6. SCP v7

**SCP v7 trains at 10,549.** `train_combined.py` now reads through `PHARM_COL`,
so a fresh run produces a 10,549 wide output layer with no further change.

SCP v6 was started before this and has a wider output layer. It is allowed to
finish. Evaluation maps it through `PACK_POS`, so its reported numbers are on
the 10,549 basis like every other model. A `--resume` of SCP v6 against the
current reader will fail loudly on a shape mismatch rather than silently
mixing conventions; start it fresh if it needs restarting.
