# The fingerprint width contract

**A PharmPrint fingerprint is 10,549 pharmacophores.** Everything in this
package reads, writes, predicts and scores 10,549. Nothing reports any other
width.

This document is the specification to check an implementation against, and it
carries the commands to check it with.

---

## 1. The authority

The reference implementation defines:

```c
#define NPHARM 10549          /* pharmacophores */
#define NINT   330            /* uint32 words per record */

fingerprint[i/32] |= (1 << 31 - i%32);
```

Three facts follow, and all three matter.

1. **There are 10,549 pharmacophores.** That is the fingerprint width, and the
   only number that should ever be called a width.

2. **The file format is 330 unsigned 32 bit words.** 330 words is what it takes
   to carry 10,549 bits, with 11 bit slots left over. **330, and the 10,560 bit
   slots it implies, are storage facts. Neither is ever a fingerprint width, a
   model output width, or a number of bits a model predicts.**

3. **Pharmacophore `i` is stored at bit `31 - i%32` of word `i/32`.** Within a
   word the pharmacophores run from the most significant bit DOWN.

## 2. The consequence that is easy to get wrong

Unpacking a little endian word least significant bit first, which is what a
straightforward reader does, puts pharmacophore `i` at array position

```
    (i//32)*32 + 31 - (i%32)
```

so the order is **reversed inside each word**. Word 329 carries pharmacophores
10528 to 10548 in its top 21 bits, which means:

- the 11 positions that carry **no** pharmacophore are array positions
  **10528 to 10538**;
- array positions **10539 to 10559 are real pharmacophores**.

**A `[:10549]` slice is therefore wrong.** It keeps the 11 positions that mean
nothing and discards 11 real pharmacophores. Worse, it leaves every other
position mirrored inside its own word, so a column index is not a pharmacophore
number. The mapping must be explicit.

## 3. Where the mapping lives

One definition, in `pharmcast.bits`:

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
output produced at the packed width, and to `unpack` and `pack`. The two are
inverses; a test asserts it.

Column `j` is pharmacophore `j` everywhere downstream. That is what lets a
column be decoded back to a pharmacophore.

## 4. Two checkpoint shapes

Both exist and both must work.

- A checkpoint whose output layer was sized to the packed word count emits
  10,560 columns. Map it with `PACK_POS`.
- A checkpoint trained at the pharmacophore width emits 10,549 columns and is
  already in pharmacophore order. Leave it alone.

`PharmCast._bits` branches on the checkpoint's own width and raises on any
third width rather than guessing.

## 5. Verify it

Run these from the repository root, with `pharmcast` installed or `src` on the
path.

```bash
# the map is self consistent
python -c "
import numpy as np
from pharmcast.bits import N_PHARM, N_BITS, PHARM_COL, PACK_POS
assert N_BITS == N_PHARM == 10549
assert list(np.nonzero(PHARM_COL < 0)[0]) == list(range(10528, 10539))
assert np.all(PHARM_COL[PACK_POS] == np.arange(10549))
print('map OK')"

# the library round trips real words
python -c "
from pharmcast import bits as B
words = [0] * 330
words[0], words[329] = 0x80000001, 0xFFFFF800
assert B.pack(B.unpack(words)) == words
print('round trip OK')"

# the test suite
python -m pytest tests -q

# nothing in the source calls the packed width a fingerprint width
grep -rnE "330 ?\* ?32|10560|10,560" src/ | grep -v "storage"
```

The last command should return nothing but deliberate references to the storage
word count.

## 6. Why similarity numbers do not move

Tanimoto is an intersection over a union, computed as a popcount. It is
invariant to a permutation of positions and to positions that are zero in both
fingerprints. Correcting the width and the ordering therefore changes **no**
similarity value. What it changes is bit level identity: which pharmacophore a
given column actually is.

Stored `.pfp` files are unaffected and never need regenerating. Only the
interpretation of eleven indices moves.
