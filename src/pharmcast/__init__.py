"""PharmCast -- 3D pharmacophore fingerprints predicted from 2D structure.

    from pharmcast import PharmCast, pharmtan, pharmsim
    pc = PharmCast.load("PharmcastSCP.22August2026.pt")
    a, b = pc.words_batch(["CCO", "c1ccccc1O"])
    pharmtan(a, b)

Batch. `words_batch` is the API that matters; a one-molecule call is dominated
by featurisation overhead.
"""
from .bits import (N_BITS, N_INTS, N_PHARM, from_bitstring, native_index, pack,
                   popcount, set_bits, to_bitstring, unpack)
from .io import read_pfp, read_smiles, write_pfp
from .model import PharmCast
from .similarity import pharmsim, pharmtan, pharmtan_matrix, tanimoto

__version__ = "0.1.0"

__all__ = [
    "PharmCast",
    "pharmtan", "pharmsim", "pharmtan_matrix", "tanimoto",
    "unpack", "pack", "set_bits", "popcount", "to_bitstring",
    "from_bitstring", "native_index",
    "read_pfp", "write_pfp", "read_smiles",
    "N_INTS", "N_BITS", "N_PHARM", "__version__",
]
