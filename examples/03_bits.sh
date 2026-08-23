#!/usr/bin/env bash
# Expand a fingerprint into ones and zeros.
#
# A .pfp stores 330 unsigned 32-bit integers per molecule, which is compact and
# unreadable. Two routes to the actual bits:
#
#   pharmcast bits   one molecule, straight from SMILES  (needs the model)
#   pfp2bits         a whole .pfp file                   (needs no model)
#
#   ./03_bits.sh /path/to/PharmCastSP.pt
set -euo pipefail
MODEL="${1:?usage: ./03_bits.sh MODEL.pt}"
cd "$(dirname "$0")"

echo "One molecule, from SMILES, as a bit string:"
pharmcast --model "$MODEL" bits "CC(=O)Nc1ccc(O)cc1" --width 120

echo
echo "The same molecule, as the positions that are set:"
pharmcast --model "$MODEL" bits "CC(=O)Nc1ccc(O)cc1" --format positions

echo
echo "A whole .pfp file, with no model loaded:"
pharmcast --model "$MODEL" fp --in data/pair.smi --out /tmp/pair.pfp
pfp2bits /tmp/pair.pfp --width 100

echo
echo "The same records as JSON, metadata included:"
pfp2bits /tmp/pair.pfp --format positions --json | head -2
