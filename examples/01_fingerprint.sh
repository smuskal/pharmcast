#!/usr/bin/env bash
# Fingerprint a file of SMILES into the native .pfp format.
#
# This is the thing PharmCast exists to do: a 3D pharmacophore fingerprint for
# every molecule, straight from 2D structure, with no conformer generation.
#
#   ./01_fingerprint.sh /path/to/PharmCastSP.pt
set -euo pipefail
MODEL="${1:?usage: ./01_fingerprint.sh MODEL.pt}"
cd "$(dirname "$0")"

echo "Fingerprinting examples/data/molecules.smi"
pharmcast --model "$MODEL" fp --in data/molecules.smi --out /tmp/molecules.pfp

echo
echo "The first record, with its descriptor block:"
head -1 /tmp/molecules.pfp | awk '{
  printf "  name        %s\n", $1
  printf "  words       %s %s %s ... (330 in total)\n", $2, $3, $4
  printf "  version     %s\n", $(NF-12)
  printf "  MW          %s\n", $(NF-11)
  printf "  heavy       %s\n", $(NF-10)
  printf "  bits set    %s\n", $(NF-9)
  printf "  rotatable   %s\n", $(NF-8)
  printf "  conformers  %s\n", $(NF-7)
}'
echo
echo "Wrote /tmp/molecules.pfp"
