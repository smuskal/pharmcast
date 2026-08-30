#!/usr/bin/env bash
# Nearest-neighbour screening: rank a target set against one or more queries.
#
# This is the port of the original PharmPrint tool PharmTanList.x, which took
# ONE query, a cutoff, and a database. This one takes many queries, top-N as
# well as a cutoff, and has no fixed database cap.
#
#   ./05_screen.sh /path/to/pharmcast_scp_v8.pt
#
# Numbered 05 because 03 and 04 were already taken by the bits and model-card
# examples; the developer's brief said 03_screen.sh, but renumbering the
# existing ones would break every link in the README that points at them.
set -euo pipefail
MODEL="${1:?usage: ./05_screen.sh MODEL.pt}"
cd "$(dirname "$0")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/queries.smi" <<'EOF'
C[C@]12CC[C@H]3[C@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O estradiol
CC(=O)Oc1ccccc1C(=O)O aspirin
EOF

cat > "$TMP/targets.smi" <<'EOF'
CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1 diethylstilbestrol
C[C@]12CC[C@H]3[C@H](CCc4cc(O)ccc34)[C@@H]1CCC2=O estrone
OC(=O)c1ccccc1O salicylate
CC(=O)Nc1ccc(O)cc1 paracetamol
CC(C)Cc1ccc(cc1)C(C)C(=O)O ibuprofen
CN1C=NC2=C1C(=O)N(C)C(=O)N2C caffeine
CC(C)(C)NC(=O)[C@@H]1CN(Cc2cccnc2)CCN1C[C@@H](O)C[C@@H](Cc1ccccc1)C(=O)N[C@H]1c2ccccc2C[C@H]1O indinavir
EOF

echo "== top 5 per query, SMILES in and SMILES out"
pharmcast --model "$MODEL" screen \
  --queries "$TMP/queries.smi" --targets "$TMP/targets.smi" --top 5 \
  | grep -v '^#' | cut -f1,3,4,6 | column -t

echo
echo "== the same screen with a cutoff, which is what PharmTanList.x did"
pharmcast --model "$MODEL" screen \
  --queries "$TMP/queries.smi" --targets "$TMP/targets.smi" --cutoff 0.3 \
  | grep -v '^#' | cut -f1,3,4,6 | column -t

echo
echo "== NO MODEL NEEDED once the fingerprints exist"
# A .pfp already holds fingerprints, whatever produced them: pfpall, PharmCast,
# or the original C tools. Screening two of them loads no checkpoint at all.
pharmcast --model "$MODEL" fp --in "$TMP/targets.smi" --out "$TMP/targets.pfp" 2>/dev/null
pharmcast --model "$MODEL" fp --in "$TMP/queries.smi" --out "$TMP/queries.pfp" 2>/dev/null
pharmcast screen --queries "$TMP/queries.pfp" --targets "$TMP/targets.pfp" --top 3 \
  | grep -v '^#' | cut -f1,3,4,6 | column -t

cat <<'TXT'

  Read the ranking, not the absolute number. These are ensemble fingerprints,
  so a value is lower than a 2D Tanimoto on the same pair and the two are not
  on a shared scale.

  Estradiol finds estrone first, which is the same scaffold, and
  diethylstilbestrol second, which is not: that second hit is the scaffold hop
  and it is the reason to run this at all.

  The last block used no --model. Screening is a comparison between
  fingerprints; where they came from is recorded in the output header and never
  inferred, so a real pfpall library and a predicted one screen the same way.
TXT
