#!/usr/bin/env bash
# Pharmacophore similarity between two molecules, from structure alone.
#
# The point of a pharmacophore fingerprint is that it sees past the scaffold.
# To show that, each pair below is scored BOTH ways: by PharmCast's 3D
# pharmacophore similarity and by ordinary 2D Morgan similarity. A scaffold hop
# is a pair that scores high on the first and low on the second.
#
#   ./02_similarity.sh /path/to/pharmcast_scp_v7.pt
set -euo pipefail
MODEL="${1:?usage: ./02_similarity.sh MODEL.pt}"
cd "$(dirname "$0")"

pair () {                       # label  smilesA  smilesB
  local label="$1" a="$2" b="$3"
  local ph two
  # THREE digits. The library prints four; a fourth digit here is below the
  # model's own measured error (median 0.016 against reference ensembles), so
  # printing it invites a reader to take a difference that is not there.
  ph=$(pharmcast --model "$MODEL" sim "$a" "$b" \
       | awk '/PharmSim/{printf "%.3f", $2}')
  two=$(python - "$a" "$b" <<'PY'
import sys
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
f = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
     for s in sys.argv[1:3]]
print("%.3f" % DataStructs.TanimotoSimilarity(f[0], f[1]))
PY
)
  printf "  %-34s  pharmacophore %-7s  2D %-7s\n" "$label" "$ph" "$two"
}

echo "                                      3D (PharmCast)   2D (Morgan)"
pair "estradiol / diethylstilbestrol" \
     "C[C@]12CC[C@H]3[C@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O" \
     'CC/C(=C(/CC)c1ccc(O)cc1)c1ccc(O)cc1'
pair "aspirin / salicylate" \
     "CC(=O)Oc1ccccc1C(=O)O" "OC(=O)c1ccccc1O"
pair "caffeine / ibuprofen" \
     "CN1C=NC2=C1C(=O)N(C)C(=O)N2C" "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

cat <<'TXT'

  Read the two columns together, not the pharmacophore score alone.
  Estradiol and diethylstilbestrol are unrelated in 2D and bind the same
  receptor: that gap between the columns is the scaffold hop. Aspirin and
  salicylate score similarly in 3D but are also obviously related in 2D, so
  nothing was discovered there. Caffeine and ibuprofen are unrelated both ways.
TXT
