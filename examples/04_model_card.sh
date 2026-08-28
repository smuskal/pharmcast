#!/usr/bin/env bash
# What does this checkpoint say about itself?
#
# Read this before trusting a score. The card carries the training corpus
# composition, the mass range each corpus covers, and the applicability domain:
# where the model is calibrated and where it is extrapolating.
#
#   ./04_model_card.sh /path/to/pharmcast_scp_v7.pt
set -euo pipefail
MODEL="${1:?usage: ./04_model_card.sh MODEL.pt}"
pharmcast --model "$MODEL" card
