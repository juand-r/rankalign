#!/bin/bash
set -e

SRC=/datastor2/jocelyn/rankalign/models
DST=/datastor1/jdr/gv-gap/rankalign/models

MODELS=(
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--full-completion--force-same-x--labelonly0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--full-completion--force-same-x--semi0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--force-same-x--labelonly0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--force-same-x--semi0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1_merged"
  "v6-google--gemma-2-9b-it-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1_merged"
)

TOTAL=${#MODELS[@]}
i=0

for m in "${MODELS[@]}"; do
  i=$((i + 1))
  if [ -d "$DST/$m" ]; then
    echo "[$i/$TOTAL] SKIP (already exists): $m"
    continue
  fi
  echo "[$i/$TOTAL] Copying: $m"
  cp -r "$SRC/$m" "$DST/$m"
  echo "         Done."
done

echo ""
echo "All $TOTAL models processed."
