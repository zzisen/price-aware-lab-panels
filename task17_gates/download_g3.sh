#!/bin/bash
# Task 17 G3: GTEx v8 tissue TPM files, public. One new folder; data only, no scripts inside.
set -e
D="$(cd "$(dirname "$0")/.." && pwd)/external_data/gtex_v8_tissues"
mkdir -p "$D"
BASE="https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue"
for t in muscle_skeletal skin_sun_exposed_lower_leg adipose_subcutaneous artery_tibial thyroid nerve_tibial lung esophagus_mucosa heart_left_ventricle colon_transverse breast_mammary_tissue pancreas; do
  f="gene_tpm_2017-06-05_v8_$t.gct.gz"
  [ -s "$D/$f" ] || curl -sS --fail -m 900 -o "$D/$f" "$BASE/$f"
  echo "$t $(stat -c %s "$D/$f")"
done
echo "Downloaded $(date +%F) for Task 17 (plan_gates) from $BASE" > "$D/SOURCE.txt"
(cd "$D" && sha256sum *.gct.gz > SHA256SUMS.txt)
