#!/bin/sh
# Task 40 downloads (2026-11-26): DEMETER2 combined RNAi (figshare 6025238) and TCGA HiSeqV2 (UCSC Xena TCGA hub) for candidate cohorts.
E="$(cd "$(dirname "$0")/.." && pwd)/external_data"
cd "$E/demeter2_rnai"
for f in "13515395 D2_combined_gene_dep_scores.csv" "13515401 D2_combined_CL_data.csv" "13515380 README.txt"; do set -- $f; curl -sSL --fail -o "$2" "https://ndownloader.figshare.com/files/$1"; done
echo "Task 40 (downloaded 2026-11-26, never analysed): DEMETER2 combined RNAi gene dependency (Achilles+DRIVE+Marcotte), McFarland et al. 2018, figshare article 6025238." > SOURCE.txt
sha256sum * > SHA256SUMS.txt
cd "$E/tcga_rnaseq_xena"
for c in BRCA KIRC LUAD UCEC THCA HNSC LGG PRAD LUSC STAD BLCA LIHC OV; do curl -sSL --fail -o "TCGA.$c.HiSeqV2.gz" "https://tcga.xenahubs.net/download/TCGA.$c.sampleMap/HiSeqV2.gz"; done
echo "Task 40 (downloaded 2026-11-26, never analysed): TCGA IlluminaHiSeq RNASeqV2 gene expression, log2(RSEM norm count+1), UCSC Xena TCGA hub https://tcga.xenahubs.net/download/TCGA.<cohort>.sampleMap/HiSeqV2.gz" > SOURCE.txt
sha256sum * > SHA256SUMS.txt
du -sb "$E/demeter2_rnai" "$E/tcga_rnaseq_xena"
