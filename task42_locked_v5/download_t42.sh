#!/bin/sh
# Task 42 downloads (2026-11-26): TCGA HiSeqV2 for further cohorts (candidates expected to have >= 150 primary tumours), into a new folder.
E="$(cd "$(dirname "$0")/.." && pwd)/external_data"
mkdir -p "$E"; cd "$E"
for c in KIRP COAD CESC SARC PAAD ESCA PCPG GBM TGCT THYM; do curl -sSL --fail -o "TCGA.$c.HiSeqV2.gz" "https://tcga.xenahubs.net/download/TCGA.$c.sampleMap/HiSeqV2.gz" || echo "fail $c"; done
echo "Task 42 (downloaded 2026-11-26, never analysed): TCGA IlluminaHiSeq RNASeqV2 gene expression, log2(RSEM norm count+1), UCSC Xena TCGA hub https://tcga.xenahubs.net/download/TCGA.<cohort>.sampleMap/HiSeqV2.gz" > SOURCE.txt
sha256sum *.gz > SHA256SUMS.txt
du -sb "$E"
