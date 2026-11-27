#!/bin/sh
# Task 48 downloads (2026-11-27): 10x Genomics FACS-purified PBMC populations (Zheng et al. 2017, Nat Commun 8:14049), filtered gene-barcode matrices.
E="$(cd "$(dirname "$0")/.." && pwd)/external_data"; cd "$E"
for c in b_cells cd14_monocytes cd34 cd4_t_helper cd56_nk cytotoxic_t memory_t naive_t naive_cytotoxic regulatory_t; do
  curl -sSL --fail -o "${c}.tar.gz" "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/$c/${c}_filtered_gene_bc_matrices.tar.gz" || echo "fail $c"
done
echo "Task 48 (downloaded 2026-11-27, never analysed): 10x Genomics purified PBMC populations, Cell Ranger 1.1.0 filtered matrices, https://cf.10xgenomics.com/samples/cell-exp/1.1.0/<type>/<type>_filtered_gene_bc_matrices.tar.gz" > SOURCE.txt
sha256sum *.tar.gz > SHA256SUMS.txt; du -sb "$E"
