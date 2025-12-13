#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Conda aktivieren
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ehrapy_ml

python src/00_build_anndata_cli.py \
  --ops "Original Daten/HLM Operationen.csv" \
  --patients "Original Daten/Patient Master Data.csv" \
  --aki "Original Daten/AKI Label.csv" \
  --out "h5ad/ops_with_age_groups.h5ad"
