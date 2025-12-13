#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Conda aktivieren
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ehrapy_ml

python src/validate_dates_and_schema.py \
  --ops "Original Daten/HLM Operationen.csv" \
  --patients "Original Daten/Patient Master Data.csv" \
  --aki "Original Daten/AKI Label.csv" \
  --features "Daten/ops_with_crea_cysc_vis_features.csv" \
  --x-features crea_delta_0_48 crea_rate_0_48 vis_auc_0_24 vis_auc_0_48 duration_minutes \
  --dayfirst
