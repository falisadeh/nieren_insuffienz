#!/usr/bin/env bash
# Prüft OPS, PATIENTS, AKI und (optional) FEATURES auf Schema/Datums/X-Features.
# Nutzung:
#   ./scripts/1_check_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Conda aktivieren
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ehrapy_ml

python src/validate_dates_and_schema.py \
  --ops "HLM Operationen.csv" \
  --patients "Patient Master Data.csv" \
  --aki "AKI Label.csv" \
  --features "Daten/ops_with_crea_cysc_vis_features.csv" \
  --x-features crea_delta_0_48 crea_rate_0_48 vis_auc_0_24 vis_auc_0_48 duration_minutes \
  --dayfirst
