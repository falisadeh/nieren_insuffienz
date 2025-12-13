#!/usr/bin/env bash
# Baut die AnnData-Datei (.h5ad) für Analysen/ML.
# Nutzung:
#   ./src/2_build_dataset.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-ehrapy_ml}"
CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda initialisation file not found: $CONDA_SH" >&2
  exit 1
fi

# Conda aktivieren
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python src/00_build_anndata_cli.py \
  --ops "Original Daten/HLM Operationen.csv" \
  --patients "Original Daten/Patient Master Data.csv" \
  --aki "Original Daten/AKI Label.csv" \
  --out "h5ad/ops_with_age_groups.h5ad"
