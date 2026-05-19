#!/usr/bin/env bash
# DATASETS_DOWNLOAD.sh
# Documented manifest of every dataset used in the MambaGuard paper.
#
# THIS SCRIPT IS A MANIFEST. The commands below show the canonical URLs
# and DOIs; you may need to adapt them depending on portal availability.
# Datasets that require manual registration are clearly labelled.
#
# Usage:
#   bash reproducibility/DATASETS_DOWNLOAD.sh        # default: data/raw
#   DATA_ROOT=/scratch/data bash reproducibility/DATASETS_DOWNLOAD.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/raw}"
mkdir -p "$DATA_ROOT"

# ----------------------------------------------------------------------------
# 1) IIS3D  --  Inter-Agent Inter-System 3D protocol traces (own dataset).
#    Open access on Zenodo.
# ----------------------------------------------------------------------------
echo "[IIS3D] (open, Zenodo DOI: 10.5281/zenodo.19129512)"
mkdir -p "$DATA_ROOT/iis3d"
curl -L "https://zenodo.org/records/19129512/files/iis3d.tar.gz" \
    -o "$DATA_ROOT/iis3d/iis3d.tar.gz"
tar -xzf "$DATA_ROOT/iis3d/iis3d.tar.gz" -C "$DATA_ROOT/iis3d"

# ----------------------------------------------------------------------------
# 2) AgentDojo  --  open benchmark for tool-using LLM agents.
# ----------------------------------------------------------------------------
echo "[AgentDojo] (open, GitHub)"
mkdir -p "$DATA_ROOT/agentdojo"
git clone --depth 1 https://github.com/ethz-spylab/agentdojo.git \
    "$DATA_ROOT/agentdojo/repo"

# ----------------------------------------------------------------------------
# 3) InjecAgent  --  open prompt-injection benchmark for agents.
# ----------------------------------------------------------------------------
echo "[InjecAgent] (open, GitHub)"
mkdir -p "$DATA_ROOT/injectagent"
git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git \
    "$DATA_ROOT/injectagent/repo"

# ----------------------------------------------------------------------------
# 4) TGB 2.0  --  Temporal Graph Benchmark v2 (open, Hugging Face).
# ----------------------------------------------------------------------------
echo "[TGB 2.0] (open, HuggingFace)"
mkdir -p "$DATA_ROOT/tgb"
# Datasets are auto-downloaded by the `tgb` Python package on first use;
# this is a no-op placeholder.
echo "  -> use:  python -c 'from tgb.linkproppred.dataset import LinkPropPredDataset; LinkPropPredDataset(name=\"tgbl-wiki\", root=\"$DATA_ROOT/tgb\")'"

# ----------------------------------------------------------------------------
# 5) CIC-IDS-2018  --  REQUIRES MANUAL REGISTRATION.
#    https://www.unb.ca/cic/datasets/ids-2018.html
# ----------------------------------------------------------------------------
echo "[CIC-IDS-2018] (MANUAL REGISTRATION REQUIRED)"
echo "  Visit: https://www.unb.ca/cic/datasets/ids-2018.html"
echo "  Place archive at: $DATA_ROOT/cic-ids-2018/CSE-CIC-IDS2018.zip"

# ----------------------------------------------------------------------------
# 6) UNSW-NB15  --  REQUIRES MANUAL REGISTRATION.
#    https://research.unsw.edu.au/projects/unsw-nb15-dataset
# ----------------------------------------------------------------------------
echo "[UNSW-NB15] (MANUAL REGISTRATION REQUIRED)"
echo "  Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
echo "  Place CSV files at: $DATA_ROOT/unsw-nb15/"

# ----------------------------------------------------------------------------
# 7) ToN_IoT  --  REQUIRES MANUAL REGISTRATION (UNSW Canberra).
#    https://research.unsw.edu.au/projects/toniot-datasets
# ----------------------------------------------------------------------------
echo "[ToN_IoT] (MANUAL REGISTRATION REQUIRED)"
echo "  Visit: https://research.unsw.edu.au/projects/toniot-datasets"
echo "  Place archive at: $DATA_ROOT/ton-iot/"

# ----------------------------------------------------------------------------
# 8) MCP Adversary Bench  --  bundled with this repository.
# ----------------------------------------------------------------------------
echo "[MCP Adversary Bench] (bundled in mambaguard/attacks/templates/)"
echo "  No download required."

echo ""
echo "Manifest complete. Datasets marked MANUAL REGISTRATION will not"
echo "auto-download; please obtain them from the linked portals."
