#!/usr/bin/env bash
# Re-run the CPU research probes locally (no GPU). Logs -> outputs/probes/*.log
# Needs: MEDSG_DATA_DIR (flattened HDA folder), models/face_landmarker.task, Stage 1-3 outputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
UV="uv run --no-project --python 3.12 --with numpy --with pandas --with mediapipe --with opencv-python-headless python"
mkdir -p outputs/probes
for p in extract_lm subspace lcurve rrr hetero pattern warp_control gain; do
  echo "== $p"; $UV tools/probes/$p.py 2>&1 | grep -v -i "warning\|W0000\|I0000\|absl\|^INFO" | tee outputs/probes/$p.log
done
