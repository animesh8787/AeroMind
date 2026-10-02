#!/usr/bin/env bash
# Install the AeroMind edge agent on a 64-bit Raspberry Pi OS. Run from the repository root.
# Installs only the "edge" extra (onnxruntime): no torch, no export tooling, no ground station.
set -euo pipefail

if [ "$(uname -m)" != "aarch64" ]; then
  echo "warning: this is $(uname -m), not aarch64. Prebuilt wheels may be missing on a 32-bit OS." >&2
fi
command -v python3 >/dev/null || { echo "python3 not found (sudo apt install python3 python3-venv)" >&2; exit 1; }

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[edge]"

python -c "import onnxruntime, sklearn, scipy; print('onnxruntime', onnxruntime.__version__)"
echo
echo "Installed. Next: copy an exported model to artifacts/onnx-fleet (see deploy/raspberry-pi/README.md),"
echo "then:  aeromind edge-agent --ground http://<laptop-ip>:8000 --tail VT-PI01"
