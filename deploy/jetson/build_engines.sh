#!/usr/bin/env bash
# Build TensorRT engines from AeroMind's TensorRT-compatible ONNX export.
# NOT YET VALIDATED ON HARDWARE.
set -euo pipefail
SRC=${1:-artifacts/onnx-trt}
OUT=${2:-engines}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
mkdir -p "$OUT"
F=$(python3 -c "import json;m=json.load(open('$SRC/manifest.json'));print(len(m['feature_names']))")
R=$(python3 -c "import json;m=json.load(open('$SRC/manifest.json'));s=m['rul_input']['shape'];print('x'.join(map(str,s)))")
RIN=$(python3 -c "import json;print(json.load(open('$SRC/manifest.json'))['rul_input']['name'])")
for prec in fp32 fp16; do
  flag=""; [ "$prec" = fp16 ] && flag="--fp16"
  "$TRTEXEC" --onnx="$SRC/anomaly.onnx" --minShapes=features:1x$F --optShapes=features:1x$F --maxShapes=features:64x$F $flag --saveEngine="$OUT/anomaly_$prec.plan"
  "$TRTEXEC" --onnx="$SRC/classifier.onnx" --minShapes=features:1x$F --optShapes=features:1x$F --maxShapes=features:64x$F $flag --saveEngine="$OUT/classifier_$prec.plan"
  "$TRTEXEC" --onnx="$SRC/rul.onnx" --minShapes=$RIN:1x$R --optShapes=$RIN:1x$R --maxShapes=$RIN:64x$R $flag --saveEngine="$OUT/rul_$prec.plan"
done
echo "Engines in $OUT (check trtexec timing output; compare outputs before use)."
