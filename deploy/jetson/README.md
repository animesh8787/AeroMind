# Jetson deployment kit

> **NOT YET VALIDATED ON HARDWARE.** Nothing in this folder has been run on a Jetson. All latency
> and memory figures in the project were measured on a laptop/cloud CPU. Treat these files as a
> starting point for the first hardware session.

## 1. Export TensorRT-compatible graphs (on a PC)

```bash
pip install -e ".[hummingbird]"                       # torch + hummingbird-ml, export machine only
python -m aeromind train --phases --conformal --model artifacts/bundle-fleet.joblib
python -m aeromind export-onnx --model artifacts/bundle-fleet.joblib --trees hummingbird --out artifacts/onnx-trt
```

`--trees hummingbird` rewrites every tree ensemble as plain tensor operations; the exporter refuses any
operator outside TensorRT's ONNX parser support list. Copy `artifacts/onnx-trt/` to the Jetson.

## 2. Build engines on the Jetson

```bash
./build_engines.sh artifacts/onnx-trt engines/          # uses trtexec from JetPack
```

## 3. Run the pipeline on the device

`Dockerfile` builds an image on an L4T JetPack base with ONNX Runtime; mount the model folder and run
`python -m aeromind demo --backend onnx --onnx-dir /models`. Record per-window latency, `tegrastats`
power and memory, and compare outputs with `check_parity` before trusting the engines.

## What to measure first

1. Do all three graphs parse and build in FP32 and FP16?
2. Per-window latency (feature extraction on the CPU + inference) at 1 window/s and at burst rate.
3. FP16 vs FP32 parity on recorded windows (`onnx_export.check_parity`).
4. Power draw in the 7 W / 15 W modes.
