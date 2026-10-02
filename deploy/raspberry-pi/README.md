# AeroMind edge on Raspberry Pi 5

> **Status: prepared, NOT validated on hardware.** Nothing in this folder has been run on a Raspberry Pi
> yet. No Raspberry Pi latency, memory or throughput figure exists in this repository. The benchmark
> command below produces them; until someone runs it on a Pi, the results are **PENDING**.

## What runs where

```
 LAPTOP (ground station)                         RASPBERRY PI 5 (edge)
 aeromind serve   <------ HTTP advisories ------ aeromind edge-agent
 dashboard, decisions, LLM copilot                same EdgePipeline + ONNX Runtime
                                                  sensor health -> features -> anomaly
                                                  -> fault -> RUL -> physics evidence -> advisory
```

- The Pi runs the **same pipeline code** as the simulated fleet (`aeromind.edge.pipeline.EdgePipeline`)
  and the exported ONNX bundle. There is no second inference implementation.
- The **LLM stays on the ground side** (the laptop). Nothing LLM-related is installed on the Pi.
- Only advisories (a few hundred bytes of JSON) cross the network, never raw sensor data.
- Sensor input on the Pi is currently the in-repo **simulator**. A real sensor adapter is not implemented.

## System requirements

| | |
|---|---|
| Board | Raspberry Pi 5 (4 GB or 8 GB), active cooling recommended for sustained load |
| OS | 64-bit Raspberry Pi OS (Bookworm, aarch64) with Python 3.10 to 3.12 |
| Storage | about 1 GB free (Python packages dominate; the model bundle is about 2 MB) |
| Network | reachable from the laptop on the ground-station port (default 8000) |

## ARM64 considerations

- Use a **64-bit** OS. `uname -m` must print `aarch64`.
- `numpy`, `scipy`, `scikit-learn` and `onnxruntime` publish aarch64 wheels on PyPI for current Python
  versions, so a plain `pip install` does not compile anything. If pip starts compiling, you are on a
  32-bit OS or an unsupported Python version.
- The edge needs only the `edge` extra (`onnxruntime`). Do **not** install `torch`, `hummingbird-ml`
  or the `app` extra on the Pi; model export and the ground station run on the laptop.
- ONNX Runtime uses the CPU execution provider. The bundle's sessions are configured for one thread
  (`edge/onnx_export.py`). One thread keeps latency predictable for a single aircraft stream; whether
  more threads help on the Pi 5 has not been measured.
- Models are exported as ONNX-ML tree operators. The Hummingbird/TensorRT export is for NVIDIA GPUs and
  is not relevant to the Pi.

## Install

On the Pi:

```bash
git clone <your repository URL> aeromind && cd aeromind
bash deploy/raspberry-pi/install.sh        # creates .venv, installs aeromind with the "edge" extra
```

On the laptop, export the model once and copy it over:

```bash
python -m aeromind serve          # first start trains and writes artifacts/onnx-fleet (then Ctrl+C), or:
python -m aeromind export-onnx --model artifacts/bundle.joblib --out artifacts/onnx-fleet
scp -r artifacts/onnx-fleet <user>@<pi-address>:~/aeromind/artifacts/
```

The model folder holds `anomaly.onnx`, `classifier.onnx`, `rul.onnx` and `manifest.json` (SHA-256 hashes;
`OnnxBundle` refuses a file that does not match).

## Network configuration

1. On the laptop, start the ground station so the Pi can reach it, with an ingest token:

   ```bash
   export AEROMIND_INGEST_TOKEN="choose-a-long-random-string"
   python -m aeromind serve --host 0.0.0.0 --port 8000
   ```

   `--host 0.0.0.0` exposes the dashboard to your LAN. Do this only on a network you trust, keep the
   token set, and allow port 8000 through the laptop firewall. The ground station has no login.
2. Find the laptop's address (`ipconfig` on Windows) and check from the Pi: `curl http://<laptop-ip>:8000/api/llm/status`.
3. On the Pi, put the same token in the environment (see `edge.env.example`).

## Run the edge agent

```bash
source .venv/bin/activate
export AEROMIND_INGEST_TOKEN="the same string"
aeromind edge-agent --ground http://<laptop-ip>:8000 --tail VT-PI01 --rate 4
```

Useful variations (everything injected is simulated):

```bash
aeromind edge-agent --ground http://<laptop-ip>:8000 --tail VT-PI01 --inject bearing_wear --inject-after 40
aeromind edge-agent --ground http://<laptop-ip>:8000 --tail VT-PI01 --sensor-fault stuck:temperature
aeromind edge-agent --tail VT-PI01 --max-windows 200 --rate 0 --out advisories.jsonl   # offline, no ground station
```

The device appears under "Remote edge devices" in the ground station; its advisories get a maintenance
decision, a work-order draft and an ACARS-sized text like any other, and the copilot can explain them.
If the ground station is unreachable the agent queues up to 200 advisories and sends them when the link returns.

To start on boot, install `aeromind-edge.service` (edit paths, user and the address in it first):

```bash
sudo cp deploy/raspberry-pi/aeromind-edge.service /etc/systemd/system/
sudo cp deploy/raspberry-pi/edge.env.example /etc/aeromind-edge.env   # then edit it
sudo systemctl daemon-reload && sudo systemctl enable --now aeromind-edge
journalctl -u aeromind-edge -f
```

## Benchmark (produces the real numbers)

On the Pi:

```bash
aeromind bench --target raspberry-pi --onnx-dir artifacts/onnx-fleet -n 1000
```

This writes `artifacts/bench/raspberry-pi.json` with model size, model load time, per-model and
end-to-end per-window latency, throughput and memory. The command **refuses** to label a result
`raspberry-pi` unless the device tree reports a Raspberry Pi, so a laptop run cannot be mistaken for one.
On the laptop run `aeromind bench --target laptop`, copy the Pi's JSON next to it, and compare:

```bash
aeromind bench --compare artifacts/bench/laptop.json artifacts/bench/raspberry-pi.json
```

| Metric | Laptop CPU | Raspberry Pi 5 |
|---|---|---|
| Model size, load time | run `--target laptop` | **PENDING** (hardware not yet run) |
| Per-model and end-to-end latency, throughput | run `--target laptop` | **PENDING** |
| Memory | run `--target laptop` | **PENDING** |

Only paste numbers here that came out of that JSON.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `no exported model at artifacts/onnx-fleet` | Copy the folder from the laptop (see Install), or pass `--train-if-missing` (slow on a Pi). |
| `pip` compiles for minutes or fails | 32-bit OS or unsupported Python. Use 64-bit Raspberry Pi OS and Python 3.10 to 3.12. |
| `ModuleNotFoundError: onnxruntime` | `pip install -e ".[edge]"` inside the venv. |
| `failed posts` rises in the summary | The laptop is unreachable: wrong IP, firewall, or `serve` bound to 127.0.0.1. Test with `curl`. Advisories are queued meanwhile. |
| HTTP 401 from `/api/ingest` | `AEROMIND_INGEST_TOKEN` differs between laptop and Pi. |
| `refusing to label this run 'raspberry-pi'` | You are not on a Pi. This is intended. |
| Slowdowns after minutes | Thermal throttling: check `vcgencmd get_throttled`; add a heatsink and fan. |
