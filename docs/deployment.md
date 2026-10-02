# Deployment

Three setups exist. Only the first has been run end to end.

| Setup | Status |
|---|---|
| Ground station + simulated fleet on one machine (Windows / Linux / macOS) | **Implemented and tested** |
| Ground station on a laptop + edge agent on another machine over HTTP | **Implemented and tested on one machine (loopback); not run across two devices** |
| Edge agent on a Raspberry Pi 5 | **Prepared, hardware validation PENDING** ([../deploy/raspberry-pi/README.md](../deploy/raspberry-pi/README.md)) |
| Jetson with TensorRT | Kit exists ([../deploy/jetson/README.md](../deploy/jetson/README.md)); **not validated on hardware** |

## One machine (ground station + simulated fleet)

```bash
pip install -e ".[app]"
python -m aeromind serve            # http://127.0.0.1:8000, works offline
```

The first start trains a flight-phase model (about 20 s) into `artifacts/onnx-fleet`. Keys and signed packages for the
over-the-air demo are created under `artifacts/ground` (git-ignored).

## Ground station reachable from other machines

```bash
export AEROMIND_INGEST_TOKEN="a-long-random-string"
python -m aeromind serve --host 0.0.0.0 --port 8000
```

The ground station has **no login**; it is meant for localhost or a trusted network. `/api/ingest` checks the token
(`X-AeroMind-Token`) when `AEROMIND_INGEST_TOKEN` is set. Do not expose it to the internet.

## Edge agent

```bash
aeromind edge-agent --ground http://<ground-station>:8000 --tail VT-PI01 --rate 4
```

It runs the same `EdgePipeline` and ONNX bundle as the simulated fleet, publishes only advisories, queues up to 200 while the
ground station is unreachable, and can write advisories to a JSONL file (`--out`). Sensor input is the simulator.

## LLM

Optional and ground-side only. See [llm.md](llm.md). With no key and no Ollama the copilot uses rule-based templates.

## Benchmarks

```bash
aeromind bench --target laptop                 # saves artifacts/bench/laptop.json
aeromind bench --target raspberry-pi           # run ON the Pi; refused elsewhere
aeromind bench --compare artifacts/bench/laptop.json artifacts/bench/raspberry-pi.json
```

Only results that came out of these commands may be quoted.
