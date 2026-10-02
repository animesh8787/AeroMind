"""Edge agent: the SAME ``EdgePipeline`` the ground-station fleet runs, as a standalone process.

Intended for a small ARM64 board (e.g. Raspberry Pi 5) next to the aircraft's sensors, sending only
advisories to the ground station over HTTP. It reuses the pipeline and the ONNX bundle unchanged;
there is no second inference implementation.

Sensor input is currently the in-repo simulator (``LiveAircraft``). A real sensor adapter is not
implemented. Nothing here has been measured on a Raspberry Pi unless a benchmark file says so.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..core.simulator import LiveAircraft
from ..edge.onnx_export import OnnxBundle
from ..edge.pipeline import EdgePipeline
from ..edge.sensor_health import CHANNELS, SENSOR_FAULT_KINDS, SensorFault, SensorFaultInjector

BACKLOG = 200  # advisories kept while the ground station is unreachable (store and forward)


@dataclass
class AgentStats:
    windows: int = 0
    advisories: int = 0
    sent: int = 0
    failed_posts: int = 0
    backlog: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class GroundLink:
    """HTTP POST of advisories to ``/api/ingest`` with a small store-and-forward queue."""

    def __init__(self, base_url: str | None, token: str | None = None, timeout: float = 3.0):
        self.base = base_url.rstrip("/") if base_url else None
        self.token = token if token is not None else os.environ.get("AEROMIND_INGEST_TOKEN")
        self.timeout = timeout
        self.queue: deque[dict] = deque(maxlen=BACKLOG)

    def _post(self, body: dict) -> None:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-AeroMind-Token"] = self.token
        req = urllib.request.Request(self.base + "/api/ingest", data=json.dumps(body).encode(), headers=headers,
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (operator-supplied URL)
            r.read()

    def send(self, body: dict, stats: AgentStats) -> None:
        if not self.base:
            return
        self.queue.append(body)
        while self.queue:
            try:
                self._post(self.queue[0])
            except urllib.error.HTTPError as e:
                if e.code in (400, 401):  # will never succeed: drop it rather than block the queue
                    self.queue.popleft()
                stats.failed_posts += 1
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                stats.failed_posts += 1
                break
            self.queue.popleft()
            stats.sent += 1
        stats.backlog = len(self.queue)


def run_agent(onnx_dir: str | Path, tail: str, ground: str | None, *, rate: float = 4.0, max_windows: int | None = None,
              inject: str | None = None, inject_after: int = 40, life: int = 320, sensor_fault: str | None = None,
              out: str | Path | None = None, seed: int = 4242, log=print) -> AgentStats:
    """Run the pipeline on a simulated sensor stream and publish advisories. ``rate`` is windows per second (0 = max)."""
    bundle = OnnxBundle(onnx_dir)
    pipe = EdgePipeline(bundle, component="ENG1-GEARBOX")
    sim = LiveAircraft(seed)
    link = GroundLink(ground)
    stats = AgentStats()
    injectors: list[SensorFaultInjector] = []
    sink = open(out, "a", encoding="utf-8") if out else None
    version = bundle.manifest.get("version", "")
    try:
        while max_windows is None or stats.windows < max_windows:
            t0 = time.perf_counter()
            if inject and stats.windows == inject_after:
                sim.inject(inject, life)
                log(f"[edge] t={stats.windows}: injected simulated {inject}")
            if sensor_fault and stats.windows == inject_after:
                kind, channel = sensor_fault.split(":")
                injectors.append(SensorFaultInjector(SensorFault(kind, channel, start=sim.t), seed=sim.t))
                log(f"[edge] t={stats.windows}: injected simulated sensor fault {sensor_fault}")
            w, _ = sim.step()
            for inj in injectors:
                w = inj.apply(w)
            adv = pipe.process(w)
            stats.windows += 1
            stats.latencies_ms.append((time.perf_counter() - t0) * 1000)
            events = [("sensor", sa) for sa in pipe.last_sensor_advisories] + ([("component", adv)] if adv else [])
            for kind, a in events:
                stats.advisories += 1
                body = {"tail": tail, "kind": kind, "advisory": a.to_dict(), "model_version": str(version)}
                if sink:
                    sink.write(json.dumps(body) + "\n")
                    sink.flush()
                link.send(body, stats)
                log(f"[edge] {kind} advisory at {a.flight_hours:.1f} FH: " +
                    (f"{a.fault_type} {a.confidence:.2f}, RUL p50 {a.rul_hours_p50:.0f} h" if kind == "component"
                     else f"{a.channel} {a.status}"))
            if rate > 0:
                time.sleep(max(0.0, 1.0 / rate - (time.perf_counter() - t0)))
    finally:
        if sink:
            sink.close()
    return stats


def parse_sensor_fault(spec: str) -> str:
    kind, _, channel = spec.partition(":")
    if kind not in SENSOR_FAULT_KINDS or channel not in CHANNELS:
        raise ValueError(f"--sensor-fault must be KIND:CHANNEL, kinds {SENSOR_FAULT_KINDS}, channels {CHANNELS}")
    return spec


__all__ = ["AgentStats", "GroundLink", "run_agent", "parse_sensor_fault"]
