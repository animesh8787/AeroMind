"""AeroMind ground station: FastAPI + WebSocket over a simulated fleet.

Run with ``python -m aeromind serve`` and open http://127.0.0.1:8000. Works offline.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from multiprocessing import get_context
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..core.config import FAULT_MODES
from ..llm import Copilot, CopilotRequest, load_dotenv
from ..llm.copilot import UnknownAircraft, UnknownTask
from ..security.signing import ModelSlots, PackageRejected, generate_keypair, sign_package
from .fleet import SENSOR_PRESETS, Fleet
from .remote import IngestError, RemoteRegistry

DEFAULT_MODEL_DIR = "artifacts/onnx-fleet"


def ensure_model(model_dir: str | Path) -> Path:
    """The phase-aware ONNX model the fleet runs; trained and exported on first start (~20 s)."""
    d = Path(model_dir)
    if not (d / "manifest.json").exists():
        from ..edge.onnx_export import export_onnx
        from ..core.train import TrainConfig, train

        print(f"No model at {d}: training a flight-phase model with conformal RUL (about 20 s)...", flush=True)
        export_onnx(train(TrainConfig(phases=True, conformal=True)), d)
    return d


def _keys(workdir: Path) -> tuple[bytes, bytes]:
    priv_f, pub_f = workdir / "keys" / "ground_ed25519.key", workdir / "keys" / "ground_ed25519.pub"
    if not priv_f.exists():
        priv_f.parent.mkdir(parents=True, exist_ok=True)
        priv, pub = generate_keypair()
        priv_f.write_bytes(priv)
        pub_f.write_bytes(pub)
    return priv_f.read_bytes(), pub_f.read_bytes()


def compute_roi_job(model_dir: str) -> dict:
    """Fleet ROI from a quick measured evaluation of the model in ``model_dir`` (runs in a worker process)."""
    from ..evaluation.evaluate import evaluate
    from ..edge.onnx_export import OnnxBundle
    from ..maintenance.roi import Assumptions, sensitivity, simulate

    ev = evaluate(OnnxBundle(model_dir), runs_per_mode=3, healthy_runs=3, phases=True)
    r = simulate(ev, Assumptions())
    return {"status": "ready", "policies": r["policies"], "sensitivity": sensitivity(ev),
            "assumptions": r["assumptions"], "evaluation_runs_per_mode": 3}


class FleetSource:
    """What the copilot may read: the simulated fleet plus any remote edge devices (structured outputs only)."""

    def __init__(self, gs: "GroundStation"):
        self.gs = gs

    def tails(self) -> list[str]:
        return [*self.gs.fleet.aircraft, *self.gs.remote.aircraft]

    def detail(self, tail: str) -> dict:
        if tail in self.gs.fleet.aircraft:
            return self.gs.fleet.aircraft[tail].detail()
        return self.gs.remote.aircraft[tail].detail()

    def events(self, tail: str) -> list[dict]:
        if tail in self.gs.fleet.aircraft:
            return list(self.gs.fleet.aircraft[tail].events)
        return list(self.gs.remote.aircraft[tail].events)


class GroundStation:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, workdir: str | Path = "artifacts/ground",
                 speed: float = 4.0, seed: int = 4242, compute_roi: bool = False, copilot: Copilot | None = None):
        self.copilot = copilot if copilot is not None else Copilot()
        self.remote = RemoteRegistry()
        self.workdir = Path(workdir)
        self.private_key, self.public_key = _keys(self.workdir)
        self.packages = self.workdir / "packages"
        self.base = ensure_model(model_dir)
        self.slots = ModelSlots(self.public_key, self.workdir / "installed")
        self._next_version = 2
        v1 = self._build_package("v1")
        self.slots.install(v1)
        self.fleet = Fleet(self.slots.active.bundle, speed=speed, seed=seed, model_version="v1")
        self.selected: dict[WebSocket, str] = {}
        self.roi: dict = {"status": "disabled"}
        if compute_roi:
            # A separate process, so the CPU-heavy evaluation never stalls the live demo (GIL).
            self.roi = {"status": "computing"}
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
            fut = self._pool.submit(compute_roi_job, str(self.slots.active.path))
            fut.add_done_callback(self._roi_done)

    def _roi_done(self, fut):
        try:
            self.roi = fut.result()
        except Exception as e:  # never take the demo down
            self.roi = {"status": "failed", "error": str(e)}
        self._pool.shutdown(wait=False)

    def _build_package(self, version: str, tamper: bool = False) -> Path:
        """Copy the base export, sign it as ``version``; optionally corrupt a model file after signing."""
        pkg = self.packages / (version + ("-tampered" if tamper else ""))
        shutil.rmtree(pkg, ignore_errors=True)
        shutil.copytree(self.base, pkg)
        for f in ("manifest.sig",):
            (pkg / f).unlink(missing_ok=True)
        sign_package(pkg, self.private_key, version)
        if tamper:
            f = pkg / "classifier.onnx"
            data = bytearray(f.read_bytes())
            data[len(data) // 2] ^= 0xFF  # one flipped byte after signing
            f.write_bytes(bytes(data))
        return pkg

    def ota(self, variant: str) -> dict:
        version = f"v{self._next_version}"
        pkg = self._build_package(version, tamper=(variant == "tampered"))
        try:
            slot = self.slots.install(pkg)
        except PackageRejected as e:
            self.fleet.log("ota", f"OTA {version} REJECTED: {e}. Fleet stays on {self.fleet.model_version}.",
                           ok=False)
            return {"accepted": False, "reason": str(e), "active": self.fleet.model_version}
        self._next_version += 1
        self.fleet.swap_bundle(slot.bundle, slot.version)
        self.fleet.log("ota", f"OTA {slot.version} verified (Ed25519 + SHA-256) and installed on all aircraft.",
                       ok=True)
        return {"accepted": True, "active": slot.version}

    def rollback(self) -> dict:
        try:
            slot = self.slots.rollback()
        except PackageRejected as e:
            return {"ok": False, "reason": str(e), "active": self.fleet.model_version}
        self.fleet.swap_bundle(slot.bundle, slot.version)
        self.fleet.log("ota", f"Rolled back to {slot.version}.", ok=True)
        return {"ok": True, "active": slot.version}

    async def run(self):
        while True:
            if not self.fleet.paused:
                self.fleet.tick()
            await self.broadcast()
            await asyncio.sleep(1.0 / self.fleet.speed)

    async def broadcast(self):
        if not self.selected:
            return
        snap = {**self.fleet.snapshot(), "remote": [a.summary() for a in self.remote.aircraft.values()]}
        details = {t: self.fleet.aircraft[t].detail() for t in set(self.selected.values()) if t in self.fleet.aircraft}
        for ws, tail in list(self.selected.items()):
            try:
                await ws.send_text(json.dumps({**snap, "detail": details.get(tail)}))
            except Exception:
                self.selected.pop(ws, None)


def create_app(model_dir: str | Path = DEFAULT_MODEL_DIR, workdir: str | Path = "artifacts/ground",
               speed: float = 4.0, seed: int = 4242, compute_roi: bool = True,
               copilot: Copilot | None = None) -> FastAPI:
    if copilot is None:
        load_dotenv()  # optional .env (never committed); environment variables win
    gs = GroundStation(model_dir, workdir, speed, seed, compute_roi=compute_roi, copilot=copilot)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(gs.run())
        yield
        task.cancel()

    app = FastAPI(title="AeroMind Ground Station", lifespan=lifespan)
    app.state.gs = gs
    page = resources.files("aeromind").joinpath("web/index.html").read_text(encoding="utf-8")

    def ac(tail):
        if tail not in gs.fleet.aircraft:
            raise HTTPException(404, f"unknown aircraft {tail}")
        return gs.fleet.aircraft[tail]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page

    @app.get("/api/fleet")
    def fleet():
        return {**gs.fleet.snapshot(), "remote": [a.summary() for a in gs.remote.aircraft.values()]}

    @app.get("/api/aircraft/{tail}")
    def aircraft(tail: str):
        return ac(tail).detail()

    @app.post("/api/aircraft/{tail}/inject")
    def inject(tail: str, body: dict = Body(...)):
        ac(tail)
        fault = body.get("fault")
        try:
            if fault in FAULT_MODES:
                gs.fleet.inject_component(tail, fault, int(body.get("life", 320)))  # inside the training range (250-450)
            elif fault in SENSOR_PRESETS:
                gs.fleet.inject_sensor(tail, *SENSOR_PRESETS[fault])
            elif body.get("kind") and body.get("channel"):
                gs.fleet.inject_sensor(tail, body["kind"], body["channel"])
            else:
                raise ValueError(f"unknown fault {fault!r}")
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        gs.fleet.log("inject", f"{tail}: injected {fault or body.get('kind')} (simulation)")
        return {"ok": True}

    @app.post("/api/aircraft/{tail}/reset")
    def reset(tail: str):
        ac(tail)
        gs.fleet.reset(tail)
        gs.fleet.log("inject", f"{tail}: reset to healthy")
        return {"ok": True}

    @app.post("/api/aircraft/{tail}/clear_sensors")
    def clear_sensors(tail: str):
        ac(tail)
        gs.fleet.clear_sensor_faults(tail)
        return {"ok": True}

    @app.post("/api/aircraft/{tail}/phase")
    def phase(tail: str, body: dict = Body(...)):
        ac(tail)
        try:
            gs.fleet.set_phase(tail, body.get("phase") or None)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"ok": True}

    @app.post("/api/speed")
    def speed(body: dict = Body(...)):
        gs.fleet.set_speed(float(body.get("speed", 4)))
        if "paused" in body:
            gs.fleet.paused = bool(body["paused"])
        return {"speed": gs.fleet.speed, "paused": gs.fleet.paused}

    @app.get("/api/roi")
    def roi():
        return gs.roi

    @app.post("/api/aircraft/{tail}/whatif")
    def whatif(tail: str, body: dict = Body(...)):
        """The decision for the latest advisory if the next check were ``hours_to_next_check`` away."""
        from ..maintenance.decision import Schedule, decide

        a = ac(tail)
        if a.last_advisory is None:
            raise HTTPException(409, "no component advisory on this aircraft yet")
        s = Schedule(hours_to_next_check=float(body.get("hours_to_next_check", 10)),
                     leg_hours=float(body.get("leg_hours", 2)), check_station=a.station)
        return decide(a.last_advisory, s).to_dict()

    # ---- LLM maintenance copilot (ground side; explains deterministic output, never replaces it)
    @app.get("/api/llm/status")
    def llm_status():
        return gs.copilot.status()

    @app.post("/api/copilot")
    def copilot_ask(body: dict = Body(...)):
        req = CopilotRequest(task=str(body.get("task") or ""), tail=str(body.get("tail") or ""),
                             question=str(body.get("question") or ""),
                             hours_to_next_check=(float(body["hours_to_next_check"])
                                                  if body.get("hours_to_next_check") is not None else None))
        try:
            return gs.copilot.ask(req, FleetSource(gs)).to_dict()
        except UnknownAircraft as e:
            raise HTTPException(404, str(e.args[0])) from e
        except UnknownTask as e:
            raise HTTPException(400, str(e)) from e

    # ---- advisories from a remote edge agent (same EdgePipeline, other machine)
    @app.post("/api/ingest")
    def ingest(body: dict = Body(...), x_aeromind_token: str | None = Header(default=None)):
        required = os.environ.get("AEROMIND_INGEST_TOKEN")
        if required and not hmac.compare_digest(x_aeromind_token or "", required):
            raise HTTPException(401, "invalid or missing ingest token")
        try:
            ev = gs.remote.ingest(body)
        except IngestError as e:
            raise HTTPException(400, str(e)) from e
        gs.fleet.events.appendleft(ev)
        return {"ok": True, "acars": ev["acars"]}

    @app.get("/api/remote")
    def remote():
        return [a.summary() for a in gs.remote.aircraft.values()]

    @app.post("/api/ota")
    def ota(body: dict = Body(...)):
        return gs.ota(body.get("variant", "valid"))

    @app.post("/api/ota/rollback")
    def rollback():
        return gs.rollback()

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        gs.selected[socket] = next(iter(gs.fleet.aircraft))
        try:
            while True:
                msg = json.loads(await socket.receive_text())
                if msg.get("select") in gs.fleet.aircraft:
                    gs.selected[socket] = msg["select"]
        except WebSocketDisconnect:
            gs.selected.pop(socket, None)

    return app
