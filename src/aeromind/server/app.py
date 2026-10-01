"""AeroMind ground station: FastAPI + WebSocket over a simulated fleet.

Run with ``python -m aeromind serve`` and open http://127.0.0.1:8000. Works offline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from importlib import resources
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..config import FAULT_MODES
from ..signing import ModelSlots, PackageRejected, generate_keypair, sign_package
from .fleet import SENSOR_PRESETS, Fleet

DEFAULT_MODEL_DIR = "artifacts/onnx-phases"


def ensure_model(model_dir: str | Path) -> Path:
    """The phase-aware ONNX model the fleet runs; trained and exported on first start (~20 s)."""
    d = Path(model_dir)
    if not (d / "manifest.json").exists():
        from ..onnx_export import export_onnx
        from ..train import TrainConfig, train

        print(f"No model at {d}: training a flight-phase model (about 20 s)...", flush=True)
        export_onnx(train(TrainConfig(phases=True)), d)
    return d


def _keys(workdir: Path) -> tuple[bytes, bytes]:
    priv_f, pub_f = workdir / "keys" / "ground_ed25519.key", workdir / "keys" / "ground_ed25519.pub"
    if not priv_f.exists():
        priv_f.parent.mkdir(parents=True, exist_ok=True)
        priv, pub = generate_keypair()
        priv_f.write_bytes(priv)
        pub_f.write_bytes(pub)
    return priv_f.read_bytes(), pub_f.read_bytes()


class GroundStation:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, workdir: str | Path = "artifacts/ground",
                 speed: float = 4.0, seed: int = 4242):
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
        snap = self.fleet.snapshot()
        details = {t: self.fleet.aircraft[t].detail() for t in set(self.selected.values()) if t in self.fleet.aircraft}
        for ws, tail in list(self.selected.items()):
            try:
                await ws.send_text(json.dumps({**snap, "detail": details.get(tail)}))
            except Exception:
                self.selected.pop(ws, None)


def create_app(model_dir: str | Path = DEFAULT_MODEL_DIR, workdir: str | Path = "artifacts/ground",
               speed: float = 4.0, seed: int = 4242) -> FastAPI:
    gs = GroundStation(model_dir, workdir, speed, seed)

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
        return gs.fleet.snapshot()

    @app.get("/api/aircraft/{tail}")
    def aircraft(tail: str):
        return ac(tail).detail()

    @app.post("/api/aircraft/{tail}/inject")
    def inject(tail: str, body: dict = Body(...)):
        ac(tail)
        fault = body.get("fault")
        try:
            if fault in FAULT_MODES:
                gs.fleet.inject_component(tail, fault, int(body.get("life", 240)))
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
