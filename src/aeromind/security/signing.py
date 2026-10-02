"""Signed over-the-air model packages (Ed25519) with verification and rollback.

A package is an ONNX export directory (``onnx_export.export_onnx``). ``manifest.json`` already
records the SHA-256 of every model file, so signing the manifest covers the whole package:

  sign:    package_version is written into manifest.json, then manifest.sig = Ed25519(manifest bytes)
  verify:  manifest.sig must verify with the fleet's trusted public key, then every file must
           match its SHA-256 in the manifest (checked again by ``OnnxBundle`` on load)

``ModelSlots`` keeps the active and the previous verified package; a package that fails
verification is never activated, and ``rollback()`` returns to the previous one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

SIG_FILE = "manifest.sig"


class PackageRejected(Exception):
    """The package failed signature or integrity verification."""


def generate_keypair() -> tuple[bytes, bytes]:
    """(private key, public key) as raw 32-byte Ed25519 keys."""
    priv = Ed25519PrivateKey.generate()
    raw = lambda k, enc: k.private_bytes(enc, serialization.PrivateFormat.Raw, serialization.NoEncryption())  # noqa: E731
    pub = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return raw(priv, serialization.Encoding.Raw), pub


def sign_package(pkg: str | Path, private_key: bytes, version: str) -> None:
    pkg = Path(pkg)
    manifest = json.loads((pkg / "manifest.json").read_text())
    manifest["package_version"] = version
    data = json.dumps(manifest, indent=2).encode()
    (pkg / "manifest.json").write_bytes(data)
    (pkg / SIG_FILE).write_bytes(Ed25519PrivateKey.from_private_bytes(private_key).sign(data))


def verify_package(pkg: str | Path, public_key: bytes) -> dict:
    """Return the manifest if the package is authentic and intact, else raise ``PackageRejected``."""
    pkg = Path(pkg)
    try:
        data = (pkg / "manifest.json").read_bytes()
        sig = (pkg / SIG_FILE).read_bytes()
    except FileNotFoundError as e:
        raise PackageRejected(f"missing {e.filename}") from e
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(sig, data)
    except InvalidSignature as e:
        raise PackageRejected("manifest signature does not verify with the fleet key") from e
    manifest = json.loads(data)
    for info in manifest["files"].values():
        f = pkg / info["file"]
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != info["sha256"]:
            raise PackageRejected(f"{info['file']} does not match its signed SHA-256")
    return manifest


@dataclass
class Slot:
    path: Path
    version: str
    bundle: object


class ModelSlots:
    """Active + previous verified model packages on the aircraft."""

    def __init__(self, public_key: bytes, store: str | Path):
        self.public_key, self.store = public_key, Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.active: Slot | None = None
        self.previous: Slot | None = None

    def install(self, pkg: str | Path) -> Slot:
        """Verify, copy into the store and activate. On any failure the active model is unchanged."""
        from ..edge.onnx_export import OnnxBundle

        manifest = verify_package(pkg, self.public_key)
        version = manifest.get("package_version", "unversioned")
        dest = self.store / version
        if Path(pkg).resolve() != dest.resolve():
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(pkg, dest)
        verify_package(dest, self.public_key)  # re-check the installed copy
        slot = Slot(dest, version, OnnxBundle(dest))
        self.previous, self.active = self.active, slot
        return slot

    def rollback(self) -> Slot:
        if self.previous is None:
            raise PackageRejected("no previous model to roll back to")
        self.active, self.previous = self.previous, self.active
        return self.active
