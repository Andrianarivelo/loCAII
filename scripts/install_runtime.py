#!/usr/bin/env python3
"""Install the pinned official GPU runtime, verifying GitHub's published SHA-256.

This package uses Vulkan on Linux, including NVIDIA GPUs. Set LOCAII_LLAMA_SERVER
at launch to use a separately built compatible CUDA llama-server instead.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

RELEASE = "b10893"
ASSET = f"llama-{RELEASE}-bin-ubuntu-vulkan-x64.tar.gz"
SHA256 = "af00feecc9f417d89dc0649f9820d77d4a08b3257f206bca64766058d6046d56"
URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{ASSET}"
ROOT = Path(__file__).resolve().parents[1]


def install(destination: Path) -> Path:
    """Download into staging, verify all bytes, then extract safe archive entries."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".install-") as temporary:
        stage = Path(temporary)
        archive = stage / ASSET
        print(f"Downloading official llama.cpp {RELEASE} Vulkan runtime...", flush=True)
        with (
            urllib.request.urlopen(URL, timeout=60) as source,
            archive.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != SHA256:
            raise RuntimeError(
                f"Runtime checksum mismatch: expected {SHA256}, got {digest}"
            )
        unpack = stage / "unpacked"
        unpack.mkdir()
        with tarfile.open(archive) as package:
            package.extractall(unpack, filter="data")
        server = next(unpack.rglob("llama-server"), None)
        if server is None:
            raise RuntimeError("Official archive does not contain llama-server")
        # Keep the release layout and adjacent shared libraries together.
        runtime = destination / RELEASE
        if runtime.exists():
            shutil.rmtree(runtime)
        shutil.move(str(unpack), runtime)
        server = runtime / server.relative_to(unpack)
        server.chmod(server.stat().st_mode | 0o111)
        manifest = {
            "release": RELEASE,
            "backend": "vulkan",
            "url": URL,
            "sha256": digest,
            "server": str(server.resolve()),
        }
        temp = destination / "runtime.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temp, destination / "runtime.json")
        print(server, flush=True)
        return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "runtime")
    install(parser.parse_args().destination)
