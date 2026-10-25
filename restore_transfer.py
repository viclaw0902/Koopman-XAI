from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def restore(verify_only=False):
    manifest = json.loads((ROOT / "transfer_manifest.json").read_text(encoding="utf-8"))
    for bundle in manifest["bundles"]:
        for part in bundle["parts"]:
            path = ROOT / part["path"]
            if not path.is_file() or path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
                raise ValueError(f"Missing or corrupt archive part: {path}")
        if verify_only:
            print(f"Verified archive parts: {bundle['name']}", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix="koopman_restore_") as temporary:
            assembled = Path(temporary) / "bundle.zip"
            with assembled.open("wb") as output:
                for part in bundle["parts"]:
                    with (ROOT / part["path"]).open("rb") as source:
                        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
            if digest(assembled) != bundle["zip_sha256"]:
                raise ValueError("Assembled archive checksum mismatch")
            with zipfile.ZipFile(assembled) as archive:
                allowed = {member["path"] for member in bundle["members"]}
                if set(archive.namelist()) != allowed:
                    raise ValueError("Archive inventory mismatch")
                for member in bundle["members"]:
                    relative = Path(member["path"])
                    target = (ROOT / relative).resolve()
                    if relative.is_absolute() or not target.is_relative_to(ROOT.resolve()):
                        raise ValueError("Unsafe archive path")
                    if target.exists():
                        if target.stat().st_size == member["bytes"] and digest(target) == member["sha256"]:
                            continue
                        raise FileExistsError(f"Refusing to overwrite modified/newer file: {target}. Restore into a fresh clone instead.")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member["path"]) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                    if target.stat().st_size != member["bytes"] or digest(target) != member["sha256"]:
                        raise ValueError(f"Extracted file checksum mismatch: {target}")
            print(f"Restored and verified: {bundle['name']}", flush=True)
    (ROOT / "Changed_structure").mkdir(exist_ok=True)
    print("Transfer verified. Training has NOT been started.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    restore(arguments.verify_only)
