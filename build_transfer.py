from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def archive(name, members):
    destination = ROOT / "artifacts"
    destination.mkdir(exist_ok=True)
    temporary = destination / f"{name}.zip"
    inventory = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as bundle:
        for source, target in members:
            inventory.append({"path": target, "bytes": source.stat().st_size, "sha256": digest(source)})
            bundle.write(source, arcname=target)
    parts = []
    with temporary.open("rb") as handle:
        number = 1
        while True:
            block = handle.read(45 * 1024 * 1024)
            if not block:
                break
            path = destination / f"{name}.zip.part{number:03d}"
            path.write_bytes(block)
            parts.append({"path": path.relative_to(ROOT).as_posix(), "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
            number += 1
    result = {"name": name, "zip_sha256": digest(temporary), "zip_bytes": temporary.stat().st_size,
              "parts": parts, "members": inventory}
    temporary.unlink()
    return result


def main(data):
    results = ROOT / "Original_structure" / "Results"
    manifest = {"created": datetime.now().astimezone().isoformat(), "format": "split ZIP, maximum45MiB per Git object",
                "bundles": []}
    data_members = [(path, f"rieth_prepared/{path.name}") for path in sorted(data.iterdir()) if path.is_file()]
    manifest["bundles"].append(archive("prepared_data", data_members))
    result_members = [(path, path.relative_to(ROOT).as_posix()) for path in sorted(results.rglob("*")) if path.is_file() and path.name != "pipeline.lock"]
    manifest["bundles"].append(archive("resume_seed42", result_members))
    (ROOT / "transfer_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot = {"python": sys.version, "platform": platform.platform(),
                "packages": sorted({f"{item.metadata['Name']}=={item.version}" for item in importlib.metadata.distributions() if item.metadata.get("Name")})}
    environment = ROOT / "environment"
    environment.mkdir(exist_ok=True)
    (environment / "source_environment_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"bundles": [{"name": bundle["name"], "parts": len(bundle["parts"]), "compressed_bytes": bundle["zip_bytes"]} for bundle in manifest["bundles"]]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.data)
