from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from core import ROOT, FAULTS, CNNGRU, WindowStore, freeze, load_config, save_csv, save_json, setup
from explain import select_endpoints


def run(seed, device_choice):
    config = load_config()
    device = setup(seed, config, device_choice)
    output = ROOT / "Results" / f"seed_{seed}"
    checkpoint = output / "classifier.pt"
    model = CNNGRU().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    freeze(model)
    classifier_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    store = WindowStore(config, "test")
    indices = np.flatnonzero(np.isin(store.labels, FAULTS))
    marker = output / "endpoint_selection.json"
    records, failures = [], []
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["classifier_sha256"] != classifier_hash:
            raise ValueError("Classifier changed since endpoint selection")
        records = previous["endpoints"]
        failures = previous["failures"]
    completed = {record["trajectory"] for record in records + failures}
    previous_report = time.monotonic()
    for trajectory in indices:
        trajectory = int(trajectory)
        if trajectory in completed:
            continue
        record = select_endpoints(model, store, trajectory, config, device)
        if record is None:
            failures.append({"trajectory": trajectory, "fault": int(store.labels[trajectory]),
                             "run": int(store.runs[trajectory]), "reason": "no fully post-injection Fault prediction"})
        else:
            records.append(record)
        save_json(marker, {"seed": seed, "classifier_sha256": classifier_hash, "requested_trajectories": len(indices),
                           "selection_complete": len(records) + len(failures) == len(indices), "endpoints": records, "failures": failures})
        if time.monotonic() - previous_report > 30 or len(records) + len(failures) == len(indices):
            print(json.dumps({"seed": seed, "endpoint_pairs": len(records), "failures": len(failures),
                              "processed": len(records) + len(failures), "total": len(indices)}), flush=True)
            previous_report = time.monotonic()
    records.sort(key=lambda record: record["trajectory"])
    windows = np.stack([np.stack((store.view[record["trajectory"], record["normal_start"]].numpy(),
                                 store.view[record["trajectory"], record["fault_start"]].numpy())) for record in records])
    np.savez_compressed(output / "endpoint_windows.npz", windows=windows,
                        trajectories=np.array([record["trajectory"] for record in records]),
                        targets=np.array([record["target"] for record in records]),
                        faults=np.array([record["fault"] for record in records]), classifier_sha256=classifier_hash)
    save_csv(output / "endpoint_selection.csv", records)
    save_json(output / "endpoint_selection_complete.json", {"pairs": len(records), "failures": len(failures),
                                                              "classifier_sha256": classifier_hash})
    print(json.dumps({"saved_pairs": len(records), "failed_trajectories": len(failures), "windows_shape": list(windows.shape)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    run(arguments.seed, arguments.device)
