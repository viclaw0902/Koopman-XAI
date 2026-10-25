from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Original_structure"))


def main(allow_cpu=False):
    from core import CNNGRU, Koopman, WindowStore, freeze, load_config, prepared_directory
    from deeplift_gru import explain
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("CUDA unavailable. Install a driver/PyTorch build compatible with your GPU; do not silently start slow CPU training.")
    torch.set_num_threads(2)
    classifier = freeze(CNNGRU().to(device))
    seed = ROOT / "Original_structure" / "Results" / "seed_42"
    classifier.load_state_dict(torch.load(seed / "classifier.pt", map_location=device, weights_only=True))
    store = WindowStore(config, "validation")
    windows, _ = store.batch(np.array([50 * store.windows_per_run, 50 * store.windows_per_run + 100], dtype=np.int64), device)
    target = int(classifier(windows[1:]).argmax(1))
    attribution, diagnostics = explain(classifier, windows[1], windows[0], target)
    if not torch.isfinite(attribution).all() or diagnostics["completeness_absolute_error"] > 1e-3:
        raise ValueError(f"DeepLIFT validation failed: {diagnostics}")
    checkpoint = torch.load(seed / "z_16" / "last.pt", map_location=device, weights_only=False)
    model = Koopman(16).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW([
        {"params": list(model.lift.parameters()) + list(model.inverse.parameters()), "lr": config["dynamics_learning_rate"]},
        {"params": [model.operators], "lr": config["operator_learning_rate"]},
    ])
    optimizer.load_state_dict(checkpoint["optimizer"])
    encoded = model.encode(torch.randn(8, 16, device=device))
    loss = model.decode_h(model.advance(encoded, torch.arange(8, device=device))).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.linalg.eigvals(model.operators.detach())
    print(json.dumps({"environment_check": "passed", "python": platform.python_version(), "torch": torch.__version__,
                      "numpy": np.__version__, "device": str(device),
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                      "cuda_runtime": torch.version.cuda,
                      "gpu_compute_capability": torch.cuda.get_device_capability(0) if device.type == "cuda" else None,
                      "pytorch_cuda_architectures": torch.cuda.get_arch_list() if device.type == "cuda" else [],
                      "data": str(prepared_directory(config)), "saved_epoch": checkpoint["epoch"],
                      "next_epoch": checkpoint["epoch"] + 1, "deeplift": diagnostics,
                      "optimizer_portability": "passed; in-memory test only, checkpoints unchanged"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    arguments = parser.parse_args()
    main(arguments.allow_cpu)
