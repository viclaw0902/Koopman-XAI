from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


ROOT = Path(__file__).resolve().parent
FEATURES = [f"XMEAS{number}" for number in range(1, 23)] + [f"XMV{number}" for number in range(1, 12)]
FAULTS = (*range(1, 13), 14, 15)
TRUTH = {
    1: ("XMV4", "XMEAS4"), 2: ("XMV4", "XMEAS4"),
    3: ("XMV1", "XMEAS2"), 4: ("XMV10", "XMEAS21"),
    5: ("XMEAS22",), 6: ("XMEAS1", "XMV3"),
    7: ("XMV4", "XMEAS4"), 8: ("XMEAS18", "XMV8"),
    9: ("XMV1", "XMEAS2"), 10: ("XMEAS18",),
    11: ("XMV10", "XMEAS9", "XMEAS21"), 12: ("XMEAS22",),
    14: ("XMV10", "XMEAS9", "XMEAS21"), 15: ("XMV11",),
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def prepared_directory(config):
    override = os.environ.get("KOOPMAN_DATA_DIR")
    candidates = [Path(override)] if override else [ROOT.parent / "rieth_prepared", Path(config["prepared_dir"])]
    for candidate in candidates:
        if (candidate / "train_values.npy").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Prepared data not found. Run ../restore_transfer.py or set KOOPMAN_DATA_DIR; keep config.json unchanged for checkpoint compatibility.")


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def setup(seed, config, device_choice="auto"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device_choice == "auto":
        device_choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_choice)


def report(output, stage, **values):
    record = {"time": datetime.now().astimezone().isoformat(), "stage": stage, **values}
    print(json.dumps(record, ensure_ascii=False), flush=True)
    save_json(Path(output) / "status.json", record)


def corrected_scaler(config):
    prepared = prepared_directory(config)
    values = np.load(prepared / "train_values.npy", mmap_mode="r")
    labels = np.load(prepared / "train_labels.npy")
    normal = np.asarray(values[labels == 0], dtype=np.float64).reshape(-1, 33)
    shift = normal.mean(0)
    scale = normal.std(0).clip(1e-8)
    original = np.load(prepared / "normal_scaler.npz")
    path = ROOT / "Results" / "normal_scaler_corrected.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, prepared_shift=shift, prepared_scale=scale,
             raw_mean=original["mean"].astype(np.float64) + original["std"].astype(np.float64) * shift,
             raw_std=original["std"].astype(np.float64) * scale, feature_names=np.asarray(FEATURES))
    return shift, scale


class WindowStore:
    def __init__(self, config, split, horizon=0):
        self.config = config
        self.split = split
        self.horizon = horizon
        prepared = prepared_directory(config)
        self.values = np.load(prepared / f"{split}_values.npy", mmap_mode="r")
        self.labels = np.load(prepared / f"{split}_labels.npy")
        self.runs = np.load(prepared / f"{split}_runs.npy")
        self.length = config["window"]
        self.windows_per_run = self.values.shape[1] - self.length + 1
        self.normal_prefix = config["normal_prefix"][split]
        self.tensor = torch.from_numpy(np.array(self.values, copy=True))
        shift, scale = corrected_scaler(config)
        self.shift = torch.from_numpy(shift.astype(np.float32))
        self.scale = torch.from_numpy(scale.astype(np.float32))
        self.tensor.sub_(self.shift).div_(self.scale)
        self.view = self.tensor.unfold(1, self.length, 1)
        count = len(self.labels) * self.windows_per_run
        self.indices = np.arange(count, dtype=np.int64)
        if horizon:
            starts = self.indices % self.windows_per_run
            faults = self.labels[self.indices // self.windows_per_run]
            same_normal = (faults == 0) | (starts + self.length + horizon <= self.normal_prefix)
            same_fault = (faults != 0) & (starts >= self.normal_prefix)
            self.indices = self.indices[(starts + horizon < self.windows_per_run) & (same_normal | same_fault)]

    def __len__(self):
        return len(self.indices)

    def batch(self, ids, device):
        trajectories = ids // self.windows_per_run
        starts = ids % self.windows_per_run
        windows = self.view[torch.from_numpy(trajectories), torch.from_numpy(starts)].contiguous()
        labels = self.labels[trajectories].copy()
        labels[starts + self.length <= self.normal_prefix] = 0
        return windows.to(device), torch.from_numpy(labels).long().to(device)

    def batches(self, batch_size, device, shuffle=False):
        order = np.random.permutation(self.indices) if shuffle else self.indices
        for offset in range(0, len(order), batch_size):
            ids = order[offset:offset + batch_size]
            windows, labels = self.batch(ids, device)
            yield ids, windows, labels


class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(33, 48, 5, padding=2), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 21))

    def representation(self, windows):
        recurrent, _ = self.gru(self.conv(windows).transpose(1, 2))
        return recurrent[:, -1]

    def forward(self, windows):
        return self.head(self.representation(windows))


class TemporalConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Sequential(nn.Conv1d(33, 48, 5, padding=2), nn.GELU(), nn.Conv1d(48, 64, 3, padding=1), nn.GELU())
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 20, 128), nn.GELU(), nn.Linear(128, 16))
        self.decoder_fc = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 64 * 20), nn.GELU())
        self.decoder_conv = nn.Sequential(nn.Conv1d(64, 48, 3, padding=1), nn.GELU(), nn.Conv1d(48, 33, 5, padding=2))

    def encode(self, windows):
        return self.encoder_fc(self.encoder_conv(windows))

    def decode(self, states):
        return self.decoder_conv(self.decoder_fc(states).reshape(-1, 64, 20))

    def forward(self, windows):
        return self.decode(self.encode(windows))


class Koopman(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.lift = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, dimension))
        self.inverse = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Linear(128, 16))
        self.operators = nn.Parameter(torch.eye(dimension).repeat(21, 1, 1))
        self.register_buffer("z_mean", torch.zeros(dimension))
        self.register_buffer("z_std", torch.ones(dimension))

    def encode(self, states):
        return (self.lift(states) - self.z_mean) / self.z_std

    def decode_h(self, states):
        return self.inverse(states * self.z_std + self.z_mean)

    def advance(self, states, classes):
        return torch.einsum("bij,bj->bi", self.operators[classes], states)

    def advance_class(self, states, target, steps=1):
        for _ in range(steps):
            states = functional.linear(states, self.operators[target])
        return states

    def decode(self, states, autoencoder):
        return autoencoder.decode(self.decode_h(states))


def classification_metrics(confusion):
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = confusion.diagonal()
    recall = diagonal / np.maximum(support, 1)
    precision = diagonal / np.maximum(predicted, 1)
    f1 = 2 * recall * precision / np.maximum(recall + precision, 1e-12)
    return {
        "accuracy": float(diagonal.sum() / max(confusion.sum(), 1)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()), "macro_precision": float(precision.mean()),
        "window_count": int(confusion.sum()),
    }


def freeze(model):
    return model.eval().requires_grad_(False)


def load_models(seed, device):
    directory = ROOT / "Results" / f"seed_{seed}"
    classifier = CNNGRU().to(device)
    autoencoder = TemporalConvAE().to(device)
    classifier.load_state_dict(torch.load(directory / "classifier.pt", map_location=device, weights_only=True))
    autoencoder.load_state_dict(torch.load(directory / "manifold.pt", map_location=device, weights_only=True))
    selection = json.loads((directory / "dimension_selection.json").read_text())
    dynamics = Koopman(selection["selected_dimension"]).to(device)
    dynamics.load_state_dict(torch.load(directory / f"z_{dynamics.dimension}" / "best.pt", map_location=device, weights_only=True))
    return freeze(classifier), freeze(autoencoder), freeze(dynamics)
