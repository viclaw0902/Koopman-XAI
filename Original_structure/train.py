from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional

from core import (
    ROOT, CNNGRU, TemporalConvAE, Koopman, WindowStore, classification_metrics,
    corrected_scaler, fingerprint, freeze, load_config, prepared_directory, report, save_csv, save_json, setup,
)


def audit_data(config):
    prepared = prepared_directory(config)
    records = {}
    split_runs = []
    for split in ("train", "validation", "test"):
        values = np.load(prepared / f"{split}_values.npy", mmap_mode="r")
        labels = np.load(prepared / f"{split}_labels.npy")
        runs = np.load(prepared / f"{split}_runs.npy")
        if values.shape[2] != 33 or sorted(np.unique(labels).tolist()) != list(range(21)):
            raise ValueError(f"Invalid features/classes: {split}")
        if len(labels) != len(runs) or len(labels) != len(values):
            raise ValueError(f"Invalid lengths: {split}")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite input: {split}")
        prefix = config["normal_prefix"][split]
        reference = {int(run): index for index, run in enumerate(runs) if labels[index] == 0}
        checked = 0
        for index in np.flatnonzero(labels != 0):
            normal = reference[int(runs[index])]
            if not np.array_equal(values[index, :prefix], values[normal, :prefix]):
                raise ValueError(f"Normal prefix differs from matched normal run: {split}/{index}")
            checked += 1
        counts = np.bincount(labels, minlength=21)
        expected = 400 if split == "train" else 50
        if not np.all(counts == expected):
            raise ValueError(f"Expected {expected} trajectories per class: {split}")
        windows = values.shape[1] - config["window"] + 1
        records[split] = {
            "shape": list(values.shape), "trajectories_per_class": counts.tolist(),
            "all_classification_windows": len(values) * windows,
            "normal_prefix": prefix, "paired_normal_prefixes_verified": checked,
            "normal_window_labels": "window ending at or before normal_prefix -> Normal; otherwise trajectory fault",
            "run_ids": sorted(set(runs.tolist())),
        }
        split_runs.append(set(runs.tolist()))
    if any(split_runs[first] & split_runs[second] for first, second in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Run ID overlap between splits")
    shift, scale = corrected_scaler(config)
    records["scaler_correction"] = {
        "reason": "Historical float32 mean accumulation shifted low-variance channels; refit equivalent affine scaler in float64 using only normal training data",
        "prepared_shift": shift.tolist(), "prepared_scale": scale.tolist(),
        "source_files_modified": False,
    }
    records["config_fingerprint"] = fingerprint(config)
    records["read_only_source"] = str(prepared)
    save_json(ROOT / "Results" / "data_audit.json", records)
    return records


@torch.no_grad()
def evaluate_classifier(model, store, config, device):
    model.eval()
    confusion = np.zeros((21, 21), dtype=np.int64)
    total_loss = 0.0
    for _, windows, labels in store.batches(config["batch_size"], device):
        logits = model(windows)
        total_loss += float(functional.cross_entropy(logits, labels, reduction="sum"))
        encoded = (21 * labels + logits.argmax(1)).cpu().numpy()
        confusion += np.bincount(encoded, minlength=441).reshape(21, 21)
    return total_loss / len(store), confusion


@torch.no_grad()
def reconstruction_error(model, store, config, device):
    model.eval()
    squared = 0.0
    elements = 0
    for _, windows, _ in store.batches(config["batch_size"], device):
        squared += float((model(windows) - windows).square().sum())
        elements += windows.numel()
    return squared / elements


def train_window_model(kind, model, training, validation, config, device, output):
    checkpoint = output / f"{kind}.pt"
    completion = output / f"{kind}_complete.json"
    if completion.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return freeze(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    history = []
    best = math.inf
    stale = 0
    start_epoch = 1
    resume_path = output / f"{kind}_last.pt"
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history, best, stale = resume["history"], resume["best"], resume["stale"]
        start_epoch = resume["epoch"] + 1
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"].cpu())
        if device.type == "cuda" and resume["cuda_rng"] is not None:
            torch.cuda.set_rng_state(resume["cuda_rng"].cpu())
    model.requires_grad_(True)
    for epoch in range(start_epoch, config["maximum_epochs"] + 1):
        model.train()
        total = 0.0
        seen = 0
        began = time.monotonic()
        previous = began
        optimizer.zero_grad(set_to_none=True)
        batch_count = math.ceil(len(training) / config["batch_size"])
        accumulation = config["gradient_accumulation"]
        for batch_index, (_, windows, labels) in enumerate(training.batches(config["batch_size"], device, shuffle=True)):
            prediction = model(windows)
            loss = functional.cross_entropy(prediction, labels) if kind == "classifier" else functional.mse_loss(prediction, windows)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {kind} loss")
            group_start = batch_index // accumulation * accumulation
            group_length = min(accumulation, batch_count - group_start)
            (loss / group_length).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(windows)
            seen += len(windows)
            now = time.monotonic()
            if now - previous >= 45 or batch_index == 0:
                report(output, kind, epoch=epoch, windows=seen, total_windows=len(training),
                       training_loss=total / seen, windows_per_second=seen / (now - began),
                       elapsed_seconds=now - began)
                previous = now
        if kind == "classifier":
            validation_loss, confusion = evaluate_classifier(model, validation, config, device)
            validation_metrics = classification_metrics(confusion)
        else:
            validation_loss = reconstruction_error(model, validation, config, device)
            validation_metrics = {}
        scheduler.step(validation_loss)
        improved = validation_loss < best - 1e-8
        if improved:
            best, stale = validation_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": total / seen, "validation_loss": validation_loss,
               "seconds": time.monotonic() - began, "learning_rate": optimizer.param_groups[0]["lr"], **validation_metrics}
        history.append(row)
        save_csv(output / f"{kind}_history.csv", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "history": history, "best": best, "stale": stale,
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}, resume_path)
        report(output, f"{kind}_epoch_complete", **row)
        if epoch >= config["minimum_epochs"] and stale >= config["patience"]:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    save_json(completion, {"best_validation_loss": best, "epochs": len(history), "config_fingerprint": fingerprint(config)})
    return freeze(model)


@torch.no_grad()
def cache_h(autoencoder, store, config, device, output):
    path = output / f"{store.split}_h.npy"
    marker = output / f"{store.split}_h_complete.json"
    expected_shape = (len(store.labels) * store.windows_per_run, 16)
    if marker.exists():
        states = np.load(path, mmap_mode="r")
        if states.shape != expected_shape:
            raise ValueError("Invalid latent cache")
        return states
    states = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=expected_shape)
    previous = time.monotonic()
    for ids, windows, _ in store.batches(config["batch_size"], device):
        states[ids] = autoencoder.encode(windows).cpu().numpy()
        if time.monotonic() - previous > 45:
            report(output, "encode_shared_manifold", split=store.split, windows=int(ids[-1]) + 1, total_windows=len(store))
            previous = time.monotonic()
    states.flush()
    save_json(marker, {"shape": list(expected_shape)})
    return np.load(path, mmap_mode="r")


def h_batch(cache, ids, device):
    return torch.from_numpy(np.asarray(cache[ids]).copy()).to(device)


def pretrain_lift(model, cache, store, config, device, output):
    path = output / "lift_pretrained.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return
    optimizer = torch.optim.AdamW(list(model.lift.parameters()) + list(model.inverse.parameters()), lr=config["learning_rate"])
    for epoch in range(config["lift_pretrain_epochs"]):
        losses = []
        previous = time.monotonic()
        for offset in range(0, len(store), config["batch_size"]):
            if offset == 0:
                order = np.random.permutation(store.indices)
            current = h_batch(cache, order[offset:offset + config["batch_size"]], device)
            encoded = model.lift(current)
            reconstructed = model.inverse(encoded)
            inverse = functional.mse_loss(reconstructed, current)
            variance = functional.relu(0.5 - encoded.std(0, unbiased=False)).square().mean()
            loss = inverse + config["variance_weight"] * variance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if time.monotonic() - previous > 45:
                report(output, "lift_pretrain", dimension=model.dimension, epoch=epoch + 1, windows=offset, loss=float(np.mean(losses)))
                previous = time.monotonic()
        report(output, "lift_pretrain_epoch_complete", dimension=model.dimension, epoch=epoch + 1, loss=float(np.mean(losses)))
    with torch.no_grad():
        total = torch.zeros(model.dimension, dtype=torch.float64, device=device)
        squared = total.clone()
        count = 0
        for offset in range(0, len(cache), config["batch_size"]):
            encoded = model.lift(h_batch(cache, np.arange(offset, min(offset + config["batch_size"], len(cache))), device)).double()
            total += encoded.sum(0)
            squared += encoded.square().sum(0)
            count += len(encoded)
        mean = total / count
        deviation = (squared / count - mean.square()).clamp_min(1e-8).sqrt()
        model.z_mean.copy_(mean.float())
        model.z_std.copy_(deviation.float())
    torch.save(model.state_dict(), path)


def dynamics_arrays(store, cache, ids, device, horizon):
    trajectories = ids // store.windows_per_run
    starts = ids % store.windows_per_run
    labels = store.labels[trajectories].copy()
    labels[starts + store.length <= store.normal_prefix] = 0
    offsets = np.arange(1, horizon + 1)
    future_ids = ids[:, None] + offsets[None]
    current = h_batch(cache, ids, device)
    future = h_batch(cache, future_ids.reshape(-1), device).reshape(len(ids), horizon, 16)
    return current, future, torch.from_numpy(labels).long().to(device)


@torch.no_grad()
def initialize_ridge(model, train_store, train_cache, validation_store, validation_cache, config, device, output):
    count = 21
    dimension = model.dimension
    gram = torch.zeros(count, dimension, dimension, dtype=torch.float64, device=device)
    cross = gram.clone()
    samples = torch.zeros(count, dtype=torch.float64, device=device)
    for offset in range(0, len(train_store), config["batch_size"]):
        ids = train_store.indices[offset:offset + config["batch_size"]]
        current, future, labels = dynamics_arrays(train_store, train_cache, ids, device, 1)
        encoded = model.encode(current).double()
        target = model.encode(future[:, 0]).double()
        for fault in labels.unique().tolist():
            mask = labels == fault
            gram[fault] += encoded[mask].T @ encoded[mask]
            cross[fault] += target[mask].T @ encoded[mask]
            samples[fault] += int(mask.sum())
    if (samples == 0).any():
        raise ValueError("No dynamics pairs for a class")
    choices = []
    losses = []
    for ridge in config["ridge_grid"]:
        covariance = gram / samples[:, None, None]
        regularized = covariance + ridge * torch.eye(dimension, device=device, dtype=torch.float64)
        operator = torch.linalg.solve(regularized, (cross / samples[:, None, None]).transpose(1, 2)).transpose(1, 2).float()
        choices.append(operator)
        totals = torch.zeros(count, dtype=torch.float64, device=device)
        denominators = totals.clone()
        for offset in range(0, len(validation_store), config["batch_size"]):
            ids = validation_store.indices[offset:offset + config["batch_size"]]
            current, future, labels = dynamics_arrays(validation_store, validation_cache, ids, device, config["horizon"])
            prediction = model.encode(current)
            target = model.encode(future.reshape(-1, 16)).reshape(len(ids), config["horizon"], dimension)
            for step in range(config["horizon"]):
                prediction = torch.einsum("bij,bj->bi", operator[labels], prediction)
                error = (prediction - target[:, step]).square().mean(1).double()
                totals.scatter_add_(0, labels, error)
                denominators.scatter_add_(0, labels, torch.ones_like(error))
        losses.append(totals / denominators)
    losses_tensor = torch.stack(losses)
    selected = losses_tensor.argmin(0)
    stack = torch.stack(choices)
    model.operators.copy_(stack[selected, torch.arange(count, device=device)])
    records = []
    for fault in range(count):
        records.append({"class": fault, "ridge": config["ridge_grid"][int(selected[fault])], "training_pairs": int(samples[fault]),
                        "validation_recursive_10step_mse": float(losses_tensor[selected[fault], fault])})
    save_csv(output / "ridge_selection.csv", records)
    torch.save(model.state_dict(), output / "ridge_initialized.pt")


def dynamics_loss(model, autoencoder, current, future, labels, config, raw=None):
    encoded = model.encode(current)
    target = model.encode(future.reshape(-1, 16)).reshape(len(current), config["horizon"], model.dimension).detach()
    prediction = encoded
    errors = []
    decoded_errors = []
    for step in range(config["horizon"]):
        prediction = model.advance(prediction, labels)
        errors.append(functional.mse_loss(prediction, target[:, step]))
        decoded_errors.append(functional.mse_loss(model.decode_h(prediction), future[:, step]))
    inverse = functional.mse_loss(model.decode_h(encoded), current)
    cycle = functional.mse_loss(model.encode(model.decode_h(encoded)), encoded.detach())
    variance = functional.relu(0.5 - encoded.std(0, unbiased=False)).square().mean()
    dynamics = config["one_step_weight"] * errors[0] + config["multi_step_weight"] * torch.stack(errors).mean()
    decoded = torch.stack(decoded_errors).mean()
    loss = dynamics + decoded + config["inverse_weight"] * inverse + config["cycle_weight"] * cycle + config["variance_weight"] * variance
    if raw is not None:
        loss = loss + config["raw_reconstruction_weight"] * functional.mse_loss(model.decode(encoded, autoencoder), raw)
    details = {"z_one_step": errors[0], "z_ten_step": errors[-1], "h_multistep": decoded, "inverse": inverse, "cycle": cycle, "variance": variance}
    return loss, details


@torch.no_grad()
def evaluate_dynamics(model, autoencoder, store, cache, config, device):
    model.eval()
    count = torch.zeros(21, dtype=torch.float64, device=device)
    sums = torch.zeros(21, 7, dtype=torch.float64, device=device)
    latent_total = torch.zeros(model.dimension, dtype=torch.float64, device=device)
    latent_outer = torch.zeros(model.dimension, model.dimension, dtype=torch.float64, device=device)
    for offset in range(0, len(store), config["batch_size"]):
        ids = store.indices[offset:offset + config["batch_size"]]
        current, future, labels = dynamics_arrays(store, cache, ids, device, config["horizon"])
        encoded = model.encode(current)
        latent_total += encoded.double().sum(0)
        latent_outer += encoded.double().T @ encoded.double()
        target = model.encode(future.reshape(-1, 16)).reshape(len(ids), config["horizon"], model.dimension)
        first = model.advance(encoded, labels)
        prediction = first
        for _ in range(1, config["horizon"]):
            prediction = model.advance(prediction, labels)
        raw, _ = store.batch(ids, device)
        future_raw, _ = store.batch(ids + config["horizon"], device)
        metrics = torch.stack((
            (first - target[:, 0]).square().mean(1),
            (prediction - target[:, -1]).square().mean(1),
            (model.decode_h(prediction) - future[:, -1]).square().mean(1),
            (model.decode(prediction, autoencoder) - future_raw).square().mean((1, 2)),
            (raw - future_raw).square().mean((1, 2)),
            (model.decode(encoded, autoencoder) - raw).square().mean((1, 2)),
            (model.decode_h(encoded) - current).square().mean(1),
        ), dim=1).double()
        count.scatter_add_(0, labels, torch.ones(len(ids), device=device, dtype=torch.float64))
        sums.index_add_(0, labels, metrics)
    means = sums / count[:, None]
    covariance = latent_outer / count.sum() - torch.outer(latent_total, latent_total) / count.sum().square()
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
    names = ("z_one_step", "z_ten_step", "h_ten_step", "raw_ten_step", "raw_persistence_ten_step", "raw_reconstruction", "h_inverse")
    by_class = [{"class": fault, "windows": int(count[fault]), **{name: float(means[fault, index]) for index, name in enumerate(names)}} for fault in range(21)]
    summary = {name: float(means[:, index].mean()) for index, name in enumerate(names)}
    summary.update(effective_rank=effective_rank, minimum_latent_variance=float(covariance.diagonal().min()),
                   selection_score=float(means[:, 3].mean() + means[:, 5].mean()),
                   maximum_spectral_radius=float(torch.linalg.eigvals(model.operators).abs().amax()))
    return summary, by_class


def train_dynamics(model, autoencoder, train_store, train_cache, validation_store, validation_cache, config, device, output):
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "complete.json"
    if complete.exists():
        model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
        return json.loads(complete.read_text())
    pretrain_lift(model, train_cache, train_store, config, device, output)
    initialized = output / "ridge_initialized.pt"
    if initialized.exists():
        model.load_state_dict(torch.load(initialized, map_location=device, weights_only=True))
    else:
        report(output, "ridge_initialization", dimension=model.dimension)
        initialize_ridge(model, train_store, train_cache, validation_store, validation_cache, config, device, output)
    optimizer = torch.optim.AdamW([
        {"params": list(model.lift.parameters()) + list(model.inverse.parameters()), "lr": config["dynamics_learning_rate"]},
        {"params": [model.operators], "lr": config["operator_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    history = []
    best = math.inf
    stale = 0
    start_epoch = 1
    last = output / "last.pt"
    if last.exists():
        resume = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        history, best, stale = resume["history"], resume["best"], resume["stale"]
        start_epoch = resume["epoch"] + 1
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"].cpu())
    for epoch in range(start_epoch, config["maximum_epochs"] + 1):
        model.train()
        order = np.random.permutation(train_store.indices)
        total = 0.0
        seen = 0
        previous = time.monotonic()
        for offset in range(0, len(order), config["batch_size"]):
            ids = order[offset:offset + config["batch_size"]]
            current, future, labels = dynamics_arrays(train_store, train_cache, ids, device, config["horizon"])
            raw, _ = train_store.batch(ids, device)
            loss, _ = dynamics_loss(model, autoencoder, current, future, labels, config, raw)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite dynamics loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            seen += len(ids)
            if time.monotonic() - previous >= 45 or offset == 0:
                report(output, "dynamics", dimension=model.dimension, epoch=epoch, windows=seen, total_windows=len(order), loss=total / seen)
                previous = time.monotonic()
        radius = torch.linalg.eigvals(model.operators).abs().amax(1)
        spectral_loss = config["spectral_weight"] * functional.relu(radius - config["spectral_cap"]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        spectral_loss.backward()
        optimizer.step()
        with torch.no_grad():
            radius = torch.linalg.eigvals(model.operators).abs().amax(1)
            scale = (config["spectral_cap"] / radius.clamp_min(1e-8)).clamp_max(1)
            model.operators.mul_(scale[:, None, None])
        summary, by_class = evaluate_dynamics(model, autoencoder, validation_store, validation_cache, config, device)
        score = summary["selection_score"]
        if score < best - 1e-8:
            best, stale = score, 0
            torch.save(model.state_dict(), output / "best.pt")
            save_json(output / "best_validation.json", summary)
            save_csv(output / "best_validation_by_class.csv", by_class)
        else:
            stale += 1
        row = {"epoch": epoch, "training_loss": total / seen, **summary}
        history.append(row)
        save_csv(output / "history.csv", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best,
                    "stale": stale, "epoch": epoch, "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, last)
        report(output, "dynamics_epoch_complete", dimension=model.dimension, **row)
        if epoch >= config["minimum_epochs"] and stale >= config["patience"]:
            break
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    summary = json.loads((output / "best_validation.json").read_text())
    record = {"dimension": model.dimension, "epochs": len(history), **summary}
    save_json(complete, record)
    for fault in range(21):
        np.savetxt(output / f"K_{fault:02d}.csv", model.operators[fault].detach().cpu().numpy(), delimiter=",")
    return record


def run(seed, config, device_choice="auto"):
    device = setup(seed, config, device_choice)
    output = ROOT / "Results" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "config.json"
    if existing.exists() and fingerprint(json.loads(existing.read_text())) != fingerprint(config):
        raise ValueError("Config changed; do not reuse old checkpoints")
    save_json(existing, config)
    report(output, "starting", seed=seed, device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    training = WindowStore(config, "train")
    validation = WindowStore(config, "validation")
    classifier = train_window_model("classifier", CNNGRU().to(device), training, validation, config, device, output)
    classification_path = output / "classification_metrics.json"
    if not classification_path.exists():
        test_store = WindowStore(config, "test")
        loss, confusion = evaluate_classifier(classifier, test_store, config, device)
        save_json(classification_path, {**classification_metrics(confusion), "loss": loss, "shared_by_all_methods": True})
        save_csv(output / "confusion_matrix.csv", [{"true_class": fault, **{f"pred_{column}": int(confusion[fault, column]) for column in range(21)}} for fault in range(21)])
        del test_store
    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    autoencoder = train_window_model("manifold", TemporalConvAE().to(device), training, validation, config, device, output)
    train_cache = cache_h(autoencoder, training, config, device, output)
    validation_cache = cache_h(autoencoder, validation, config, device, output)
    training.horizon = config["horizon"]
    validation.horizon = config["horizon"]
    for store in (training, validation):
        starts = store.indices % store.windows_per_run
        faults = store.labels[store.indices // store.windows_per_run]
        normal = (faults == 0) | (starts + store.length + config["horizon"] <= store.normal_prefix)
        faulty = (faults != 0) & (starts >= store.normal_prefix)
        store.indices = store.indices[(starts + config["horizon"] < store.windows_per_run) & (normal | faulty)]
    dimensions = []
    for dimension in config["z_dimensions"]:
        setup(seed * 1000 + dimension, config, device_choice)
        model = Koopman(dimension).to(device)
        dimensions.append(train_dynamics(model, autoencoder, training, train_cache, validation, validation_cache, config, device, output / f"z_{dimension}"))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores = [record["selection_score"] for record in dimensions]
    minimum = min(scores)
    eligible = [record for record in dimensions if record["selection_score"] <= minimum * (1 + config["dimension_plateau_tolerance"]) and record["effective_rank"] >= 8]
    if not eligible:
        raise RuntimeError("All latent candidates collapsed; review diagnostics before explanation")
    selected = min(eligible, key=lambda record: record["dimension"])
    save_json(output / "dimension_selection.json", {"selected_dimension": selected["dimension"], "rule": "smallest noncollapsed dimension within 2% of minimum validation decoded prediction + reconstruction MSE", "candidates": dimensions})
    report(output, "training_complete", seed=seed, selected_dimension=selected["dimension"])


def self_check(config, device_choice="auto"):
    device = setup(42, config, device_choice)
    store = WindowStore(config, "validation")
    ids = np.array([0, 1, 19, store.windows_per_run, 50 * store.windows_per_run, 50 * store.windows_per_run + 1])
    windows, labels = store.batch(ids, device)
    assert windows.shape == (6, 33, 20)
    assert labels.tolist() == [0, 0, 0, 0, 0, 1]
    assert torch.equal(windows[0, :, 1:], windows[1, :, :-1])
    classifier = CNNGRU().to(device)
    autoencoder = TemporalConvAE().to(device)
    logits = classifier(windows)
    assert logits.shape == (6, 21)
    functional.cross_entropy(logits, labels).backward()
    freeze(autoencoder)
    for dimension in config["z_dimensions"]:
        model = Koopman(dimension).to(device)
        current = autoencoder.encode(windows).detach()
        future = current[:, None].expand(-1, config["horizon"], -1).contiguous()
        loss, _ = dynamics_loss(model, autoencoder, current, future, labels, config, windows)
        loss.backward()
        assert model.inverse[0].weight.grad is not None
        assert model.inverse[0].weight.shape[1] == dimension
        assert model.lift[0].weight.grad is not None
        assert model.operators.grad is not None
        assert all(parameter.grad is None for parameter in autoencoder.parameters())
    with torch.no_grad():
        normal_mean = store.tensor[:50].double().reshape(-1, 33).mean(0)
        assert torch.isfinite(normal_mean).all()
    temporary = ROOT / "Results" / "self_check"
    temporary.mkdir(parents=True, exist_ok=True)
    mini = SimpleNamespace(
        labels=np.arange(21), windows_per_run=24, length=20, normal_prefix=0,
        indices=np.array([fault * 24 + position for fault in range(21) for position in range(3)], dtype=np.int64),
    )
    class MiniStore:
        def __init__(self, attributes):
            self.__dict__.update(attributes.__dict__)

        def __len__(self):
            return len(self.indices)

        def batch(self, ids, active_device):
            return torch.zeros(len(ids), 33, 20, device=active_device), torch.from_numpy(self.labels[ids // self.windows_per_run]).to(active_device)
    mini = MiniStore(mini)
    cache = np.random.default_rng(42).normal(size=(21 * 24, 16)).astype(np.float32)
    short_config = dict(config, ridge_grid=[0.01, 0.1], batch_size=32)
    model = Koopman(16).to(device)
    initialize_ridge(model, mini, cache, mini, cache, short_config, device, temporary)
    summary, by_class = evaluate_dynamics(model, autoencoder, mini, cache, short_config, device)
    assert len(by_class) == 21 and all(math.isfinite(value) for value in summary.values())
    current, future, classes = dynamics_arrays(mini, cache, mini.indices[:16], device, config["horizon"])
    loss, _ = dynamics_loss(model, autoencoder, current, future, classes, config)
    loss.backward()
    assert model.inverse[0].weight.grad.abs().sum() > 0
    print(json.dumps({"self_check": "passed", "device": str(device), "all_dimensions": config["z_dimensions"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    configuration = load_config()
    if arguments.audit:
        print(json.dumps(audit_data(configuration), ensure_ascii=False), flush=True)
    elif arguments.self_check:
        self_check(configuration, arguments.device)
    else:
        run(arguments.seed, configuration, arguments.device)
