from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from core import (
    ROOT, FEATURES, FAULTS, TRUTH, CNNGRU, TemporalConvAE, Koopman, WindowStore,
    freeze, load_config, load_models, report, save_csv, save_json, setup,
)
from deeplift_gru import explain as deeplift_explain, paired_forward


METHODS = ("route1_normal_ig", "route2_geodesic_ig", "koopman_ig", "saliency", "idgi_zero", "vanilla_ig_zero", "guided_ig_zero", "expected_ig", "deeplift_normal", "abigx")
METRICS = ("correctness_auc", "correctness_sum", "consistency_add", "consistency_del")


def gradients(model, path, target, config, device):
    scores, derivatives = [], []
    with torch.backends.cudnn.flags(enabled=False):
        for offset in range(0, len(path), config["gradient_batch_size"]):
            windows = path[offset:offset + config["gradient_batch_size"]].detach().to(device).requires_grad_(True)
            logits = model(windows)[:, target]
            gradient = torch.autograd.grad(logits.sum(), windows)[0]
            scores.append(logits.detach())
            derivatives.append(gradient.detach())
    return torch.cat(scores), torch.cat(derivatives)


def path_ig(model, path, target, config, device):
    scores, derivative = gradients(model, path, target, config, device)
    difference = path[1:].to(device) - path[:-1].to(device)
    attribution = ((derivative[1:] + derivative[:-1]) * difference / 2).sum(0)
    score_difference = scores[-1] - scores[0]
    error = float((attribution.sum() - score_difference).abs())
    return attribution, {"completeness_absolute_error": error, "score_difference": float(score_difference), "path_nodes": len(path)}, scores, derivative


def refine_polyline(path, intervals):
    refined = []
    for start, end in zip(path[:-1], path[1:]):
        alpha = torch.arange(intervals, device=path.device, dtype=path.dtype) / intervals
        shape = (intervals,) + (1,) * start.ndim
        refined.append(start[None] + alpha.reshape(shape) * (end - start)[None])
    refined.append(path[-1:])
    return torch.cat(refined)


def adaptive_ig(model, path, target, config, device, decoder=None, latent_nodes=None):
    intervals = max(1, math.ceil(config["ig_steps"] / (len(path) - 1)))
    while True:
        if decoder is None:
            refined = refine_polyline(path, intervals)
        else:
            latent_refined = refine_polyline(latent_nodes, intervals)
            with torch.no_grad():
                decoded = torch.cat([decoder(latent_refined[offset:offset + config["gradient_batch_size"]])
                                     for offset in range(0, len(latent_refined), config["gradient_batch_size"])])
                first = refine_polyline(torch.stack((path[0], decoded[0])), intervals)
                last = refine_polyline(torch.stack((decoded[-1], path[-1])), intervals)
                refined = torch.cat((first[:-1], decoded, last[1:]))
        attribution, diagnostics, _, _ = path_ig(model, refined, target, config, device)
        tolerance = config["ig_completeness_relative_tolerance"] * max(abs(diagnostics["score_difference"]), 1.0)
        converged = diagnostics["completeness_absolute_error"] <= tolerance
        if converged or 2 * (len(refined) - 1) > config["ig_max_steps"]:
            diagnostics["integration_converged"] = converged
            return attribution, diagnostics
        intervals *= 2


@torch.no_grad()
def exact_medoid(windows):
    flattened = windows.flatten(1).double()
    costs = []
    for offset in range(0, len(windows), 128):
        costs.append(torch.cdist(flattened[offset:offset + 128], flattened).sum(1))
    return int(torch.cat(costs).argmin())


@torch.no_grad()
def select_endpoints(model, store, trajectory, config, device):
    windows = store.view[trajectory]
    predictions = []
    for offset in range(0, len(windows), config["batch_size"]):
        predictions.append(model(windows[offset:offset + config["batch_size"]].to(device)).argmax(1).cpu())
    predictions = torch.cat(predictions)
    starts = torch.arange(len(windows))
    normal_candidates = starts[starts + config["window"] <= store.normal_prefix]
    fault_candidates = starts[(starts >= store.normal_prefix) & (predictions != 0)]
    if not len(fault_candidates):
        return None
    normal_position = exact_medoid(windows[normal_candidates].to(device))
    fault_position = exact_medoid(windows[fault_candidates].to(device))
    normal_start = int(normal_candidates[normal_position])
    fault_start = int(fault_candidates[fault_position])
    return {
        "trajectory": trajectory, "run": int(store.runs[trajectory]), "fault": int(store.labels[trajectory]),
        "normal_start": normal_start, "fault_start": fault_start,
        "normal_endpoint_sample": normal_start + config["window"],
        "fault_endpoint_sample": fault_start + config["window"],
        "target": int(predictions[fault_start]), "normal_candidates": len(normal_candidates),
        "fault_candidates": len(fault_candidates), "normal_medoid_prediction": int(predictions[normal_start]),
    }


def optimize_path(autoencoder, dynamics, observed, elapsed_steps, target, config, device):
    with torch.no_grad():
        states = autoencoder.encode(observed.to(device))
        if dynamics is not None:
            states = dynamics.encode(states)
    nodes = states.clone()
    interior = torch.nn.Parameter(nodes[1:-1].clone())
    optimizer = torch.optim.Adam([interior], lr=config["path_learning_rate"])
    decode = autoencoder.decode if dynamics is None else lambda values: dynamics.decode(values, autoencoder)
    best = math.inf
    best_nodes = nodes.clone()
    stale = 0
    history = []
    converged = False
    for iteration in range(config["path_optimization_steps"]):
        nodes = torch.cat((states[:1], interior, states[-1:]))
        decoded = decode(nodes)
        geometry = (decoded[1:] - decoded[:-1]).square().mean() * (len(nodes) - 1) ** 2
        if dynamics is None:
            validity = torch.zeros((), device=device)
            dynamic_error = validity
            objective = geometry
        else:
            validity = functional.mse_loss(dynamics.encode(dynamics.decode_h(nodes)), nodes)
            residuals = []
            for index, steps in enumerate(elapsed_steps):
                prediction = dynamics.advance_class(nodes[index:index + 1], target, int(steps))
                residuals.append((prediction[0] - nodes[index + 1]).square().mean())
            dynamic_error = torch.stack(residuals).mean()
            objective = config["path_geometry_weight"] * geometry + config["path_dynamics_weight"] * dynamic_error + config["path_validity_weight"] * validity
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite path objective")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = float(interior.grad.norm())
        value = float(objective.detach())
        if value < best - max(1e-9, abs(best) * 1e-7 if math.isfinite(best) else 0):
            best = value
            best_nodes = nodes.detach().clone()
            stale = 0
        else:
            stale += 1
        optimizer.step()
        history.append({"iteration": iteration + 1, "objective": value, "geometry": float(geometry.detach()),
                        "dynamics": float(dynamic_error.detach()), "validity": float(validity.detach()), "gradient_norm": gradient_norm})
        if gradient_norm < 1e-5 or (iteration >= 100 and stale >= 40):
            converged = True
            break
    with torch.no_grad():
        decoded = decode(best_nodes).detach()
        baseline = observed[0].to(device)
        endpoint = observed[-1].to(device)
        path = torch.cat((baseline[None], decoded, endpoint[None]))
        reconstruction = {
            "normal_endpoint_reconstruction_mse": float((decoded[0] - baseline).square().mean()),
            "fault_endpoint_reconstruction_mse": float((decoded[-1] - endpoint).square().mean()),
            "optimization_converged": converged, "iterations": len(history), "best_objective": best,
            "latent_endpoints_fixed": bool(torch.equal(best_nodes[0], states[0]) and torch.equal(best_nodes[-1], states[-1])),
        }
    return path, best_nodes, reconstruction, history


def guided_ig(model, endpoint, baseline, target, config, device):
    original = endpoint.cpu().numpy().astype(np.float64)
    reference = baseline.cpu().numpy().astype(np.float64)
    current = reference.copy()
    total_difference = original - reference
    total_distance = np.abs(total_difference).sum()
    attribution = np.zeros_like(original)
    if total_distance == 0:
        return torch.from_numpy(attribution).float().to(device), {"score_difference": 0.0}
    steps = config["ig_steps"]
    for step in range(steps):
        _, gradient = gradients(model, torch.from_numpy(current.astype(np.float32))[None], target, config, device)
        actual_gradient = gradient[0].cpu().numpy().astype(np.float64)
        available_gradient = actual_gradient.copy()
        alpha = (step + 1) / steps
        lower_alpha = max(alpha - config["guided_max_distance"], 0)
        upper_alpha = min(alpha + config["guided_max_distance"], 1)
        lower = reference + lower_alpha * total_difference
        upper = reference + upper_alpha * total_difference
        target_distance = total_distance * (1 - alpha)
        for _ in range(original.size + 2):
            previous = current.copy()
            location = np.divide(current - reference, total_difference, out=np.full_like(current, upper_alpha), where=total_difference != 0)
            current[location < lower_alpha] = lower[location < lower_alpha]
            distance = np.abs(current - original).sum()
            if math.isclose(distance, target_distance, rel_tol=1e-9, abs_tol=1e-9):
                attribution += (current - previous) * actual_gradient
                break
            available_gradient[current == upper] = np.inf
            threshold = np.quantile(np.abs(available_gradient), config["guided_fraction"], method="lower")
            selected = (np.abs(available_gradient) <= threshold) & np.isfinite(available_gradient)
            reduction = (np.abs(current - upper) * selected).sum()
            if reduction <= 0:
                raise RuntimeError("Guided IG cannot reach requested distance")
            fraction = (distance - target_distance) / reduction
            current[selected] += min(fraction, 1.0) * (upper - current)[selected]
            attribution += (current - previous) * actual_gradient
            if fraction <= 1:
                break
        else:
            raise RuntimeError("Guided IG inner iteration limit")
    with torch.no_grad():
        difference = float(model(endpoint[None])[0, target] - model(baseline[None])[0, target])
    return torch.from_numpy(attribution).float().to(device), {
        "score_difference": difference, "completeness_absolute_error": abs(float(attribution.sum()) - difference),
        "algorithm": "bounded Guided IG, lowest-absolute-gradient coordinate updates; left Riemann contributions",
    }


def afr_baseline(model, endpoint, representation_center, config, device):
    current = endpoint.detach().clone()
    best = current.clone()
    best_loss = math.inf
    initial_loss = None
    with torch.backends.cudnn.flags(enabled=False):
        for iteration in range(config["afr_steps"]):
            current = current.detach().requires_grad_(True)
            loss = functional.mse_loss(model.representation(current[None])[0], representation_center)
            if initial_loss is None:
                initial_loss = float(loss.detach())
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                best = current.detach().clone()
            gradient = torch.autograd.grad(loss, current)[0]
            with torch.no_grad():
                current = current - config["afr_learning_rate"] * gradient / gradient.norm().clamp_min(1e-12)
                displacement = current - endpoint
                displacement *= min(1.0, config["afr_radius"] / max(float(displacement.norm()), 1e-12))
                current = (endpoint + displacement).clamp(-1000, 1000)
    with torch.no_grad():
        predicted = int(model(best[None]).argmax(1))
    return best, {"afr_initial_representation_mse": initial_loss, "afr_final_representation_mse": best_loss,
                  "afr_predicted_class": predicted, "afr_normality_success": predicted == 0,
                  "afr_steps": config["afr_steps"], "afr_distance": float((best - endpoint).norm())}


@torch.no_grad()
def normal_background(model, config, device, seed, output):
    training = WindowStore(config, "train")
    trajectories = np.flatnonzero(training.labels == 0)
    generator = np.random.default_rng(seed)
    pool_ids = (trajectories[:, None] * training.windows_per_run + np.arange(training.windows_per_run)[None]).reshape(-1)
    center = torch.zeros(64, device=device, dtype=torch.float64)
    for offset in range(0, len(pool_ids), config["batch_size"]):
        windows, _ = training.batch(pool_ids[offset:offset + config["batch_size"]], device)
        center += model.representation(windows).double().sum(0)
    selected = generator.choice(pool_ids, size=config["expected_background_count"], replace=False)
    background, _ = training.batch(selected, device)
    save_json(output / "background.json", {"source": "fault-free training trajectories only", "expected_ig_window_ids": selected.tolist(), "afr_center_windows": len(pool_ids)})
    return background, (center / len(pool_ids)).float()


def root_auc(scores, positive):
    correct = scores[positive]
    other = scores[~positive]
    return float(((correct[:, None] > other[None]).astype(float) + 0.5 * (correct[:, None] == other[None])).mean())


@torch.no_grad()
def metrics(model, attribution, baseline, endpoint, target, fault, config, device):
    scores = attribution.abs().sum(1).detach().cpu().numpy().astype(float)
    scores /= max(scores.sum(), 1e-12)
    truth = np.array([name in TRUTH[fault] for name in FEATURES])
    order = np.argsort(-scores, kind="stable")
    additions, deletions = [], []
    for count in range(34):
        mask = torch.zeros(33, 1, device=device)
        mask[order[:count]] = 1
        additions.append(baseline + mask * (endpoint - baseline))
        deletions.append(endpoint - mask * (endpoint - baseline))
    points = torch.stack(additions + deletions)
    probabilities = []
    for offset in range(0, len(points), config["gradient_batch_size"]):
        probabilities.append(model(points[offset:offset + config["gradient_batch_size"]]).softmax(1)[:, target])
    probability = torch.cat(probabilities).cpu().numpy()
    add_curve, delete_curve = probability[:34], probability[34:]
    integration = np.trapz
    return {
        "correctness_auc": root_auc(scores, truth), "correctness_sum": float(scores[truth].sum()),
        "consistency_add": float(integration(add_curve, dx=1 / 33)),
        "consistency_del": float(integration(delete_curve, dx=1 / 33)),
        "top3": ";".join(FEATURES[index] for index in order[:3]),
        "top3_hit": int(truth[order[:3]].any()), "top3_coverage": float(truth[order[:3]].sum() / truth.sum()),
        "top3_precision": float(truth[order[:3]].mean()),
        "baseline_target_probability": float(add_curve[0]), "endpoint_target_probability": float(add_curve[-1]),
    }, scores, add_curve, delete_curve


def attribution_figure(attribution, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = attribution.detach().cpu().numpy()
    signed = values.sum(1)
    absolute = np.abs(values).sum(1)
    positions = np.arange(33)
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, data, name in zip(axes, (signed, absolute), ("Signed contribution", "Absolute contribution")):
        bars = axis.bar(positions, data)
        axis.bar_label(bars, fmt="%.2g", fontsize=5, rotation=90)
        axis.set_ylabel(name)
    axes[-1].set_xticks(positions, FEATURES, rotation=90)
    figure.suptitle(title)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(Path(path).with_suffix(f".{suffix}"), dpi=160)
    plt.close(figure)


def run(seed, config, device_choice="auto"):
    device = setup(seed, config, device_choice)
    classifier, autoencoder, dynamics = load_models(seed, device)
    output = ROOT / "Results" / f"seed_{seed}" / "explanations"
    output.mkdir(parents=True, exist_ok=True)
    store = WindowStore(config, "test")
    saved_endpoints = output.parent / "endpoint_selection.json"
    endpoint_cache = None
    if saved_endpoints.exists():
        cache = json.loads(saved_endpoints.read_text(encoding="utf-8"))
        classifier_hash = hashlib.sha256((output.parent / "classifier.pt").read_bytes()).hexdigest()
        if cache["classifier_sha256"] != classifier_hash:
            raise ValueError("Saved endpoints belong to a different classifier")
        endpoint_cache = {record["trajectory"]: record for record in cache["endpoints"]}
        for failure in cache["failures"]:
            endpoint_cache[failure["trajectory"]] = None
    background, representation_center = normal_background(classifier, config, device, seed, output)
    all_rows = []
    failures = []
    endpoints = []
    for fault in FAULTS:
        trajectories = np.flatnonzero(store.labels == fault)
        for trajectory in trajectories:
            directory = output / f"Fault_{fault:02d}" / f"run_{store.runs[trajectory]:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            completed = directory / "complete.json"
            if completed.exists():
                record = json.loads(completed.read_text())
                all_rows.extend(record["rows"])
                if record.get("endpoint"):
                    endpoints.append(record["endpoint"])
                if record.get("failure"):
                    failures.append(record["failure"])
                continue
            began = time.monotonic()
            metadata = endpoint_cache[int(trajectory)] if endpoint_cache is not None else select_endpoints(classifier, store, int(trajectory), config, device)
            if metadata is None:
                failure = {"fault": fault, "trajectory": int(trajectory), "run": int(store.runs[trajectory]), "reason": "no fully post-injection window predicted as Fault"}
                failures.append(failure)
                save_json(completed, {"rows": [], "failure": failure})
                continue
            endpoints.append(metadata)
            target = metadata["target"]
            baseline = store.view[trajectory, metadata["normal_start"]].to(device)
            endpoint = store.view[trajectory, metadata["fault_start"]].to(device)
            zero = torch.zeros_like(baseline)
            interval_count = min(config["path_max_intervals"], metadata["fault_start"] - metadata["normal_start"])
            starts = np.rint(np.linspace(metadata["normal_start"], metadata["fault_start"], interval_count + 1)).astype(int)
            observed = store.view[trajectory, torch.from_numpy(starts)].to(device)
            elapsed = np.diff(starts)
            save_json(directory / "endpoints.json", metadata)
            np.savez_compressed(directory / "endpoints.npz", baseline=baseline.cpu().numpy(), endpoint=endpoint.cpu().numpy(), window_starts=starts)
            rows = []
            for method in METHODS:
                saved_row = directory / f"{method}.json"
                if saved_row.exists():
                    row = json.loads(saved_row.read_text())
                    rows.append(row)
                    continue
                report(output, "explaining", seed=seed, fault=fault, run=int(store.runs[trajectory]), method=method, target=target)
                diagnostics = {}
                if method == "saliency":
                    _, gradient = gradients(classifier, endpoint[None], target, config, device)
                    attribution = gradient[0]
                elif method == "deeplift_normal":
                    attribution, diagnostics = deeplift_explain(classifier, endpoint, baseline, target)
                elif method == "guided_ig_zero":
                    attribution, diagnostics = guided_ig(classifier, endpoint, zero, target, config, device)
                elif method == "expected_ig":
                    values, checks = [], []
                    for reference in background:
                        value, check = adaptive_ig(classifier, torch.stack((reference, endpoint)), target, config, device)
                        values.append(value)
                        checks.append(check)
                    attribution = torch.stack(values).mean(0)
                    diagnostics = {"background_count": len(background), "mean_completeness_absolute_error": float(np.mean([check["completeness_absolute_error"] for check in checks]))}
                elif method == "idgi_zero":
                    alpha = torch.linspace(0, 1, config["ig_steps"] + 1, device=device)
                    path = zero[None] + alpha[:, None, None] * (endpoint - zero)[None]
                    scores, derivative = gradients(classifier, path, target, config, device)
                    weights = derivative[:-1].square()
                    denominator = weights.sum((1, 2), keepdim=True)
                    attribution = (weights / denominator.clamp_min(1e-20) * (scores[1:] - scores[:-1])[:, None, None]).sum(0)
                    diagnostics = {"zero_gradient_segments": int((denominator == 0).sum()), "completeness_absolute_error": float((attribution.sum() - (scores[-1] - scores[0])).abs())}
                elif method in ("route2_geodesic_ig", "koopman_ig"):
                    path, nodes, path_check, history = optimize_path(autoencoder, dynamics if method == "koopman_ig" else None, observed, elapsed, target, config, device)
                    decoder = autoencoder.decode if method == "route2_geodesic_ig" else lambda values: dynamics.decode(values, autoencoder)
                    attribution, diagnostics = adaptive_ig(classifier, path, target, config, device, decoder=decoder, latent_nodes=nodes)
                    diagnostics.update(path_check)
                    save_csv(directory / f"{method}_optimization.csv", history)
                    np.savez_compressed(directory / f"{method}_path.npz", original_space_path=path.cpu().numpy(), latent_path=nodes.cpu().numpy(), physical_steps=elapsed)
                    connector_attr = torch.zeros_like(attribution)
                    for connector in (path[:2], path[-2:]):
                        value, _ = adaptive_ig(classifier, connector, target, config, device)
                        connector_attr += value
                    diagnostics["endpoint_connector_absolute_attribution_ratio"] = float(connector_attr.abs().sum() / attribution.abs().sum().clamp_min(1e-12))
                else:
                    reference = baseline if method == "route1_normal_ig" else zero
                    if method == "abigx":
                        reference, diagnostics = afr_baseline(classifier, endpoint, representation_center, config, device)
                        np.save(directory / "afr_baseline.npy", reference.cpu().numpy())
                    attribution, check = adaptive_ig(classifier, torch.stack((reference, endpoint)), target, config, device)
                    diagnostics.update(check)
                metric, scores, add_curve, delete_curve = metrics(classifier, attribution, baseline, endpoint, target, fault, config, device)
                row = {"seed": seed, "method": method, "fault": fault, "run": int(store.runs[trajectory]), "target": target,
                       "classification_correct": target == fault, **metric}
                np.savez_compressed(directory / f"{method}.npz", attribution=attribution.cpu().numpy(), feature_scores=scores, add_curve=add_curve, delete_curve=delete_curve)
                save_csv(directory / f"{method}_attribution.csv", [{"feature": name, "signed": float(attribution[index].sum()), "absolute": float(attribution[index].abs().sum()), "normalized_absolute": float(scores[index])} for index, name in enumerate(FEATURES)])
                save_json(directory / f"{method}_diagnostics.json", diagnostics)
                save_json(saved_row, row)
                if config["plot_trajectory_attributions"]:
                    attribution_figure(attribution, directory / method, f"{method}: Fault {fault}, run {store.runs[trajectory]}")
                rows.append(row)
            save_json(completed, {"rows": rows, "endpoint": metadata, "elapsed_seconds": time.monotonic() - began})
            all_rows.extend(rows)
            save_csv(output / "per_trajectory.csv", all_rows)
            save_json(output / "failures.json", failures)
    save_csv(output / "per_trajectory.csv", all_rows)
    save_json(output / "endpoints.json", endpoints)
    save_json(output / "failures.json", failures)
    save_json(output / "complete.json", {"valid_trajectories": len(endpoints), "requested_trajectories": 700, "failures": len(failures), "methods": METHODS})
    report(output, "explanation_complete", seed=seed, rows=len(all_rows), valid_trajectories=len(endpoints))


def self_check():
    config = load_config()
    config["threads"] = 2
    device = setup(42, config, "cpu")
    model = freeze(CNNGRU())
    endpoint = torch.randn(33, 20)
    baseline = torch.randn(33, 20)
    paired, reference = paired_forward(model, endpoint[None], baseline[None])
    forward_error = float((paired - model(endpoint[None])).abs().max())
    reference_error = float((reference - model(baseline[None])).abs().max())
    attribution, diagnostics = deeplift_explain(model, endpoint, baseline, 3)
    assert forward_error < 1e-5 and reference_error < 1e-5
    assert diagnostics["completeness_absolute_error"] < 1e-4, diagnostics
    _, check = adaptive_ig(model, torch.stack((baseline, endpoint)), 3, config, device)
    assert check["integration_converged"]
    config["ig_steps"] = 8
    guided, _ = guided_ig(model, endpoint, torch.zeros_like(baseline), 3, config, device)
    assert torch.isfinite(guided).all()
    linear = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(660, 21, bias=False))
    with torch.no_grad():
        linear[1].weight.fill_(0)
        linear[1].weight[3, :20] = 1
    linear = freeze(linear)
    linear_attr, linear_check = adaptive_ig(linear, torch.stack((baseline, endpoint)), 3, config, device)
    assert linear_check["completeness_absolute_error"] < 1e-4
    metric, _, add_curve, delete_curve = metrics(linear, linear_attr, baseline, endpoint, 3, 6, config, device)
    assert np.allclose(add_curve[[0, -1]], delete_curve[[-1, 0]])
    assert root_auc(np.ones(33), np.arange(33) < 2) == 0.5
    samples = torch.tensor([[[0.0]], [[1.0]], [[10.0]]])
    assert exact_medoid(samples) == 1
    autoencoder = freeze(TemporalConvAE())
    dynamics = freeze(Koopman(32))
    config["path_optimization_steps"] = 3
    observed = torch.stack((baseline, (baseline + endpoint) / 2, endpoint))
    for active in (None, dynamics):
        path, _, checks, _ = optimize_path(autoencoder, active, observed, np.array([2, 3]), 3, config, device)
        assert checks["latent_endpoints_fixed"]
        assert torch.equal(path[0], baseline) and torch.equal(path[-1], endpoint)
    print(json.dumps({"explanation_self_check": "passed", "gru_forward_max_error": forward_error,
                      "gru_reference_max_error": reference_error, "deeplift": diagnostics, "path_ig": check, "add_del": metric}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check()
    else:
        run(arguments.seed, load_config(), arguments.device)
