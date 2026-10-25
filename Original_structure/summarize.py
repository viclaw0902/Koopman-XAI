from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import ROOT, FEATURES, FAULTS, load_config, save_csv, save_json
from explain import METHODS, METRICS, attribution_figure


LABELS = {
    "route1_normal_ig": "Route 1: normal-medoid IG",
    "route2_geodesic_ig": "Route 2: manifold geodesic IG",
    "koopman_ig": "Koopman IG",
    "saliency": "Saliency Map", "idgi_zero": "IDGI (zero)",
    "vanilla_ig_zero": "Vanilla IG (zero)", "guided_ig_zero": "Guided IG (zero)",
    "expected_ig": "Expected IG", "deeplift_normal": "DeepLIFT (normal)", "abigx": "ABIGX",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_rows(rows):
    grouped = {}
    for row in rows:
        key = (int(row["seed"]), int(row["fault"]), int(row["run"]))
        grouped.setdefault(key, {})[row["method"]] = row
    return [row for methods in grouped.values() if set(methods) == set(METHODS) for row in methods.values()]


def aggregate(rows):
    faults = []
    seeds = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        for method in METHODS:
            current_faults = []
            for fault in FAULTS:
                current = [row for row in rows if int(row["seed"]) == seed and row["method"] == method and int(row["fault"]) == fault]
                if not current:
                    continue
                record = {"seed": seed, "method": method, "fault": fault, "trajectories": len(current),
                          **{metric: float(np.mean([float(row[metric]) for row in current])) for metric in METRICS}}
                faults.append(record)
                current_faults.append(record)
            if len(current_faults) == len(FAULTS):
                seeds.append({"seed": seed, "method": method, "faults": len(current_faults),
                              "trajectories": sum(row["trajectories"] for row in current_faults),
                              **{metric: float(np.mean([row[metric] for row in current_faults])) for metric in METRICS}})
    summary = []
    for method in METHODS:
        current = [row for row in seeds if row["method"] == method]
        if not current:
            continue
        record = {"method": method, "seeds": len(current)}
        for metric in METRICS:
            values = [row[metric] for row in current]
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(record)
    return faults, seeds, summary


def save_figure(figure, path):
    figure.tight_layout()
    for extension in ("png", "svg"):
        figure.savefig(Path(path).with_suffix(f".{extension}"), dpi=180)
    plt.close(figure)


def boxplots(rows, methods, path, title, level):
    if not rows:
        return
    figure, axes = plt.subplots(2, 2, figsize=(max(11, len(methods) * 1.4), 10))
    generator = np.random.default_rng(42)
    for axis, metric in zip(axes.flat, METRICS):
        arrays = [[float(row[metric]) for row in rows if row["method"] == method] for method in methods]
        axis.boxplot(arrays, labels=[LABELS[method] for method in methods], showmeans=True)
        for position, values in enumerate(arrays, start=1):
            if not values:
                continue
            jitter = generator.uniform(-0.08, 0.08, len(values))
            axis.scatter(position + jitter, values, s=18 if level == "seed" else 4, alpha=0.8 if level == "seed" else 0.15)
            axis.annotate(f"mean={np.mean(values):.4f}\nn={len(values)}", (position, np.max(values)),
                          xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
        axis.set_title(f"{metric} {'(lower is better)' if metric == 'consistency_del' else '(higher is better)'}")
        axis.tick_params(axis="x", labelrotation=55)
        axis.grid(axis="y", alpha=0.2)
        axis.margins(y=0.2)
    figure.suptitle(f"{title}\nIndependent unit: {level}")
    save_figure(figure, path)


def classification_figures(output, seeds):
    records = []
    for seed in seeds:
        directory = ROOT / "Results" / f"seed_{seed}"
        metrics = directory / "classification_metrics.json"
        if not metrics.exists():
            continue
        record = json.loads(metrics.read_text())
        records.append({"seed": seed, **record})
        confusion_rows = read_csv(directory / "confusion_matrix.csv")
        matrix = np.array([[int(row[f"pred_{column}"]) for column in range(21)] for row in confusion_rows])
        figure, axis = plt.subplots(figsize=(13, 11))
        normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        names = ["Normal"] + [f"Fault{fault}" for fault in range(1, 21)]
        axis.set_xticks(range(21), names, rotation=90)
        axis.set_yticks(range(21), names)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"Shared CNN-GRU, seed {seed}; accuracy={record['accuracy']:.4f}")
        for row in range(21):
            for column in range(21):
                axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center", fontsize=5,
                          color="white" if normalized[row, column] > 0.5 else "black")
        figure.colorbar(image, ax=axis)
        save_figure(figure, output / f"confusion_matrix_seed_{seed}")
    save_csv(output / "classification_by_seed.csv", records)
    if records:
        save_json(output / "classification_summary.json", {
            "shared_classifier": True,
            "mean": {metric: float(np.mean([row[metric] for row in records])) for metric in ("accuracy", "balanced_accuracy", "macro_f1")},
            "std": {metric: float(np.std([row[metric] for row in records], ddof=1)) if len(records) > 1 else 0 for metric in ("accuracy", "balanced_accuracy", "macro_f1")},
            "completed_seeds": [row["seed"] for row in records],
        })


def aggregate_attributions(rows, output):
    import torch
    for method in METHODS:
        for fault in FAULTS:
            selected = [row for row in rows if row["method"] == method and int(row["fault"]) == fault]
            if not selected:
                continue
            signed = []
            absolute = []
            by_seed = {}
            for row in selected:
                archive_path = ROOT / "Results" / f"seed_{row['seed']}" / "explanations" / f"Fault_{fault:02d}" / f"run_{int(row['run']):03d}" / f"{method}.npz"
                attribution = np.load(archive_path)["attribution"]
                by_seed.setdefault(row["seed"], []).append(attribution)
            for values in by_seed.values():
                stacked = np.stack(values)
                signed.append(stacked.sum(2).mean(0))
                absolute.append(np.abs(stacked).sum(2).mean(0))
            signed_mean = np.mean(signed, 0)
            absolute_mean = np.mean(absolute, 0)
            destination = output / "feature_attributions" / method
            destination.mkdir(parents=True, exist_ok=True)
            save_csv(destination / f"Fault_{fault:02d}.csv", [{"feature": feature, "signed_mean": float(signed_mean[index]), "absolute_mean": float(absolute_mean[index])} for index, feature in enumerate(FEATURES)])
            figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            for axis, values, label in zip(axes, (signed_mean, absolute_mean), ("Mean signed contribution", "Mean absolute contribution")):
                bars = axis.bar(range(33), values)
                axis.bar_label(bars, fmt="%.2g", fontsize=5, rotation=90)
                axis.set_ylabel(label)
            axes[-1].set_xticks(range(33), FEATURES, rotation=90)
            figure.suptitle(f"{LABELS[method]} / Fault {fault}; mean over trajectories, then seeds")
            save_figure(figure, destination / f"Fault_{fault:02d}")


def main(include_attributions=True):
    config = load_config()
    output = ROOT / "Results" / "Summary"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    completeness = []
    coverage = []
    for seed in config["seeds"]:
        source = ROOT / "Results" / f"seed_{seed}" / "explanations"
        if (source / "per_trajectory.csv").exists():
            rows.extend(read_csv(source / "per_trajectory.csv"))
        if (source / "complete.json").exists():
            coverage.append({"seed": seed, **json.loads((source / "complete.json").read_text())})
    rows = paired_rows(rows)
    faults, seeds, summary = aggregate(rows)
    save_csv(output / "per_trajectory_paired.csv", rows)
    save_csv(output / "per_fault.csv", faults)
    save_csv(output / "per_seed.csv", seeds)
    save_csv(output / "mean_std.csv", summary)
    save_json(output / "coverage.json", coverage)
    ablation = ("route1_normal_ig", "route2_geodesic_ig", "koopman_ig")
    comparison = ("vanilla_ig_zero", "guided_ig_zero", "expected_ig", "idgi_zero", "saliency", "deeplift_normal", "abigx", "koopman_ig")
    for methods, name in ((ablation, "Ablation"), (comparison, "Method_comparison")):
        boxplots(seeds, methods, output / f"{name}_seed_boxplot", name, "seed")
        boxplots(rows, methods, output / f"{name}_trajectory_boxplot", name, "trajectory nested within seed (descriptive only)")
    classification_figures(output, config["seeds"])
    if include_attributions:
        aggregate_attributions(rows, output)
    save_json(output / "status.json", {"paired_rows": len(rows), "seed_level_rows": len(seeds),
                                      "complete_seeds": [row["seed"] for row in coverage],
                                      "requested_seeds": config["seeds"], "not_final_until_all_seeds_complete": len(coverage) != len(config["seeds"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-attribution-figures", action="store_true")
    arguments = parser.parse_args()
    main(not arguments.skip_attribution_figures)
