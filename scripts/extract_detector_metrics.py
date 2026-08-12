"""Extract base-detector validation metrics from the saved best checkpoints.

The script performs no inference. It selects the best row in each Ultralytics
results.csv by validation mAP50-95, verifies that row against the metrics stored
inside best.pt, and reports seed-cluster bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


from source_disjoint_validation import validate_protocol_cell
CSV_METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
CHECKPOINT_METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_bootstrap_ci(
    records: list[dict], key: str, *, rng_seed: int, draws: int
) -> tuple[float, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_seed[int(record["seed"])].append(float(record[key]))
    cluster_means = [mean(values) for _, values in sorted(by_seed.items())]
    rng = random.Random(rng_seed)
    samples = sorted(
        mean(cluster_means[rng.randrange(len(cluster_means))] for _ in cluster_means)
        for _ in range(draws)
    )
    return samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]


def load_checkpoint_metrics(path: Path, config_root: Path) -> tuple[dict, dict]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_root)
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_metrics = dict(checkpoint.get("train_metrics") or {})
    train_args = dict(checkpoint.get("train_args") or {})
    del checkpoint
    gc.collect()
    return train_metrics, train_args


def best_csv_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("empty results.csv")
    # Ultralytics 8.3.239 stores best.pt at the maximum validation mAP50-95.
    # A later epoch wins an exact tie, matching checkpoint replacement behavior.
    return max(
        rows,
        key=lambda row: (
            float(row[CSV_METRICS["map50_95"]]),
            int(float(row["epoch"])),
        ),
    )


def verify_metrics(
    tag: str, selected: dict[str, str], checkpoint_metrics: dict
) -> None:
    for short_name, csv_name in CSV_METRICS.items():
        checkpoint_name = CHECKPOINT_METRICS[short_name]
        if checkpoint_name not in checkpoint_metrics:
            raise ValueError(f"{tag}: checkpoint is missing {checkpoint_name}")
        csv_value = float(selected[csv_name])
        checkpoint_value = float(checkpoint_metrics[checkpoint_name])
        if abs(csv_value - checkpoint_value) > 5e-5:
            raise ValueError(
                f"{tag}: best-row {csv_name}={csv_value} does not match "
                f"best.pt train_metrics={checkpoint_value}"
            )


def latex_summary(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{@{}c c c c c@{}}",
        r"\toprule",
        r"Labels & Precision & Recall & mAP@0.50 & mAP@0.50:0.95 \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [f"{int(row['rate'])}\\%"]
        for key in ("precision", "recall", "map50", "map50_95"):
            cells.append(
                f"{float(row[key + '_mean']):.3f} "
                f"[{float(row[key + '_ci_low']):.3f}, {float(row[key + '_ci_high']):.3f}]"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path(r"D:\ppe_pilot\runs"))
    parser.add_argument(
        "--run-name-template", default="FT_r{rate}_s{seed}_f{fold}"
    )
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/detector_metrics_20260805"),
    )
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--draws", type=int, default=20_000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    config_root = Path(__file__).resolve().parents[1] / ".ultralytics"
    cell_rows: list[dict] = []
    for rate in args.rates:
        for seed in args.seeds:
            for fold in args.folds:
                cell_tag = f"r{rate}_s{seed}_f{fold}"
                tag = args.run_name_template.format(
                    rate=rate, seed=seed, fold=fold
                )
                run = args.runs_root / tag
                if args.protocol_root is not None:
                    cell_root = (
                        args.protocol_root
                        / f"rate_{rate}pct"
                        / f"seed{seed}"
                        / f"fold{fold}"
                    )
                    validate_protocol_cell(cell_root, run, cell_tag)
                results_path = run / "results.csv"
                weights_path = run / "weights" / "best.pt"
                if not results_path.is_file() or not weights_path.is_file():
                    raise FileNotFoundError(f"{tag}: missing results.csv or best.pt")
                history = read_csv(results_path)
                selected = best_csv_row(history)
                checkpoint_metrics, train_args = load_checkpoint_metrics(
                    weights_path, config_root
                )
                verify_metrics(tag, selected, checkpoint_metrics)
                planned_epochs = int(train_args.get("epochs", 100))
                record = {
                    "rate": rate,
                    "seed": seed,
                    "fold": fold,
                    "tag": tag,
                    "history_rows": len(history),
                    "planned_epochs": planned_epochs,
                    "early_stopped": int(len(history) < planned_epochs),
                    "best_epoch": int(float(selected["epoch"])),
                    "selection_metric": CSV_METRICS["map50_95"],
                    "checkpoint_verified": 1,
                }
                for short_name, csv_name in CSV_METRICS.items():
                    record[short_name] = float(selected[csv_name])
                cell_rows.append(record)
                print(
                    f"[{len(cell_rows):02d}/{len(args.rates) * len(args.seeds) * len(args.folds)}] "
                    f"{tag}: epoch={record['best_epoch']} "
                    f"mAP50-95={record['map50_95']:.5f}",
                    flush=True,
                )

    summary_rows: list[dict] = []
    for rate_index, rate in enumerate(args.rates):
        group = [row for row in cell_rows if int(row["rate"]) == rate]
        summary = {
            "rate": rate,
            "splits": len(group),
            "seed_clusters": len({int(row["seed"]) for row in group}),
            "best_epoch_mean": mean(float(row["best_epoch"]) for row in group),
            "best_epoch_min": min(int(row["best_epoch"]) for row in group),
            "best_epoch_max": max(int(row["best_epoch"]) for row in group),
            "early_stopped_splits": sum(int(row["early_stopped"]) for row in group),
        }
        for metric_index, key in enumerate(CSV_METRICS):
            low, high = cluster_bootstrap_ci(
                group,
                key,
                rng_seed=20260805 + rate_index * 101 + metric_index,
                draws=args.draws,
            )
            summary[f"{key}_mean"] = mean(float(row[key]) for row in group)
            summary[f"{key}_ci_low"] = low
            summary[f"{key}_ci_high"] = high
        summary_rows.append(summary)

    write_csv(args.out / "detector_metric_cells.csv", cell_rows)
    write_csv(args.out / "detector_metric_summary.csv", summary_rows)
    (args.out / "detector_metric_table.tex").write_text(
        latex_summary(summary_rows), encoding="ascii"
    )
    manifest = {
        "runs_root": str(args.runs_root.resolve()),
        "run_name_template": args.run_name_template,
        "protocol_root": str(args.protocol_root.resolve()) if args.protocol_root else None,
        "rates": args.rates,
        "seeds": args.seeds,
        "folds": args.folds,
        "selection_rule": (
            "maximum validation metrics/mAP50-95(B) in results.csv; later epoch "
            "wins an exact tie; all four metrics verified against best.pt train_metrics"
        ),
        "uncertainty": (
            f"95% percentile bootstrap with {args.draws} draws; five seeds are "
            "resampled as clusters and all three folds remain together"
        ),
        "gpu_inference": False,
        "checkpoint_verification_count": sum(
            int(row["checkpoint_verified"]) for row in cell_rows
        ),
    }
    (args.out / "detector_metric_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
