"""Wait for source-disjoint training, then cache and analyze predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def expected_tags(rates, seeds, folds):
    return [f"r{rate}_s{seed}_f{fold}" for rate in rates for seed in seeds for fold in folds]


def completed_tags(runs_root: Path, tags: list[str]) -> list[str]:
    return [
        tag
        for tag in tags
        if (runs_root / f"SD4_FT_{tag}" / "source_disjoint_training_complete.json").is_file()
    ]


def run(command: list[str], cwd: Path) -> None:
    print(f"[run] {subprocess.list2cmdline(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--cache-out", type=Path, required=True)
    parser.add_argument("--detector-metrics-out", type=Path)
    parser.add_argument("--analysis-out", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-wait-hours", type=float, default=48.0)
    parser.add_argument(
        "--cache-batch",
        type=int,
        default=8,
        help="Inference batch size for prediction caching; lower this when VRAM is shared.",
    )
    args = parser.parse_args()
    tags = expected_tags(args.rates, args.seeds, args.folds)
    started = time.time()
    previous = -1
    while True:
        completed = completed_tags(args.runs_root, tags)
        if len(completed) != previous:
            print(f"[training-progress] {len(completed)}/{len(tags)}", flush=True)
            previous = len(completed)
        if len(completed) == len(tags):
            break
        if time.time() - started > args.max_wait_hours * 3600:
            missing = sorted(set(tags) - set(completed))
            raise TimeoutError(f"training wait expired; missing={missing}")
        time.sleep(args.poll_seconds)

    cache_command = [
        str(args.python),
        str(args.workspace / "scripts" / "cache_source_disjoint_predictions.py"),
        "--protocol-root",
        str(args.protocol_root),
        "--runs-root",
        str(args.runs_root),
        "--out",
        str(args.cache_out),
        "--rates",
        *map(str, args.rates),
        "--seeds",
        *map(str, args.seeds),
        "--folds",
        *map(str, args.folds),
        "--batch",
        str(args.cache_batch),
        "--workers",
        "0",
        "--gpu-poll-seconds",
        "60",
    ]
    run(cache_command, args.workspace)

    detector_metrics_out = (
        args.detector_metrics_out
        or args.analysis_out.parent / f"{args.analysis_out.name}_detector_metrics"
    )
    if detector_metrics_out.exists():
        raise FileExistsError(
            f"refusing to overwrite detector metrics: {detector_metrics_out}"
        )
    detector_command = [
        str(args.python),
        str(args.workspace / "scripts" / "extract_detector_metrics.py"),
        "--runs-root",
        str(args.runs_root),
        "--run-name-template",
        "SD4_FT_r{rate}_s{seed}_f{fold}",
        "--protocol-root",
        str(args.protocol_root),
        "--out",
        str(detector_metrics_out),
        "--rates",
        *map(str, args.rates),
        "--seeds",
        *map(str, args.seeds),
        "--folds",
        *map(str, args.folds),
    ]
    run(detector_command, args.workspace)

    if args.analysis_out.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.analysis_out}")
    analysis_command = [
        str(args.python),
        str(args.workspace / "scripts" / "analyze_cached_rc_wssi_robustness.py"),
        "--cache-root",
        str(args.cache_out / "prediction_cache"),
        "--out",
        str(args.analysis_out),
        "--alphas",
        "0.10",
        "0.20",
        "0.30",
        "--deltas",
        "0.05",
        "0.10",
        "--bootstrap-draws",
        "5000",
    ]
    run(analysis_command, args.workspace)
    marker = {
        "status": "complete",
        "training_cells": len(tags),
        "cache_out": str(args.cache_out),
        "detector_metrics_out": str(detector_metrics_out),
        "analysis_out": str(args.analysis_out),
        "elapsed_sec": round(time.time() - started, 2),
    }
    (args.cache_out / "pipeline_complete.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )
    print(f"[pipeline-complete] {json.dumps(marker)}", flush=True)


if __name__ == "__main__":
    main()
