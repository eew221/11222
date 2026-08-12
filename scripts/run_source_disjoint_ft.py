"""Train YOLO baselines on the four-way source-disjoint protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


def completed_epochs(run_dir: Path) -> int:
    history = run_dir / "results.csv"
    if not history.is_file():
        return 0
    with history.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def gpu_processes() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        rows.append({"pid": pid, "process_name": fields[1]})
    return rows


def blocking_gpu_processes() -> list[dict]:
    current_pid = os.getpid()
    ignored_names = ("chatgpt.exe", "codex")
    return [
        row
        for row in gpu_processes()
        if row["pid"] != current_pid
        and not any(name in row["process_name"].lower() for name in ignored_names)
    ]


def wait_for_gpu(interval: int, max_wait_minutes: int) -> None:
    started = time.time()
    while True:
        blockers = blocking_gpu_processes()
        if not blockers:
            return
        print(f"[gpu-busy] {json.dumps(blockers)}", flush=True)
        if max_wait_minutes > 0 and time.time() - started >= max_wait_minutes * 60:
            raise TimeoutError("GPU remained busy beyond max wait")
        time.sleep(interval)


def marker_path(run_dir: Path) -> Path:
    return run_dir / "source_disjoint_training_complete.json"


def train_cell(args, rate: int, split_seed: int, fold: int) -> dict:
    from ultralytics import YOLO

    tag = f"r{rate}_s{split_seed}_f{fold}"
    name = f"SD4_FT_{tag}"
    run_dir = args.project / name
    yaml_path = (
        args.protocol_root
        / f"rate_{rate}pct"
        / f"seed{split_seed}"
        / f"fold{fold}"
        / "detector.yaml"
    )
    if not yaml_path.is_file():
        raise FileNotFoundError(f"missing protocol YAML: {yaml_path}")
    if marker_path(run_dir).is_file() and not args.force:
        record = json.loads(marker_path(run_dir).read_text(encoding="utf-8"))
        print(f"[skip] {tag}: audited completion marker exists", flush=True)
        return record

    if not args.allow_shared_gpu:
        wait_for_gpu(args.gpu_poll_seconds, args.max_gpu_wait_minutes)
    started = time.time()
    history_epochs = completed_epochs(run_dir)
    resume = (
        history_epochs > 0
        and (run_dir / "weights" / "last.pt").is_file()
        and not args.force
    )
    print(
        f"[{'resume' if resume else 'start'}] {tag} data={yaml_path} "
        f"existing_epochs={history_epochs}",
        flush=True,
    )
    if resume:
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(
            resume=True,
            device=args.device,
            workers=args.workers,
            batch=args.batch,
            imgsz=args.imgsz,
            cache=False,
            plots=False,
        )
    else:
        model = YOLO(str(args.model))
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            project=str(args.project),
            name=name,
            exist_ok=False,
            seed=20260805 + rate * 10000 + split_seed * 100 + fold,
            deterministic=True,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            plots=False,
            verbose=True,
            amp=args.amp,
            cache=False,
            mosaic=1.0,
            scale=0.5,
            fliplr=0.5,
        )
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"training did not produce best.pt and last.pt for {tag}")
    record = {
        "tag": tag,
        "name": name,
        "status": "complete",
        "protocol": "source_disjoint_four_way_v1",
        "data_yaml": str(yaml_path),
        "best": str(best),
        "last": str(last),
        "epochs_completed": completed_epochs(run_dir),
        "planned_epochs": args.epochs,
        "patience": args.patience,
        "checkpoint_selection_split": "detector_val",
        "elapsed_sec": round(time.time() - started, 2),
    }
    marker_path(run_dir).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[complete] {json.dumps(record)}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(r"D:\yolov8s.pt"))
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-poll-seconds", type=int, default=60)
    parser.add_argument("--max-gpu-wait-minutes", type=int, default=0)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Permit explicitly coordinated experiment shards to share an otherwise idle GPU.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.protocol_root.is_dir():
        parser.error(f"protocol root does not exist: {args.protocol_root}")
    if not args.model.is_file():
        parser.error(f"initial model does not exist: {args.model}")
    args.project.mkdir(parents=True, exist_ok=True)

    records = []
    for rate in args.rates:
        for seed in args.seeds:
            for fold in args.folds:
                records.append(train_cell(args, rate, seed, fold))
    summary = args.project / "source_disjoint_training_summary.json"
    summary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[summary] {summary}", flush=True)


if __name__ == "__main__":
    main()
