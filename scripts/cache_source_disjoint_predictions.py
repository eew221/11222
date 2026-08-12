"""Cache detector predictions for source-disjoint risk calibration and testing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import formal_worker_state_experiment as formal
import run_person_gate_batch as gate
from run_source_disjoint_ft import wait_for_gpu
from source_disjoint_validation import validate_existing_cache, validate_protocol_cell


EXPECTED_NAMES = {
    0: "helmet",
    1: "no_helmet",
    2: "no_reflective_vest",
    3: "person",
    4: "reflective_vest",
}


def normalize_names(names) -> dict[int, str]:
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(key): str(value) for key, value in (names or {}).items()}


def validate_names(model) -> dict[int, str]:
    names = normalize_names(getattr(model, "names", None))
    if names != EXPECTED_NAMES:
        raise RuntimeError(f"checkpoint class map mismatch: expected={EXPECTED_NAMES}, actual={names}")
    return names


def result_rows(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    return [
        {
            "cls": int(class_id),
            "conf": float(confidence),
            "xyxy": [float(value) for value in box],
        }
        for box, class_id, confidence in zip(xyxy, classes, confidences)
    ]


def predict_paths(model, paths: list[Path], args) -> dict[Path, list[dict]]:
    predictions = {}
    # A Python list is loaded as one in-memory source by Ultralytics, so its
    # predictor does not bound the actual forward batch with `batch=` alone.
    for start in range(0, len(paths), args.batch):
        batch_paths = paths[start : start + args.batch]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            device=args.device,
            batch=len(batch_paths),
            workers=args.workers,
            verbose=False,
            stream=True,
        )
        for path, result in zip(batch_paths, results):
            predictions[path] = result_rows(result)
    if len(predictions) != len(paths):
        raise RuntimeError(f"prediction count mismatch: {len(predictions)} != {len(paths)}")
    return predictions


def source_group_from_manifest(cell_root: Path) -> dict[str, str]:
    import csv

    mapping = {}
    with (cell_root / "manifest.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            mapping[row["image_name"]] = row["source"]
    return mapping


def cache_records(role: str, paths, predictions, ground_truth, sources) -> list[dict]:
    return [
        {
            "role": role,
            "image_path": str(path),
            "image_name": path.name,
            "source_group": sources[path.name],
            "predictions": predictions[path],
            "ground_truth": ground_truth[path],
        }
        for path in paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--gpu-poll-seconds", type=int, default=60)
    parser.add_argument("--max-gpu-wait-minutes", type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_root = args.out / "prediction_cache"
    cache_root.mkdir(exist_ok=True)

    from ultralytics import YOLO

    completed = []
    for rate in args.rates:
        for seed in args.seeds:
            for fold in args.folds:
                tag = f"r{rate}_s{seed}_f{fold}"
                cache_path = cache_root / f"{tag}.json"
                cell_root = (
                    args.protocol_root
                    / f"rate_{rate}pct"
                    / f"seed{seed}"
                    / f"fold{fold}"
                )
                run_dir = args.runs_root / f"SD4_FT_{tag}"
                weights = run_dir / "weights" / "best.pt"
                protocol = validate_protocol_cell(cell_root, run_dir, tag)
                if cache_path.is_file():
                    validate_existing_cache(cache_path, tag, weights)
                    print(f"[skip] {tag}: compatible audited cache exists", flush=True)
                    completed.append(tag)
                    continue
                sources = source_group_from_manifest(cell_root)
                calibration_gt = gate.collect_val_gt(cell_root, "calibration")
                test_gt = gate.collect_val_gt(cell_root, "test")
                calibration_paths = sorted(calibration_gt)
                test_paths = sorted(test_gt)
                wait_for_gpu(args.gpu_poll_seconds, args.max_gpu_wait_minutes)
                started = time.time()
                print(
                    f"[start] {tag} calibration={len(calibration_paths)} test={len(test_paths)}",
                    flush=True,
                )
                model = YOLO(str(weights))
                names = validate_names(model)
                calibration_predictions = predict_paths(model, calibration_paths, args)
                test_predictions = predict_paths(model, test_paths, args)
                payload = {
                    "tag": tag,
                    "protocol": "source_disjoint_four_way_v1",
                    "weights": str(weights),
                    "checkpoint_metadata_class_names": names,
                    "annotation_semantic_class_names": EXPECTED_NAMES,
                    "split_metadata": protocol["roles"],
                    "images": cache_records(
                        "calibration",
                        calibration_paths,
                        calibration_predictions,
                        calibration_gt,
                        sources,
                    )
                    + cache_records("test", test_paths, test_predictions, test_gt, sources),
                    "elapsed_sec": round(time.time() - started, 2),
                }
                temporary = cache_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                temporary.replace(cache_path)
                completed.append(tag)
                print(f"[complete] {tag} cache={cache_path}", flush=True)
                del model

    manifest = {
        "protocol": "source_disjoint_four_way_v1",
        "protocol_root": str(args.protocol_root),
        "runs_root": str(args.runs_root),
        "cells": completed,
        "cell_count": len(completed),
        "confidence_floor": args.conf,
        "imgsz": args.imgsz,
    }
    (args.out / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[summary] cached {len(completed)} cells", flush=True)


if __name__ == "__main__":
    main()
