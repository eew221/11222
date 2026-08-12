"""Independently audit a four-way source-disjoint PPE protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml

from source_disjoint_validation import EXPECTED_NAMES, normalize_names


ROLES = ("train", "detector_val", "calibration", "test")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_digest(path: Path, cache: dict[Path, str]) -> str:
    path = path.resolve()
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def audit_cell(
    cell_root: Path,
    rate: int,
    seed: int,
    fold: int,
    verify_files: bool,
    digest_cache: dict[Path, str],
) -> dict:
    tag = f"r{rate}_s{seed}_f{fold}"
    protocol_path = cell_root / "protocol.json"
    manifest_path = cell_root / "manifest.csv"
    yaml_path = cell_root / "detector.yaml"
    required = (protocol_path, manifest_path, yaml_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{tag}: missing protocol files: {missing}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    rows = read_csv(manifest_path)
    if protocol.get("protocol") != "source_disjoint_four_way_v1":
        raise RuntimeError(f"{tag}: unexpected protocol identifier")
    if int(protocol.get("training_label_rate_percent", -1)) != rate:
        raise RuntimeError(f"{tag}: training-rate metadata mismatch")
    if config.get("train") != "images/train" or config.get("val") != "images/detector_val":
        raise RuntimeError(f"{tag}: detector YAML does not isolate detector validation")
    if normalize_names(config.get("names")) != EXPECTED_NAMES:
        raise RuntimeError(f"{tag}: detector YAML class-map mismatch")

    by_role: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_role[row["role"]].append(row)
    if set(by_role) != set(ROLES):
        raise RuntimeError(f"{tag}: unexpected roles {sorted(by_role)}")
    source_sets = {role: {row["source"] for row in by_role[role]} for role in ROLES}
    hash_sets = {role: {row["image_sha256"] for row in by_role[role]} for role in ROLES}
    for left, right in itertools.combinations(ROLES, 2):
        if source_sets[left] & source_sets[right]:
            raise RuntimeError(f"{tag}: source leakage {left}/{right}")
        if hash_sets[left] & hash_sets[right]:
            raise RuntimeError(f"{tag}: content leakage {left}/{right}")
    for role in ROLES:
        if len(hash_sets[role]) != len(by_role[role]):
            raise RuntimeError(f"{tag}: duplicate image content within {role}")
        summary = protocol["roles"][role]
        if int(summary["images"]) != len(by_role[role]):
            raise RuntimeError(f"{tag}: {role} image count differs from protocol")
        if set(summary["sources"]) != source_sets[role]:
            raise RuntimeError(f"{tag}: {role} source list differs from protocol")
    for section in protocol.get("leakage_audit", {}).values():
        if any(int(value) != 0 for value in section.values()):
            raise RuntimeError(f"{tag}: stored leakage audit is nonzero")

    expected_train = max(
        len(source_sets["train"]),
        int(round(int(protocol["training_pool_images"]) * rate / 100.0)),
    )
    if len(by_role["train"]) != expected_train:
        raise RuntimeError(f"{tag}: source-restricted label budget mismatch")

    if verify_files:
        for row in rows:
            original = Path(row["original_image"])
            image = cell_root / "images" / row["role"] / row["image_name"]
            label = cell_root / "labels" / row["role"] / f"{Path(row['image_name']).stem}.txt"
            if not original.is_file() or not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"{tag}: missing materialized data for {row['image_name']}")
            if not os.path.samefile(original, image):
                if file_digest(original, digest_cache) != file_digest(image, digest_cache):
                    raise RuntimeError(f"{tag}: copied image differs from source")
            if file_digest(original, digest_cache) != row["image_sha256"]:
                raise RuntimeError(f"{tag}: manifest hash differs from source")

    return {
        "tag": tag,
        "rate": rate,
        "seed": seed,
        "fold": fold,
        **{f"{role}_images": len(by_role[role]) for role in ROLES},
        **{f"{role}_sources": len(source_sets[role]) for role in ROLES},
        **{f"{role}_source_ids": "|".join(sorted(source_sets[role])) for role in ROLES},
        "path_overlap_count": 0,
        "source_overlap_count": 0,
        "hash_overlap_count": 0,
        "status": "pass",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    digest_cache: dict[Path, str] = {}
    rows = []
    for rate in args.rates:
        for seed in args.seeds:
            for fold in args.folds:
                cell_root = args.protocol_root / f"rate_{rate}pct" / f"seed{seed}" / f"fold{fold}"
                rows.append(audit_cell(cell_root, rate, seed, fold, args.verify_files, digest_cache))
                print(f"[pass] r{rate}_s{seed}_f{fold}", flush=True)

    for fold in args.folds:
        fold_rows = [row for row in rows if row["fold"] == fold]
        for role in ROLES:
            if len({row[f"{role}_source_ids"] for row in fold_rows}) != 1:
                raise RuntimeError(f"fold {fold}: {role} sources vary across cells")
    test_sets = {
        fold: set(next(row for row in rows if row["fold"] == fold)["test_source_ids"].split("|"))
        for fold in args.folds
    }
    for left, right in itertools.combinations(args.folds, 2):
        if test_sets[left] & test_sets[right]:
            raise RuntimeError(f"outer test folds {left} and {right} overlap")

    write_csv(args.out / "cell_audit.csv", rows)
    summary = {
        "status": "pass",
        "protocol": "source_disjoint_four_way_v1",
        "protocol_root": str(args.protocol_root.resolve()),
        "cells": len(rows),
        "rates": args.rates,
        "seeds": args.seeds,
        "folds": args.folds,
        "verify_files": args.verify_files,
        "path_overlap_max": 0,
        "source_overlap_max": 0,
        "hash_overlap_max": 0,
        "outer_test_sources": {str(fold): sorted(test_sets[fold]) for fold in args.folds},
    }
    (args.out / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
