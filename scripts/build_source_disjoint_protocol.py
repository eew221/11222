"""Build a four-way source-disjoint PPE protocol.

The protocol keeps detector training, detector checkpoint validation, risk
calibration, and final testing disjoint at the inferred tunnel/video source
level. Training labels are sampled at the requested percentage of the
source-restricted training pool; validation, calibration, and test labels are
held-out evaluation resources and do not count toward the training budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet",
    2: "no_reflective_vest",
    3: "person",
    4: "reflective_vest",
}


def source_group_id(path: Path) -> str:
    """Remove frame indices and accidental repeated image suffixes."""
    stem = path.stem
    while True:
        previous = stem
        stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
        if Path(stem).suffix.lower() in IMAGE_SUFFIXES:
            stem = Path(stem).stem
        if stem == previous:
            break
    return stem.strip() or path.stem


def label_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            counts[int(float(fields[0]))] += 1
    return counts


def collect_records(dataset_root: Path) -> list[dict]:
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError("dataset root must contain images/ and labels/")
    records = []
    for image in sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = label_root / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing label for {image}: {label}")
        counts = label_counts(label)
        records.append(
            {
                "image": image,
                "label": label,
                "name": image.name,
                "source": source_group_id(image),
                "counts": counts,
            }
        )
    if not records:
        raise RuntimeError(f"no images found under {image_root}")
    return records


def source_statistics(records: list[dict]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)
    for source, rows in grouped.items():
        classes = Counter()
        for row in rows:
            classes.update(row["counts"])
        stats[source] = {
            "images": len(rows),
            "persons": classes[3],
            "unsafe_evidence": classes[1] + classes[2],
            **{f"class_{index}": classes[index] for index in CLASS_NAMES},
        }
    return stats


def fold_score(assignments: dict[str, int], stats: dict[str, dict[str, int]]) -> float:
    score = 0.0
    feature_weights = {"images": 1.0, "persons": 0.6, "unsafe_evidence": 1.5}
    for feature, weight in feature_weights.items():
        total = sum(row[feature] for row in stats.values())
        target = total / 3.0
        for fold in range(3):
            value = sum(stats[source][feature] for source, part in assignments.items() if part == fold)
            score += weight * ((value - target) / max(target, 1.0)) ** 2
    target_groups = len(stats) / 3.0
    for fold in range(3):
        count = sum(part == fold for part in assignments.values())
        score += 0.08 * ((count - target_groups) / target_groups) ** 2
    largest = sorted(stats, key=lambda source: (-stats[source]["images"], source))[:3]
    for left, right in itertools.combinations(largest, 2):
        if assignments[left] == assignments[right]:
            score += 5.0
    return score


def choose_outer_folds(stats: dict[str, dict[str, int]]) -> dict[str, int]:
    sources = sorted(stats)
    if len(sources) < 9:
        raise ValueError("at least nine source groups are required for the four-way protocol")
    anchor = max(sources, key=lambda source: (stats[source]["images"], source))
    remaining = [source for source in sources if source != anchor]
    best = None
    for values in itertools.product(range(3), repeat=len(remaining)):
        assignment = {anchor: 0, **dict(zip(remaining, values))}
        counts = [sum(part == fold for part in assignment.values()) for fold in range(3)]
        if min(counts) < 3 or max(counts) > 4:
            continue
        if any(
            sum(stats[source]["unsafe_evidence"] for source, part in assignment.items() if part == fold) == 0
            for fold in range(3)
        ):
            continue
        candidate = (fold_score(assignment, stats), tuple(assignment[source] for source in sources))
        if best is None or candidate < best[0]:
            best = (candidate, assignment)
    if best is None:
        raise RuntimeError("could not construct balanced outer source folds")
    return best[1]


def role_totals(sources: set[str], stats: dict[str, dict[str, int]]) -> dict[str, int]:
    return {
        feature: sum(stats[source][feature] for source in sources)
        for feature in ("images", "persons", "unsafe_evidence")
    }


def choose_inner_roles(
    test_sources: set[str], stats: dict[str, dict[str, int]]
) -> dict[str, set[str]]:
    remaining = sorted(set(stats) - test_sources)
    if len(remaining) < 7:
        raise ValueError("not enough non-test sources for detector-val, calibration, and training")
    must_train = set(
        sorted(remaining, key=lambda source: (-stats[source]["images"], source))[:2]
    )
    candidates = [source for source in remaining if source not in must_train]
    all_totals = role_totals(set(remaining), stats)
    targets = {"detector_val": 0.15, "calibration": 0.18, "train": 0.67}
    best = None
    for detector_tuple in itertools.combinations(candidates, 2):
        detector = set(detector_tuple)
        cal_candidates = [source for source in candidates if source not in detector]
        for calibration_tuple in itertools.combinations(cal_candidates, 2):
            calibration = set(calibration_tuple)
            train = set(remaining) - detector - calibration
            if len(train) < 3 or not must_train.issubset(train):
                continue
            roles = {
                "train": train,
                "detector_val": detector,
                "calibration": calibration,
                "test": set(test_sources),
            }
            score = 0.0
            for role in ("detector_val", "calibration", "train"):
                totals = role_totals(roles[role], stats)
                for feature, weight in (("images", 1.0), ("persons", 0.5), ("unsafe_evidence", 1.5)):
                    target = all_totals[feature] * targets[role]
                    score += weight * ((totals[feature] - target) / max(target, 1.0)) ** 2
            if role_totals(calibration, stats)["unsafe_evidence"] == 0:
                continue
            tie = (
                tuple(sorted(detector)),
                tuple(sorted(calibration)),
                tuple(sorted(train)),
            )
            candidate = (score, tie)
            if best is None or candidate < best[0]:
                best = (candidate, roles)
    if best is None:
        raise RuntimeError("could not construct inner source roles")
    return best[1]


def stratified_training_sample(records: list[dict], rate: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)
    target = max(len(grouped), int(round(len(records) * rate / 100.0)))
    rng = random.Random(seed)
    quotas = {}
    fractions = []
    for source in sorted(grouped):
        ideal = target * len(grouped[source]) / len(records)
        base = max(1, int(math.floor(ideal)))
        base = min(base, len(grouped[source]))
        quotas[source] = base
        fractions.append((ideal - math.floor(ideal), rng.random(), source))
    while sum(quotas.values()) > target:
        removable = [
            source for source in sorted(quotas) if quotas[source] > 1
        ]
        if not removable:
            raise RuntimeError("training target is smaller than the number of source groups")
        source = max(removable, key=lambda item: (quotas[item], len(grouped[item]), item))
        quotas[source] -= 1
    for _, _, source in sorted(fractions, reverse=True):
        if sum(quotas.values()) >= target:
            break
        if quotas[source] < len(grouped[source]):
            quotas[source] += 1
    while sum(quotas.values()) < target:
        candidates = [source for source in sorted(grouped) if quotas[source] < len(grouped[source])]
        if not candidates:
            break
        source = max(candidates, key=lambda item: len(grouped[item]) - quotas[item])
        quotas[source] += 1
    selected = []
    for source in sorted(grouped):
        rows = sorted(grouped[source], key=lambda row: row["name"])
        selected.extend(rng.sample(rows, quotas[source]))
    return sorted(selected, key=lambda row: (row["source"], row["name"]))


def unique_records(
    records: list[dict], hashes: dict[Path, str], blocked_hashes: set[str] | None = None
) -> tuple[list[dict], list[dict]]:
    """Keep one image per exact-content hash and honor higher-priority roles."""
    blocked = set(blocked_hashes or set())
    kept = []
    excluded = []
    for record in sorted(records, key=lambda row: (row["source"], row["name"])):
        digest = hashes[record["image"]]
        if digest in blocked:
            excluded.append(record)
            continue
        blocked.add(digest)
        kept.append(record)
    return kept, excluded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def write_yaml(cell_root: Path) -> None:
    lines = [
        f"path: {cell_root.as_posix()}",
        "train: images/train",
        "val: images/detector_val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in CLASS_NAMES.items())
    (cell_root / "detector.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_disjoint(role_rows: dict[str, list[dict]], hashes: dict[Path, str]) -> dict:
    roles = sorted(role_rows)
    path_sets = {role: {row["image"].resolve() for row in rows} for role, rows in role_rows.items()}
    source_sets = {role: {row["source"] for row in rows} for role, rows in role_rows.items()}
    hash_sets = {role: {hashes[row["image"]] for row in rows} for role, rows in role_rows.items()}
    audit = {"path_overlap": {}, "source_overlap": {}, "hash_overlap": {}}
    for left, right in itertools.combinations(roles, 2):
        key = f"{left}__{right}"
        audit["path_overlap"][key] = len(path_sets[left] & path_sets[right])
        audit["source_overlap"][key] = len(source_sets[left] & source_sets[right])
        audit["hash_overlap"][key] = len(hash_sets[left] & hash_sets[right])
    if any(value for section in audit.values() for value in section.values()):
        raise RuntimeError(f"split leakage detected: {audit}")
    return audit


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--no-hash-audit", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing protocol: {args.out}")
    if any(rate <= 0 or rate >= 100 for rate in args.rates):
        parser.error("rates must be between 0 and 100")

    records = collect_records(args.dataset_root)
    stats = source_statistics(records)
    outer = choose_outer_folds(stats)
    roles_by_fold = {
        fold: choose_inner_roles(
            {source for source, assigned_fold in outer.items() if assigned_fold == fold},
            stats,
        )
        for fold in range(3)
    }
    hashes = {
        record["image"]: (record["image"].name if args.no_hash_audit else sha256(record["image"]))
        for record in records
    }

    args.out.mkdir(parents=True)
    source_rows = []
    for source in sorted(stats):
        source_rows.append({"source": source, "outer_test_fold": outer[source], **stats[source]})
    write_csv(args.out / "source_statistics.csv", source_rows)

    all_manifest_rows = []
    audit_rows = []
    materialization_modes = Counter()
    for rate in args.rates:
        for split_seed in args.seeds:
            for fold in range(3):
                roles = roles_by_fold[fold]
                raw_roles = {
                    "detector_val": [row for row in records if row["source"] in roles["detector_val"]],
                    "calibration": [row for row in records if row["source"] in roles["calibration"]],
                    "test": [row for row in records if row["source"] in roles["test"]],
                }
                excluded_duplicates = []
                blocked_hashes: set[str] = set()
                heldout_rows = {}
                for role in ("test", "calibration", "detector_val"):
                    heldout_rows[role], excluded = unique_records(
                        raw_roles[role], hashes, blocked_hashes
                    )
                    blocked_hashes.update(hashes[row["image"]] for row in heldout_rows[role])
                    excluded_duplicates.extend(
                        {"role": role, "image": row["name"], "source": row["source"]}
                        for row in excluded
                    )
                raw_train_pool = [row for row in records if row["source"] in roles["train"]]
                train_pool, excluded = unique_records(raw_train_pool, hashes, blocked_hashes)
                excluded_duplicates.extend(
                    {"role": "train", "image": row["name"], "source": row["source"]}
                    for row in excluded
                )
                sampled_train = stratified_training_sample(
                    train_pool,
                    rate,
                    20260805 + rate * 10000 + split_seed * 100 + fold,
                )
                role_rows = {"train": sampled_train, **heldout_rows}
                audit = assert_disjoint(role_rows, hashes)
                cell_root = args.out / f"rate_{rate}pct" / f"seed{split_seed}" / f"fold{fold}"
                cell_root.mkdir(parents=True)
                manifest_rows = []
                for role, rows in role_rows.items():
                    for row in rows:
                        image_target = cell_root / "images" / role / row["image"].name
                        label_target = cell_root / "labels" / role / row["label"].name
                        materialization_modes[link_or_copy(row["image"], image_target)] += 1
                        materialization_modes[link_or_copy(row["label"], label_target)] += 1
                        item = {
                            "rate": rate,
                            "seed": split_seed,
                            "fold": fold,
                            "role": role,
                            "source": row["source"],
                            "image_name": row["image"].name,
                            "image_sha256": hashes[row["image"]],
                            "original_image": str(row["image"]),
                            "original_label": str(row["label"]),
                            **{f"class_{index}": row["counts"][index] for index in CLASS_NAMES},
                        }
                        manifest_rows.append(item)
                        all_manifest_rows.append(item)
                write_csv(cell_root / "manifest.csv", manifest_rows)
                write_yaml(cell_root)
                role_summary = {
                    role: {
                        "sources": sorted({row["source"] for row in role_rows[role]}),
                        "source_count": len({row["source"] for row in role_rows[role]}),
                        "images": len(role_rows[role]),
                        "persons": sum(row["counts"][3] for row in role_rows[role]),
                        "unsafe_evidence": sum(
                            row["counts"][1] + row["counts"][2] for row in role_rows[role]
                        ),
                    }
                    for role in role_rows
                }
                protocol = {
                    "protocol": "source_disjoint_four_way_v1",
                    "dataset_root": str(args.dataset_root.resolve()),
                    "training_label_rate_percent": rate,
                    "training_pool_images": len(train_pool),
                    "sampled_training_images": len(sampled_train),
                    "sampling_seed": 20260805 + rate * 10000 + split_seed * 100 + fold,
                    "class_names": CLASS_NAMES,
                    "roles": role_summary,
                    "leakage_audit": audit,
                    "excluded_exact_duplicate_images": excluded_duplicates,
                }
                (cell_root / "protocol.json").write_text(
                    json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                audit_rows.append(
                    {
                        "rate": rate,
                        "seed": split_seed,
                        "fold": fold,
                        **{f"{role}_images": values["images"] for role, values in role_summary.items()},
                        **{f"{role}_sources": values["source_count"] for role, values in role_summary.items()},
                        "all_path_overlaps": sum(audit["path_overlap"].values()),
                        "all_source_overlaps": sum(audit["source_overlap"].values()),
                        "all_hash_overlaps": sum(audit["hash_overlap"].values()),
                    }
                )
                print(
                    f"[built] r{rate}_s{split_seed}_f{fold} "
                    + " ".join(f"{role}={len(rows)}" for role, rows in role_rows.items()),
                    flush=True,
                )

    write_csv(args.out / "all_manifest.csv", all_manifest_rows)
    write_csv(args.out / "split_audit.csv", audit_rows)
    summary = {
        "protocol": "source_disjoint_four_way_v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_images": len(records),
        "source_groups": len(stats),
        "rates": args.rates,
        "seeds": args.seeds,
        "folds": [0, 1, 2],
        "outer_test_folds": {
            str(fold): sorted(source for source, assigned in outer.items() if assigned == fold)
            for fold in range(3)
        },
        "fold_roles": {
            str(fold): {role: sorted(sources) for role, sources in roles.items()}
            for fold, roles in roles_by_fold.items()
        },
        "materialization_modes": dict(materialization_modes),
        "hash_audit": not args.no_hash_audit,
        "cells": len(audit_rows),
        "all_overlap_counts_zero": all(
            row["all_path_overlaps"] == 0
            and row["all_source_overlaps"] == 0
            and row["all_hash_overlaps"] == 0
            for row in audit_rows
        ),
    }
    (args.out / "protocol_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[complete] {args.out}", flush=True)


if __name__ == "__main__":
    main()
