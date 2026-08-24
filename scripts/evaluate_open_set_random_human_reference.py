"""Score allocation rules against the frozen open-set random human audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import formal_worker_state_experiment as formal
from analyze_cached_rc_wssi_robustness import assign
from evaluate_person_conditioned_gate import imread, read_yolo_boxes


METHODS = (
    "proposed_lexicographic", "center_inside", "hungarian_iou", "max_iou", "global_pooling",
)
SPECIAL = {"NONE", "AMBIGUOUS"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owner_sets(persons: list[dict], evidence: list[dict], method: str) -> dict[int, set[str]]:
    output = {id(item): set() for item in evidence}
    for person_index, items in assign(persons, evidence, method).items():
        for item in items:
            output[id(item)].add(f"P{person_index + 1}")
    return output


def truth(label: str) -> set[str] | None:
    label = label.upper()
    if label == "AMBIGUOUS":
        return None
    if label == "NONE":
        return set()
    if label.startswith("P") and label[1:].isdigit():
        return {label}
    raise ValueError(f"invalid final human label: {label!r}")


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[round(q * (len(values) - 1))] if values else math.nan


def bootstrap(rows: list[dict], method: str, draws: int, seed: int) -> tuple[float, float, float]:
    data = [row for row in rows if row["method"] == method and row["human_assignment"] != "AMBIGUOUS"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        grouped[row["source_group"]].append(row)
    groups = sorted(grouped)
    estimate = statistics.mean(int(row["exact_owner_set"]) for row in data)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        values = [int(row["exact_owner_set"]) for group in (rng.choice(groups) for _ in groups) for row in grouped[group]]
        samples.append(statistics.mean(values))
    return estimate, quantile(samples, 0.025), quantile(samples, 0.975)


def paired(rows: list[dict], other: str, draws: int, seed: int) -> tuple[float, float, float]:
    left = {(r["audit_id"], r["evidence_id"]): r for r in rows if r["method"] == "proposed_lexicographic" and r["human_assignment"] != "AMBIGUOUS"}
    right = {(r["audit_id"], r["evidence_id"]): r for r in rows if r["method"] == other and r["human_assignment"] != "AMBIGUOUS"}
    if set(left) != set(right):
        raise ValueError("paired methods use different rows")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in left:
        grouped[left[key]["source_group"]].append(int(left[key]["exact_owner_set"]) - int(right[key]["exact_owner_set"]))
    groups = sorted(grouped)
    estimate = statistics.mean(value for values in grouped.values() for value in values)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        values = [value for group in (rng.choice(groups) for _ in groups) for value in grouped[group]]
        samples.append(statistics.mean(values))
    return estimate, quantile(samples, 0.025), quantile(samples, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--human-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    manifest = {row["audit_id"]: row for row in read_csv(args.audit_root / "audit_image_manifest.csv")}
    human_rows = read_csv(args.human_reference)
    human = {(row["audit_id"], row["evidence_id"]): row for row in human_rows}
    if len(human) != len(human_rows):
        raise ValueError("duplicate human-reference key")

    scored: list[dict] = []
    for audit_id in sorted(manifest, key=lambda value: int(value[1:])):
        item = manifest[audit_id]
        image_path = Path(item["original_image"])
        label_path = image_path.parents[2] / "labels" / "test" / f"{image_path.stem}.txt"
        image = imread(image_path)
        if image is None or not label_path.is_file():
            raise FileNotFoundError(f"missing local image or annotation: {image_path}")
        height, width = image.shape[:2]
        boxes = read_yolo_boxes(label_path, width, height)
        persons = sorted((box for box in boxes if box["cls"] == formal.PERSON_CLASS), key=lambda box: (box["xyxy"][0], box["xyxy"][1]))
        evidence = sorted((box for box in boxes if box["cls"] in formal.SAFETY_CLASSES), key=lambda box: (box["cls"], box["xyxy"][1], box["xyxy"][0]))
        if len(persons) != int(item["person_count"]) or len(evidence) != int(item["evidence_count"]):
            raise ValueError(f"render/label box count mismatch for {audit_id}")
        predictions = {method: owner_sets(persons, evidence, method) for method in METHODS}
        for position, evidence_box in enumerate(evidence, 1):
            key = (audit_id, f"E{position}")
            if key not in human:
                raise ValueError(f"missing human reference for {key}")
            label = human[key]["final_assignment"].upper()
            expected = truth(label)
            for method in METHODS:
                predicted = predictions[method][id(evidence_box)]
                exact = "" if expected is None else int(predicted == expected)
                scored.append({
                    "audit_id": audit_id, "source_group": item["source_group"], "fold": item["fold"],
                    "image_name": item["image_name"], "evidence_id": f"E{position}",
                    "evidence_class": human[key]["evidence_class"], "human_assignment": label,
                    "human_label_source": human[key]["final_label_source"], "method": method,
                    "predicted_owner_set": "|".join(sorted(predicted)) or "NONE", "exact_owner_set": exact,
                    "owner_link_tp": "" if expected is None else len(predicted & expected),
                    "owner_link_fp": "" if expected is None else len(predicted - expected),
                    "owner_link_fn": "" if expected is None else len(expected - predicted),
                })
    if {key for key in human} != {(row["audit_id"], row["evidence_id"]) for row in scored if row["method"] == METHODS[0]}:
        raise ValueError("human-reference keys do not exactly match audit rows")

    summary, paired_rows = [], []
    for method in METHODS:
        rows = [row for row in scored if row["method"] == method and row["human_assignment"] != "AMBIGUOUS"]
        exact = sum(int(row["exact_owner_set"]) for row in rows)
        tp, fp, fn = (sum(int(row[field]) for row in rows) for field in ("owner_link_tp", "owner_link_fp", "owner_link_fn"))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        image_scores = defaultdict(list); group_scores = defaultdict(list)
        for row in rows: image_scores[row["audit_id"]].append(int(row["exact_owner_set"])); group_scores[row["source_group"]].append(int(row["exact_owner_set"]))
        estimate, low, high = bootstrap(scored, method, args.bootstrap_draws, int(hashlib.sha256(method.encode()).hexdigest()[:8], 16))
        summary.append({"method": method, "evidence_boxes": len(rows), "exact_correct": exact, "box_exact_accuracy": exact / len(rows), "filename_group_bootstrap_low": low, "filename_group_bootstrap_high": high, "image_macro_accuracy": statistics.mean(statistics.mean(v) for v in image_scores.values()), "filename_group_macro_accuracy": statistics.mean(statistics.mean(v) for v in group_scores.values()), "owner_link_precision": precision, "owner_link_recall": recall, "owner_link_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "extra_owner_links": fp, "missed_owner_links": fn, "ambiguous_human_rows_excluded": sum(row["human_assignment"] == "AMBIGUOUS" for row in scored if row["method"] == method), "interval_note": "descriptive filename-group resampling interval; filename groups are not verified independent clusters"})
    for method in METHODS[1:]:
        estimate, low, high = paired(scored, method, args.bootstrap_draws, int(hashlib.sha256(("paired:" + method).encode()).hexdigest()[:8], 16))
        paired_rows.append({"comparison": f"RC-WSSI minus {method}", "exact_accuracy_difference": estimate, "filename_group_bootstrap_low": low, "filename_group_bootstrap_high": high, "interval_note": "descriptive paired filename-group resampling interval; not cluster-valid inference"})

    args.out.mkdir(parents=True)
    write_csv(args.out / "open_set_random_human_reference_rows.csv", scored)
    write_csv(args.out / "open_set_random_human_reference_summary.csv", summary)
    write_csv(args.out / "open_set_random_human_reference_paired_differences.csv", paired_rows)
    report = {"status": "complete_open_set_random_human_reference_evaluation", "images": len(manifest), "filename_groups": len({row["source_group"] for row in manifest.values()}), "human_reference_rows": len(human), "ambiguous_rows_excluded_from_exact_scoring": sum(row["final_assignment"] == "AMBIGUOUS" for row in human_rows), "methods": list(METHODS), "bootstrap_draws": args.bootstrap_draws, "input_sha256": {"human_reference": sha256(args.human_reference)}, "scope": "open-candidate random sampled PPE-owner evaluation only; not full-corpus worker-state truth, cluster-valid inference, or deployment validation"}
    (args.out / "evaluation_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "summary": summary, "paired": paired_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
