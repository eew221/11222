"""Analyze blinded two-annotator PPE-association adjudication files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


MODES = ("proposed_lexicographic", "max_iou", "center_inside", "hungarian_iou")
SPECIAL = {"NONE", "AMBIGUOUS"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict) -> tuple[str, str]:
    return row["audit_id"].strip(), row["evidence_id"].strip()


def label(row: dict) -> str:
    value = row.get("assigned_person_id", "").strip().upper()
    if value in SPECIAL or (value.startswith("P") and value[1:].isdigit()):
        return value
    raise ValueError(f"{row_key(row)}: invalid assigned_person_id={value!r}")


def kappa_label(row: dict) -> str:
    value = label(row)
    if value in SPECIAL:
        return value
    candidates = [item.upper() for item in row.get("candidate_person_ids", "").split("|") if item]
    return f"candidate_{candidates.index(value) + 1}" if value in candidates else "outside_candidates"


def indexed(path: Path, expected: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    rows = read_csv(path)
    output = {row_key(row): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate audit/evidence key in {path}")
    if set(output) != expected:
        raise ValueError(f"key mismatch in {path}")
    for row in rows:
        label(row)
    return output


def cohen_kappa(left: list[str], right: list[str]) -> tuple[float, float]:
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(
        left_counts[item] / len(left) * right_counts[item] / len(right)
        for item in set(left_counts) | set(right_counts)
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return observed, kappa


def wilson(correct: int, total: int) -> tuple[float, float, float]:
    if not total:
        return math.nan, math.nan, math.nan
    estimate, z = correct / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total)) / denominator
    return estimate, max(0.0, center - radius), min(1.0, center + radius)


def score(rows: list[dict], mode: str) -> float:
    determinate = [row for row in rows if row["adjudicated"] != "AMBIGUOUS"]
    if not determinate:
        return math.nan
    return sum(row[mode] == row["adjudicated"] for row in determinate) / len(determinate)


def bootstrap(
    grouped: dict[str, list[dict]], mode: str, weights: dict[str, float], draws: int
) -> tuple[float, float, float]:
    estimate = sum(weights[name] * score(rows, mode) for name, rows in grouped.items())
    rng = random.Random(20260805 + MODES.index(mode))
    samples = []
    for _ in range(draws):
        value = 0.0
        for name, rows in grouped.items():
            sample_score = score([rng.choice(rows) for _ in rows], mode)
            if math.isnan(sample_score):
                value = math.nan
                break
            value += weights[name] * sample_score
        if not math.isnan(value):
            samples.append(value)
    samples.sort()
    if not samples:
        return estimate, math.nan, math.nan
    return estimate, samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    template_rows = read_csv(args.audit_root / "blinded_priority_adjudication_template.csv")
    template = {row_key(row): row for row in template_rows}
    expected = set(template)
    annotator_a = indexed(args.annotator_a, expected)
    annotator_b = indexed(args.annotator_b, expected)
    adjudicated = indexed(args.adjudicated, expected)
    matrix_path = args.audit_root / "sealed_rule_assignment_matrix.csv"
    matrix = {row_key(row): row for row in read_csv(matrix_path)}
    if set(matrix) < expected:
        raise ValueError("sealed rule matrix does not cover all priority rows")
    manifest = json.loads((args.audit_root / "adjudication_manifest.json").read_text(encoding="utf-8"))
    matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    if matrix_hash != manifest.get("matrix_sha256"):
        raise RuntimeError("sealed rule-assignment matrix hash mismatch")

    keys = sorted(expected)
    labels_a = [label(annotator_a[item]) for item in keys]
    labels_b = [label(annotator_b[item]) for item in keys]
    normalized_a = [kappa_label(annotator_a[item]) for item in keys]
    normalized_b = [kappa_label(annotator_b[item]) for item in keys]
    raw_agreement, raw_kappa = cohen_kappa(labels_a, labels_b)
    normalized_agreement, normalized_kappa = cohen_kappa(normalized_a, normalized_b)

    rows = []
    for item in keys:
        row = {
            "audit_id": item[0],
            "evidence_id": item[1],
            "audit_stratum": template[item]["audit_stratum"],
            "annotator_a": label(annotator_a[item]),
            "annotator_b": label(annotator_b[item]),
            "adjudicated": label(adjudicated[item]),
            "annotators_agree": int(label(annotator_a[item]) == label(annotator_b[item])),
        }
        row.update({mode: matrix[item][mode].strip().upper() for mode in MODES})
        rows.append(row)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["audit_stratum"]].append(row)
    population = {
        "rule_disagreement": int(manifest["rule_disagreements"]),
        "agreement_control": int(manifest["rule_agreements"]),
    }
    if set(grouped) != set(population):
        raise ValueError(f"unexpected adjudication strata: {sorted(grouped)}")
    weights = {name: count / sum(population.values()) for name, count in population.items()}

    by_stratum = []
    for stratum, items in sorted(grouped.items()):
        determinate = [item for item in items if item["adjudicated"] != "AMBIGUOUS"]
        for mode in MODES:
            correct = sum(item[mode] == item["adjudicated"] for item in determinate)
            estimate, low, high = wilson(correct, len(determinate))
            by_stratum.append({
                "audit_stratum": stratum,
                "assignment_mode": mode,
                "sample_rows": len(items),
                "determinate_rows": len(determinate),
                "ambiguous_rows": len(items) - len(determinate),
                "correct": correct,
                "accuracy": estimate,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
            })
    stratified = []
    for mode in MODES:
        estimate, low, high = bootstrap(grouped, mode, weights, args.bootstrap_draws)
        stratified.append({
            "assignment_mode": mode,
            "stratified_accuracy": estimate,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "population_rule_disagreements": population["rule_disagreement"],
            "population_rule_agreements": population["agreement_control"],
            "human_priority_rows": len(rows),
        })

    args.out.mkdir(parents=True)
    write_csv(args.out / "adjudicated_rows.csv", rows)
    write_csv(args.out / "rule_accuracy_by_stratum.csv", by_stratum)
    write_csv(args.out / "rule_accuracy_stratified.csv", stratified)
    summary = {
        "status": "complete_human_adjudication_analyzed",
        "priority_rows": len(rows),
        "raw_exact_agreement": raw_agreement,
        "cohen_kappa_raw_labels": raw_kappa,
        "candidate_normalized_agreement": normalized_agreement,
        "cohen_kappa_candidate_normalized": normalized_kappa,
        "annotator_disagreements": sum(a != b for a, b in zip(labels_a, labels_b)),
        "adjudicated_ambiguous_rows": sum(row["adjudicated"] == "AMBIGUOUS" for row in rows),
        "bootstrap_draws": args.bootstrap_draws,
        "sealed_matrix_sha256": matrix_hash,
    }
    (args.out / "analysis_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
