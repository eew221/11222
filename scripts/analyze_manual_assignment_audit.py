"""Analyze two blinded PPE-to-worker annotation passes after human completion.

This tool intentionally refuses to infer labels. It reports agreement only from
two completed independent CSV files created by build_manual_assignment_audit.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


REQUIRED = {"audit_id", "evidence_id", "assigned_person_id"}
VALID_ASSIGNMENT = re.compile(r"^(P[1-9][0-9]*|NONE|AMBIGUOUS)$")


def read_rows(path: Path) -> dict[tuple[str, str], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(REQUIRED)}")
        rows: dict[tuple[str, str], str] = {}
        for line, row in enumerate(reader, start=2):
            key = (row["audit_id"].strip(), row["evidence_id"].strip())
            value = row["assigned_person_id"].strip().upper()
            if not key[0] or not key[1]:
                raise ValueError(f"{path}:{line}: blank audit_id or evidence_id")
            if not VALID_ASSIGNMENT.fullmatch(value):
                raise ValueError(
                    f"{path}:{line}: assigned_person_id must be Pn, NONE, or AMBIGUOUS"
                )
            if key in rows:
                raise ValueError(f"{path}:{line}: duplicate evidence row {key}")
            rows[key] = value
    if not rows:
        raise ValueError(f"{path}: no annotation rows")
    return rows


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(
        left_counts[label] * right_counts[label] for label in set(left_counts) | set(right_counts)
    ) / (len(left) * len(left))
    if abs(1.0 - expected) < 1e-12:
        return None
    return (observed - expected) / (1.0 - expected)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    first = read_rows(args.annotator_a)
    second = read_rows(args.annotator_b)
    if set(first) != set(second):
        missing_first = sorted(set(second) - set(first))
        missing_second = sorted(set(first) - set(second))
        raise ValueError(
            "annotation passes have different evidence rows: "
            f"missing_from_a={missing_first[:5]}, missing_from_b={missing_second[:5]}"
        )

    rows = []
    values_a, values_b = [], []
    for audit_id, evidence_id in sorted(first):
        value_a, value_b = first[(audit_id, evidence_id)], second[(audit_id, evidence_id)]
        values_a.append(value_a)
        values_b.append(value_b)
        rows.append(
            {
                "audit_id": audit_id,
                "evidence_id": evidence_id,
                "annotator_a": value_a,
                "annotator_b": value_b,
                "exact_agreement": int(value_a == value_b),
                "requires_adjudication": int(value_a != value_b),
            }
        )

    args.out.mkdir(parents=True)
    write_csv(args.out / "pairwise_assignment_rows.csv", rows)
    total = len(rows)
    exact = sum(row["exact_agreement"] for row in rows)
    categories = sorted(set(values_a) | set(values_b))
    summary = {
        "annotation_status": "two_independent_passes_compared_not_yet_adjudicated",
        "evidence_boxes": total,
        "exact_agreement_count": exact,
        "exact_agreement_rate": exact / total,
        "requires_adjudication_count": total - exact,
        "cohen_kappa": cohen_kappa(values_a, values_b),
        "categories_observed": categories,
        "note": (
            "This is annotator agreement, not RC-WSSI semantic validity. "
            "Resolve disagreements independently before comparing adjudicated labels "
            "with the sealed proposed assignment reference."
        ),
    }
    (args.out / "pairwise_assignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
