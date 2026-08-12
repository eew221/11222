"""Merge two independent annotation passes and frozen adjudications."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ASSIGNMENT = re.compile(r"^(P[1-9][0-9]*|NONE|AMBIGUOUS)$")
SPECIAL_ASSIGNMENTS = {"NONE", "AMBIGUOUS"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    a_rows = read(args.annotator_a)
    b_rows = read(args.annotator_b)
    c_rows = read(args.adjudication)
    a = {(row["audit_id"], row["evidence_id"]): row for row in a_rows}
    b = {(row["audit_id"], row["evidence_id"]): row for row in b_rows}
    c = {(row["audit_id"], row["evidence_id"]): row for row in c_rows}
    if set(a) != set(b):
        raise ValueError("A and B contain different evidence keys")

    disagreements = {key for key in a if a[key]["assigned_person_id"] != b[key]["assigned_person_id"]}
    if set(c) != disagreements:
        raise ValueError(f"adjudication keys do not exactly match disagreements: expected={len(disagreements)} got={len(c)}")
    unresolved = [key for key, row in c.items() if not row["adjudicated_assignment"]]
    if unresolved:
        raise ValueError(f"unresolved adjudications: {unresolved[:5]}")
    for key, row in c.items():
        candidates = set(filter(None, a[key]["candidate_person_ids"].split("|")))
        if not ASSIGNMENT.fullmatch(row["adjudicated_assignment"]) or row["adjudicated_assignment"] not in candidates | SPECIAL_ASSIGNMENTS:
            raise ValueError(f"invalid adjudicated label at {key}")

    fields = [
        "audit_id", "evidence_id", "evidence_class", "candidate_person_ids",
        "annotator_a", "annotator_b", "final_assignment", "final_label_source",
        "adjudication_confidence", "adjudication_occluded_or_ambiguous", "adjudication_rationale",
    ]
    final_rows = []
    for key in sorted(a, key=lambda item: (int(item[0][1:]), item[1])):
        left, right = a[key], b[key]
        if key in c:
            adjudication = c[key]
            final_assignment = adjudication["adjudicated_assignment"]
            source = "third_expert_adjudication"
            confidence = adjudication["decision_confidence"]
            occlusion = adjudication["occluded_or_ambiguous"]
            rationale = adjudication["rationale"]
        else:
            final_assignment = left["assigned_person_id"]
            source = "independent_exact_agreement"
            confidence = ""
            occlusion = ""
            rationale = ""
        final_rows.append({
            "audit_id": key[0],
            "evidence_id": key[1],
            "evidence_class": left["evidence_class"],
            "candidate_person_ids": left["candidate_person_ids"],
            "annotator_a": left["assigned_person_id"],
            "annotator_b": right["assigned_person_id"],
            "final_assignment": final_assignment,
            "final_label_source": source,
            "adjudication_confidence": confidence,
            "adjudication_occluded_or_ambiguous": occlusion,
            "adjudication_rationale": rationale,
        })

    args.out.mkdir(parents=True)
    final_path = args.out / "final_adjudicated_assignment.csv"
    write(final_path, fields, final_rows)
    source_counts = Counter(row["final_label_source"] for row in final_rows)
    label_counts = Counter(row["final_assignment"] for row in final_rows)
    summary = {
        "status": "final_adjudicated_human_reference_ready",
        "evidence_boxes": len(final_rows),
        "initial_exact_agreement": len(final_rows) - len(disagreements),
        "initial_disagreements": len(disagreements),
        "adjudicated_disagreements": len(c),
        "final_label_counts": dict(sorted(label_counts.items())),
        "final_label_source_counts": dict(sorted(source_counts.items())),
        "source_csv_sha256": {
            "annotator_a": sha256(args.annotator_a),
            "annotator_b": sha256(args.annotator_b),
            "adjudication": sha256(args.adjudication),
            "final": sha256(final_path),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation_note": "Final labels are an adjudicated human association reference; they do not establish deployment safety or cluster-valid risk guarantees.",
    }
    (args.out / "final_adjudicated_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
