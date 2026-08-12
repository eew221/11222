"""Prioritize rule-disagreement cases for blinded visual adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from analyze_cached_rc_wssi_robustness import assign
from evaluate_person_conditioned_gate import imread, read_yolo_boxes
import formal_worker_state_experiment as formal


MODES = ("proposed_lexicographic", "max_iou", "center_inside", "hungarian_iou")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assignment_map(persons, evidence, mode: str) -> dict[int, str]:
    grouped = assign(persons, evidence, mode)
    return {
        id(item): f"P{person_index + 1}"
        for person_index, items in grouped.items()
        for item in items
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--agreement-controls", type=int, default=30)
    args = parser.parse_args()
    image_rows = read_csv(args.audit_root / "audit_image_manifest.csv")
    evidence_template = {
        (row["audit_id"], row["evidence_id"]): row
        for row in read_csv(args.audit_root / "blinded_evidence_assignment_template.csv")
    }
    matrix = []
    for image_row in image_rows:
        image_path = Path(image_row["original_image"])
        image = imread(image_path)
        height, width = image.shape[:2]
        label_path = image_path.parents[2] / "labels" / "test" / f"{image_path.stem}.txt"
        boxes = read_yolo_boxes(label_path, width, height)
        persons = sorted(
            [box for box in boxes if box["cls"] == formal.PERSON_CLASS],
            key=lambda box: (box["xyxy"][0], box["xyxy"][1]),
        )
        evidence = sorted(
            [box for box in boxes if box["cls"] in formal.SAFETY_CLASSES],
            key=lambda box: (box["cls"], box["xyxy"][1], box["xyxy"][0]),
        )
        mode_maps = {mode: assignment_map(persons, evidence, mode) for mode in MODES}
        for evidence_index, item in enumerate(evidence, 1):
            evidence_id = f"E{evidence_index}"
            values = {mode: mode_maps[mode].get(id(item), "NONE") for mode in MODES}
            unique = sorted(set(values.values()))
            template = evidence_template[(image_row["audit_id"], evidence_id)]
            matrix.append(
                {
                    "audit_id": image_row["audit_id"],
                    "source_group": image_row["source_group"],
                    "image_name": image_row["image_name"],
                    "evidence_id": evidence_id,
                    "evidence_class": template["evidence_class"],
                    "candidate_person_ids": template["candidate_person_ids"],
                    **values,
                    "rule_agreement": int(len(unique) == 1),
                    "unique_rule_outcomes": "|".join(unique),
                }
            )

    disagreements = [row for row in matrix if not row["rule_agreement"]]
    controls = [row for row in matrix if row["rule_agreement"]]
    rng = random.Random(20260805)
    rng.shuffle(controls)
    selected_controls = controls[: args.agreement_controls]
    priority_keys = {
        (row["audit_id"], row["evidence_id"]): "rule_disagreement"
        for row in disagreements
    }
    priority_keys.update(
        {
            (row["audit_id"], row["evidence_id"]): "agreement_control"
            for row in selected_controls
        }
    )
    blinded = []
    for key, reason in sorted(priority_keys.items()):
        template = evidence_template[key]
        blinded.append(
            {
                "audit_id": key[0],
                "evidence_id": key[1],
                "evidence_class": template["evidence_class"],
                "candidate_person_ids": template["candidate_person_ids"],
                "audit_stratum": reason,
                "assigned_person_id": "",
                "assignment_confidence": "",
                "occluded_or_ambiguous": "",
                "notes": "",
            }
        )
    write_csv(args.audit_root / "sealed_rule_assignment_matrix.csv", matrix)
    write_csv(args.audit_root / "blinded_priority_adjudication_template.csv", blinded)
    manifest = {
        "evidence_boxes": len(matrix),
        "rule_disagreements": len(disagreements),
        "rule_agreements": len(controls),
        "agreement_controls": len(selected_controls),
        "priority_rows": len(blinded),
        "modes": MODES,
        "annotation_status": "priority_template_created_not_yet_human_signed",
        "matrix_sha256": hashlib.sha256(
            (args.audit_root / "sealed_rule_assignment_matrix.csv").read_bytes()
        ).hexdigest(),
    }
    (args.audit_root / "adjudication_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
