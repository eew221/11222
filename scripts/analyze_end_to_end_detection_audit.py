"""Analyze three frozen blind passes from a detector-output PPE-owner audit.

This script intentionally separates semantic owner agreement on predicted PPE
from detector failures: unmatched predicted PPE, PPE missed by the detector,
and real owners outside the detected-person set.  It does not construct a
deployment-risk interval or a worker-state safety guarantee.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SPECIAL = {"NONE", "AMBIGUOUS", "OUTSIDE_DETECTED_PERSON_SET", "FALSE_DETECTION"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty result: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def frozen_rows(root: Path, annotator: str) -> dict[tuple[str, str], dict[str, str]]:
    path = root / f"annotator_{annotator}" / "evidence_assignment.csv"
    lock = path.parent / "ANNOTATION_FINALIZED.json"
    if not lock.is_file():
        raise ValueError(f"annotator {annotator} has not frozen a pass")
    expected = json.loads(lock.read_text(encoding="utf-8")).get("csv_sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"annotator {annotator} frozen CSV hash mismatch")
    rows = read_csv(path)
    output = {(row["audit_id"], row["evidence_id"]): row for row in rows}
    if len(rows) != len(output) or any(not row["assigned_person_id"] for row in rows):
        raise ValueError(f"annotator {annotator} has duplicate or incomplete rows")
    return output


def majority(values: list[str]) -> tuple[str, str]:
    counts = Counter(values)
    label, count = counts.most_common(1)[0]
    return (label, "blind_majority") if count >= 2 else ("AMBIGUOUS", "three_way_disagreement_retained_ambiguous")


def count_partition(values: list[dict[str, str]]) -> dict[str, int]:
    """Return mutually exclusive detector-output audit counts for one scope."""
    predicted = [row for row in values if row["row_type"] == "predicted_ppe"]
    missed = [row for row in values if row["row_type"] == "reference_ppe_missed_by_detector"]
    predicted_in_set = [row for row in predicted if row["final_human_label"].startswith("P")]
    predicted_outside = [row for row in predicted if row["final_human_label"] == "OUTSIDE_DETECTED_PERSON_SET"]
    predicted_false = [row for row in predicted if row["final_human_label"] == "FALSE_DETECTION"]
    predicted_ambiguous = [row for row in predicted if row["final_human_label"] == "AMBIGUOUS"]
    predicted_none = [row for row in predicted if row["final_human_label"] == "NONE"]
    matched_predicted = [row for row in predicted if row["detector_status"] == "matched_reference"]
    unmatched_predicted = [row for row in predicted if row["detector_status"] == "unmatched_predicted_ppe"]
    determinate = predicted_in_set + predicted_outside
    rc_exact = [row for row in determinate if int(row["rc_exact_on_determinate_real_owner"])]
    in_set_mismatch = [row for row in predicted_in_set if not int(row["rc_exact_on_determinate_real_owner"])]
    missed_in_set = [row for row in missed if row["final_human_label"].startswith("P")]
    missed_outside = [row for row in missed if row["final_human_label"] == "OUTSIDE_DETECTED_PERSON_SET"]
    missed_ambiguous = [row for row in missed if row["final_human_label"] == "AMBIGUOUS"]
    missed_none = [row for row in missed if row["final_human_label"] == "NONE"]
    return {
        "audit_rows": len(values),
        "predicted_ppe": len(predicted),
        "matched_predicted_ppe": len(matched_predicted),
        "unmatched_predicted_ppe": len(unmatched_predicted),
        "matched_predicted_human_false_detection": sum(row["final_human_label"] == "FALSE_DETECTION" for row in matched_predicted),
        "matched_predicted_not_human_false_detection": sum(row["final_human_label"] != "FALSE_DETECTION" for row in matched_predicted),
        "unmatched_predicted_human_false_detection": sum(row["final_human_label"] == "FALSE_DETECTION" for row in unmatched_predicted),
        "unmatched_predicted_not_human_false_detection": sum(row["final_human_label"] != "FALSE_DETECTION" for row in unmatched_predicted),
        "predicted_owner_in_detected_person_set": len(predicted_in_set),
        "predicted_owner_outside_detected_person_set": len(predicted_outside),
        "human_false_detection": len(predicted_false),
        "predicted_ambiguous": len(predicted_ambiguous),
        "predicted_none": len(predicted_none),
        "determinate_owner_predicted_ppe": len(determinate),
        "rc_exact_owner": len(rc_exact),
        "rc_owner_mismatch": len(determinate) - len(rc_exact),
        "rc_in_set_owner_mismatch": len(in_set_mismatch),
        "missed_reference_ppe": len(missed),
        "missed_owner_in_detected_person_set": len(missed_in_set),
        "missed_owner_outside_detected_person_set": len(missed_outside),
        "missed_ambiguous": len(missed_ambiguous),
        "missed_none": len(missed_none),
        "all_owner_outside_detected_person_set": len(predicted_outside) + len(missed_outside),
        "all_ambiguous": len(predicted_ambiguous) + len(missed_ambiguous),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--emit-row-level",
        action="store_true",
        help="Write per-row annotator labels. Use only in an authorized restricted location.",
    )
    args = parser.parse_args()
    root, out = args.audit_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing results: {out}")
    passes = {name: frozen_rows(root, name) for name in "ABC"}
    keys = set(passes["A"])
    if any(set(rows) != keys for rows in passes.values()):
        raise ValueError("three frozen passes use different evidence rows")
    sealed = {(row["audit_id"], row["evidence_id"]): row for row in read_csv(root / "sealed_detector_audit_reference.csv")}
    if set(sealed) != keys:
        raise ValueError("sealed reference and blind annotation rows do not match")
    images = {row["audit_id"]: row for row in read_csv(root / "audit_image_manifest.csv")}
    rows = []
    pair_agreement = {"A_B": 0, "A_C": 0, "B_C": 0}
    for key in sorted(keys, key=lambda value: (int(value[0][1:]), value[1][0], int(value[1][1:]))):
        labels = {name: passes[name][key]["assigned_person_id"].strip().upper() for name in "ABC"}
        if any(label not in SPECIAL and not label.startswith("P") for label in labels.values()):
            raise ValueError(f"invalid label at {key}: {labels}")
        final, source = majority(list(labels.values()))
        pair_agreement["A_B"] += labels["A"] == labels["B"]
        pair_agreement["A_C"] += labels["A"] == labels["C"]
        pair_agreement["B_C"] += labels["B"] == labels["C"]
        reference = sealed[key]
        rc_owner = reference["rc_wssi_owner"]
        row_type = reference["row_type"]
        determinate_real_owner = final.startswith("P") or final == "OUTSIDE_DETECTED_PERSON_SET"
        rc_exact = int(determinate_real_owner and final == rc_owner)
        rows.append({
            "audit_id": key[0], "evidence_id": key[1], "source_group": images[key[0]]["source_group"],
            "row_type": row_type, "detector_status": reference["detector_status"],
            "annotator_a": labels["A"], "annotator_b": labels["B"], "annotator_c": labels["C"],
            "final_human_label": final, "final_label_source": source,
            "all_three_agree": int(len(set(labels.values())) == 1),
            "rc_wssi_owner": rc_owner, "rc_exact_on_determinate_real_owner": rc_exact,
            "global_pool_owner_set": reference["global_pool_owner_set"],
        })
    totals = count_partition(rows)
    summary = {
        "status": "complete_frozen_three_pass_detector_output_audit",
        "rows": totals["audit_rows"], "predicted_ppe_rows": totals["predicted_ppe"], "missed_reference_ppe_rows": totals["missed_reference_ppe"],
        "all_three_exact_agreement": f"{sum(int(row['all_three_agree']) for row in rows)}/{len(rows)}",
        "pairwise_exact_agreement": {name: value / len(rows) for name, value in pair_agreement.items()},
        **totals,
        "rc_exact_owner_on_determinate_real_owner_predicted_ppe": f"{totals['rc_exact_owner']}/{totals['determinate_owner_predicted_ppe']}" if totals["determinate_owner_predicted_ppe"] else "NA",
        "scope": "sampled detector-output error decomposition and ownership audit; not full-corpus worker-state truth, cluster-valid inference, calibrated composite risk, or deployment validation",
    }
    per_source = []
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_source[row["source_group"]].append(row)
    for source, values in sorted(by_source.items()):
        counts = count_partition(values)
        counts["source_group"] = source
        counts["rc_exact_rate_on_determinate_owner"] = (
            counts["rc_exact_owner"] / counts["determinate_owner_predicted_ppe"]
            if counts["determinate_owner_predicted_ppe"] else "NA"
        )
        per_source.append(counts)
    partition_rows = [
        {"scope": "predicted_PPE", "category": "all predicted PPE", "count": totals["predicted_ppe"]},
        {"scope": "predicted_PPE", "category": "owner in detected-person set", "count": totals["predicted_owner_in_detected_person_set"]},
        {"scope": "predicted_PPE", "category": "owner outside detected-person set", "count": totals["predicted_owner_outside_detected_person_set"]},
        {"scope": "predicted_PPE", "category": "human false detection", "count": totals["human_false_detection"]},
        {"scope": "predicted_PPE", "category": "ambiguous", "count": totals["predicted_ambiguous"]},
        {"scope": "conditional owner score", "category": "determinate owner", "count": totals["determinate_owner_predicted_ppe"]},
        {"scope": "conditional owner score", "category": "RC exact owner", "count": totals["rc_exact_owner"]},
        {"scope": "conditional owner score", "category": "RC mismatch: in-set wrong owner", "count": totals["rc_in_set_owner_mismatch"]},
        {"scope": "conditional owner score", "category": "RC mismatch: owner outside set", "count": totals["predicted_owner_outside_detected_person_set"]},
        {"scope": "missed reference PPE", "category": "all missed reference PPE", "count": totals["missed_reference_ppe"]},
        {"scope": "missed reference PPE", "category": "owner in detected-person set", "count": totals["missed_owner_in_detected_person_set"]},
        {"scope": "missed reference PPE", "category": "owner outside detected-person set", "count": totals["missed_owner_outside_detected_person_set"]},
        {"scope": "prediction-reference cross-tab", "category": "matched prediction and human false detection", "count": totals["matched_predicted_human_false_detection"]},
        {"scope": "prediction-reference cross-tab", "category": "matched prediction and not human false detection", "count": totals["matched_predicted_not_human_false_detection"]},
        {"scope": "prediction-reference cross-tab", "category": "unmatched prediction and human false detection", "count": totals["unmatched_predicted_human_false_detection"]},
        {"scope": "prediction-reference cross-tab", "category": "unmatched prediction and not human false detection", "count": totals["unmatched_predicted_not_human_false_detection"]},
    ]
    out.mkdir(parents=True)
    if args.emit_row_level:
        write_csv(out / "end_to_end_detector_output_rows.csv", rows)
    write_csv(out / "end_to_end_detector_output_by_source.csv", per_source)
    write_csv(out / "end_to_end_detector_output_partition.csv", partition_rows)
    (out / "end_to_end_detector_output_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
