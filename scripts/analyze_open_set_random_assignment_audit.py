"""Analyze three frozen blind passes from an open-set random owner audit.

The final human reference uses a predeclared majority rule.  A three-way split
is retained as AMBIGUOUS, avoiding a post-hoc adjudicator who can be anchored
by the other experts' selections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = {
    "audit_id", "evidence_id", "candidate_person_ids", "all_visible_person_ids",
    "assigned_person_id",
}
SPECIAL = {"NONE", "AMBIGUOUS"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns")
        return list(reader)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    output = {(row["audit_id"], row["evidence_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{path}: duplicate audit/evidence keys")
    for key, row in output.items():
        value = row["assigned_person_id"].strip().upper()
        if not value:
            raise ValueError(f"{path}: annotation is incomplete at {key}")
        visible = set(filter(None, row["all_visible_person_ids"].split("|")))
        if value not in visible | SPECIAL:
            raise ValueError(f"{path}: invalid assignment {key}={value!r}")
    return output


def lock_record(path: Path, expected_hash: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing frozen annotation record: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("csv_sha256") != expected_hash:
        raise ValueError(f"frozen CSV hash mismatch: {path}")
    return record


def exact_rate(left: list[str], right: list[str]) -> float:
    return sum(a == b for a, b in zip(left, right)) / len(left) if left else math.nan


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    observed = exact_rate(left, right)
    a, b = Counter(left), Counter(right)
    expected = sum(a[key] * b[key] for key in set(a) | set(b)) / (len(left) ** 2)
    return None if abs(1 - expected) < 1e-12 else (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.audit_root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")

    paths = {name: root / f"annotator_{name}" / "evidence_assignment.csv" for name in "ABC"}
    data = {name: index(path) for name, path in paths.items()}
    keys = set(data["A"])
    if any(set(rows) != keys for rows in data.values()):
        raise ValueError("the three blind passes do not contain identical evidence rows")
    locks = {name: lock_record(paths[name].parent / "ANNOTATION_FINALIZED.json", sha256(paths[name])) for name in "ABC"}

    with (root / "audit_image_manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        images = {row["audit_id"]: row for row in csv.DictReader(stream)}

    rows: list[dict] = []
    pairs = {"A_B": ([], []), "A_C": ([], []), "B_C": ([], [])}
    for key in sorted(keys, key=lambda item: (int(item[0][1:]), int(item[1][1:]))):
        annotations = {name: data[name][key]["assigned_person_id"].strip().upper() for name in "ABC"}
        for pair, (left, right) in {"A_B": ("A", "B"), "A_C": ("A", "C"), "B_C": ("B", "C")}.items():
            pairs[pair][0].append(annotations[left]); pairs[pair][1].append(annotations[right])
        votes = Counter(annotations.values())
        label, count = votes.most_common(1)[0]
        final = label if count >= 2 else "AMBIGUOUS"
        source = "blind_majority" if count >= 2 else "three_way_disagreement_retained_ambiguous"
        template = data["A"][key]
        candidates = set(filter(None, template["candidate_person_ids"].split("|")))
        rows.append({
            "audit_id": key[0], "evidence_id": key[1], "source_group": images[key[0]]["source_group"],
            "sampling_stratum": images[key[0]]["sampling_stratum"],
            "evidence_class": template["evidence_class"], "candidate_person_ids": template["candidate_person_ids"],
            "all_visible_person_ids": template["all_visible_person_ids"],
            "annotator_a": annotations["A"], "annotator_b": annotations["B"], "annotator_c": annotations["C"],
            "final_assignment": final, "final_label_source": source,
            "all_three_agree": int(len(votes) == 1), "human_owner_in_geometric_candidates": "" if final in SPECIAL else int(final in candidates),
            "human_owner_outside_geometric_candidates": "" if final in SPECIAL else int(final not in candidates),
        })

    out.mkdir(parents=True)
    write_csv(out / "three_pass_open_set_assignment_rows.csv", rows)
    pairwise = []
    for name, (left, right) in pairs.items():
        pairwise.append({"pair": name, "evidence_boxes": len(left), "exact_agreement_rate": exact_rate(left, right), "cohen_kappa_auxiliary": cohen_kappa(left, right)})
    write_csv(out / "pairwise_agreement.csv", pairwise)

    determinate = [row for row in rows if row["final_assignment"] not in SPECIAL]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_source[row["source_group"]].append(row)
    source_rows = []
    for source, items in sorted(by_source.items()):
        source_determinate = [row for row in items if row["final_assignment"] not in SPECIAL]
        source_rows.append({
            "source_group": source, "evidence_boxes": len(items), "determinate_human_owner_rows": len(source_determinate),
            "none_rows": sum(row["final_assignment"] == "NONE" for row in items), "ambiguous_rows": sum(row["final_assignment"] == "AMBIGUOUS" for row in items),
            "all_three_agree_rows": sum(row["all_three_agree"] for row in items),
            "owner_in_geometric_candidate_count": sum(row["human_owner_in_geometric_candidates"] == 1 for row in source_determinate),
            "owner_in_geometric_candidate_rate": (sum(row["human_owner_in_geometric_candidates"] == 1 for row in source_determinate) / len(source_determinate)) if source_determinate else math.nan,
        })
    write_csv(out / "candidate_coverage_by_source_group.csv", source_rows)

    summary = {
        "status": "three_frozen_open_set_blind_passes_analyzed",
        "selection": "pre-frozen source-stratified random image sample; no resampling",
        "images": len(images), "evidence_boxes": len(rows), "all_three_exact_agreement_count": sum(row["all_three_agree"] for row in rows),
        "all_three_exact_agreement_rate": sum(row["all_three_agree"] for row in rows) / len(rows),
        "pairwise": pairwise,
        "blind_majority_rows": sum(row["final_label_source"] == "blind_majority" for row in rows),
        "three_way_disagreement_rows_retained_ambiguous": sum(row["final_label_source"] != "blind_majority" for row in rows),
        "determinate_visible_human_owner_rows": len(determinate),
        "human_owner_in_geometric_candidate_count": sum(row["human_owner_in_geometric_candidates"] == 1 for row in determinate),
        "human_owner_in_geometric_candidate_recall": (sum(row["human_owner_in_geometric_candidates"] == 1 for row in determinate) / len(determinate)) if determinate else math.nan,
        "human_owner_outside_geometric_candidate_count": sum(row["human_owner_outside_geometric_candidates"] == 1 for row in determinate),
        "none_rows": sum(row["final_assignment"] == "NONE" for row in rows),
        "ambiguous_rows": sum(row["final_assignment"] == "AMBIGUOUS" for row in rows),
        "frozen_passes": locks,
        "interpretation": (
            "This quantifies agreement and whether a blind open-set human owner falls within the "
            "original geometric candidate list on the sampled rows. It is not a full-corpus semantic "
            "truth, a temporal identity study, a cluster-valid risk guarantee, or deployment validation."
        ),
    }
    (out / "open_set_random_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
