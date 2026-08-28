"""Check and summarize an independent worker-state audit.

This command never invents a consensus label. It requires two frozen passes,
reports their agreement, and writes a C-adjudication template when rows
disagree. Only after adjudication is supplied will it write consensus labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("audit_id", "person_id", "helmet_state", "vest_state", "overall_state", "annotator_confidence", "visibility_issue", "notes")
STATE_FIELDS = ("helmet_state", "vest_state", "overall_state")
META_FIELDS = ("annotator_confidence", "visibility_issue")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if tuple(rows[0].keys()) != FIELDS:
        raise ValueError(f"{path}: unexpected header")
    return [{field: row.get(field, "") for field in FIELDS} for row in rows]


def read_adjudication(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"audit_id", "person_id", "final_helmet_state", "final_vest_state", "final_overall_state", "adjudicator_confidence", "adjudication_notes"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: unexpected adjudication header")
    return rows


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else list(FIELDS)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def frozen(root: Path, annotator: str) -> tuple[list[dict[str, str]], dict]:
    folder = root / f"annotator_{annotator}"
    lock = folder / "ANNOTATION_FINALIZED.json"
    csv_path = folder / "worker_state.csv"
    if not lock.is_file() or not csv_path.is_file():
        raise RuntimeError(f"annotator {annotator} is not frozen")
    record = read_json(lock)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if record.get("csv_sha256") != digest:
        raise RuntimeError(f"annotator {annotator}: CSV changed after freeze")
    rows = read_csv(csv_path)
    if len(rows) != record.get("completed_worker_rows"):
        raise RuntimeError(f"annotator {annotator}: frozen row count mismatch")
    return rows, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, default=None, help="completed C CSV for disagreement rows")
    args = parser.parse_args()
    annotator_b = "B_retry" if (args.audit_root / "annotator_B_retry" / "ANNOTATION_FINALIZED.json").is_file() else "B"
    a_rows, a_lock = frozen(args.audit_root, "A")
    b_rows, b_lock = frozen(args.audit_root, annotator_b)
    a = {(r["audit_id"], r["person_id"]): r for r in a_rows}
    b = {(r["audit_id"], r["person_id"]): r for r in b_rows}
    if set(a) != set(b):
        raise RuntimeError("A/B worker row keys differ")
    manifest = {r["audit_id"]: r for r in csv.DictReader((args.audit_root / "audit_image_manifest.csv").open("r", encoding="utf-8-sig", newline=""))}
    disagreements = []
    agreements = []
    for key in sorted(a, key=lambda x: (int(x[0][1:]), int(x[1][1:]))):
        same = all(a[key][field] == b[key][field] for field in STATE_FIELDS)
        base = {"audit_id": key[0], "person_id": key[1], "source_group": manifest[key[0]]["source_group"], "image_name": manifest[key[0]]["image_name"], "annotator_A_helmet": a[key]["helmet_state"], "annotator_A_vest": a[key]["vest_state"], "annotator_A_overall": a[key]["overall_state"], "annotator_B_helmet": b[key]["helmet_state"], "annotator_B_vest": b[key]["vest_state"], "annotator_B_overall": b[key]["overall_state"], "A_confidence": a[key]["annotator_confidence"], "B_confidence": b[key]["annotator_confidence"]}
        (agreements if same else disagreements).append(base)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "worker_state_disagreements.csv", disagreements)
    if disagreements:
        adjudication = []
        for row in disagreements:
            adjudication.append({**row, "final_helmet_state":"", "final_vest_state":"", "final_overall_state":"", "adjudicator_confidence":"", "adjudication_notes":""})
        write_csv(args.out / "adjudication_template.csv", adjudication)
    else:
        write_csv(args.out / "adjudication_template.csv", [])

    def agreement(field):
        return sum(a[k][field] == b[k][field] for k in a) / len(a) if a else 0.0
    strata = defaultdict(lambda: {"rows": 0, "agreement_rows": 0})
    for row in agreements + disagreements:
        s = manifest[row["audit_id"]]["sampling_stratum"]
        strata[s]["rows"] += 1
        strata[s]["agreement_rows"] += int(row in agreements)
    summary = {
        "protocol": "independent_open_set_worker_state_audit_v1",
        "annotator_A_lock": a_lock,
        "annotator_B": annotator_b,
        "annotator_B_lock": b_lock,
        "worker_rows": len(a),
        "exact_component_row_agreement": len(agreements) / len(a) if a else 0.0,
        "disagreement_rows": len(disagreements),
        "helmet_agreement": agreement("helmet_state"),
        "vest_agreement": agreement("vest_state"),
        "overall_agreement": agreement("overall_state"),
        "confidence_agreement": agreement("annotator_confidence"),
        "visibility_issue_agreement": agreement("visibility_issue"),
        "strata": dict(strata),
        "status": "requires_C_adjudication" if disagreements else "A_B_agree_no_adjudication_required",
    }
    if args.adjudication:
        c_rows = read_adjudication(args.adjudication)
        by_key = {(r["audit_id"], r["person_id"]): r for r in c_rows}
        expected = {(r["audit_id"], r["person_id"]) for r in disagreements}
        if set(by_key) != expected:
            raise RuntimeError("adjudication keys do not exactly match disagreement rows")
        allowed = {"SAFE", "UNSAFE", "REVIEW"}
        if any(r[f"final_{field}"] not in allowed for r in c_rows for field in ("helmet_state", "vest_state", "overall_state")):
            raise RuntimeError("adjudication contains an invalid or incomplete state")
        consensus = []
        c_by_key = {(r["audit_id"], r["person_id"]): r for r in c_rows}
        for key in sorted(a, key=lambda x: (int(x[0][1:]), int(x[1][1:]))):
            if key in c_by_key:
                c = c_by_key[key]
                consensus.append({**a[key], "helmet_state":c["final_helmet_state"], "vest_state":c["final_vest_state"], "overall_state":c["final_overall_state"], "annotator_confidence":c["adjudicator_confidence"], "visibility_issue":a[key]["visibility_issue"], "notes":"adjudicated: " + c["adjudication_notes"]})
            else:
                consensus.append(a[key])
        write_csv(args.out / "human_consensus_worker_state.csv", consensus)
        summary["status"] = "consensus_written_after_C_adjudication"
        summary["adjudication_rows"] = len(c_rows)
    (args.out / "worker_state_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
