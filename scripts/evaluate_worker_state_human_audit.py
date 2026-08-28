"""Evaluate cached detector decisions against the frozen open-set human states.

The human overall_state is the primary audit endpoint because it is the
annotator's direct worker-level judgment. Component states are retained as
descriptive fields and are never silently recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import formal_worker_state_experiment as formal
from analyze_cached_rc_wssi_robustness import (
    ASSIGNMENT_MODES,
    evaluate_counts,
    evaluate_record,
    infer_worker_state,
    risk_values,
    select_ordered,
)


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ALPHA = 0.30
DELTA = 0.10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cache_records(cache: dict) -> tuple[list[dict], list[dict]]:
    images = cache["images"]
    return ([r for r in images if r.get("role") == "calibration"], [r for r in images if r.get("role") == "test"])


def chosen_policy(calibration: list[dict], mode: str) -> tuple[float | None, bool, dict | None]:
    rows = []
    for threshold in THRESHOLDS:
        counts = evaluate_counts(calibration, threshold, mode)
        errors, trials, risk, ucb = risk_values(counts, "automatic_safe_all_unsafe", DELTA)
        rows.append({"threshold": threshold, "selection_errors": errors, "selection_trials": trials, "selection_risk": risk, "selection_ucb": ucb, "coverage": formal.safe_div(counts["matched_decided"], counts["matched"])})
    selected = select_ordered(rows, ALPHA)
    return (None, True, None) if selected is None else (float(selected["threshold"]), False, selected)


def infer_for_mode(record: dict, threshold: float, mode: str, fallback: bool) -> list[dict]:
    persons = [p for p in record["predictions"] if p["cls"] == formal.PERSON_CLASS]
    evidence = [p for p in record["predictions"] if p["cls"] in formal.SAFETY_CLASSES]
    if fallback:
        states = infer_worker_state(persons, evidence, threshold, "global_pooling")
        for state in states:
            state["state"] = "review"
        return states
    return infer_worker_state(persons, evidence, threshold, mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--human-consensus", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    human = read_csv(args.human_consensus)
    if len({(r["audit_id"], r["person_id"]) for r in human}) != len(human):
        raise ValueError("duplicate human consensus key")
    manifest = {r["audit_id"]: r for r in read_csv(args.audit_root / "audit_image_manifest.csv")}
    by_audit = defaultdict(dict)
    for row in human:
        by_audit[row["audit_id"]][row["person_id"]] = row
    caches = {}
    for path in args.cache_root.glob("r10_s*_f*.json"):
        cache = load_cache(path)
        calibration, test = cache_records(cache)
        tag = cache["tag"]
        parts = tag.split("_")
        rate = int(parts[0][1:]); seed = int(parts[1][1:]); fold = int(parts[2][1:])
        if rate == 10:
            caches[(seed, fold)] = (calibration, test, tag)
    if len(caches) != 15:
        raise FileNotFoundError(f"expected 15 rate-10 seed-fold caches, found {len(caches)}")

    policies = {}
    for (seed, fold), (calibration, _, tag) in sorted(caches.items()):
        for mode in ASSIGNMENT_MODES:
            threshold, fallback, selected = chosen_policy(calibration, mode)
            policies[(seed, fold, mode)] = {"tag": tag, "threshold": threshold if threshold is not None else max(THRESHOLDS), "fallback": fallback, "selection_ucb": "" if selected is None else selected["selection_ucb"], "calibration_trials": "" if selected is None else selected["selection_trials"], "calibration_errors": "" if selected is None else selected["selection_errors"]}

    record_index = {}
    for (seed, fold), (_, test, _) in sorted(caches.items()):
        for record in test:
            record_index[(seed, fold, record["source_group"], record["image_name"])] = record

    raw = []
    ordered_audits = sorted(manifest, key=lambda x: int(x[1:]))
    for (seed, fold), (_, _, _) in sorted(caches.items()):
        policy_by_mode = {mode: policies[(seed, fold, mode)] for mode in ASSIGNMENT_MODES}
        for audit_id in ordered_audits:
            image = manifest[audit_id]
            if int(image["fold"]) != fold:
                continue
            record = record_index.get(
                (seed, fold, image["source_group"], image["image_name"])
            )
            if record is None:
                raise KeyError(
                    f"missing test record for seed={seed}, fold={fold}, "
                    f"group={image['source_group']}, image={image['image_name']}"
                )
            gt_persons = sorted(
                [g for g in record["ground_truth"] if g["cls"] == formal.PERSON_CLASS],
                key=lambda b: (b["xyxy"][0], b["xyxy"][1]),
            )
            pred_persons = [
                p for p in record["predictions"] if p["cls"] == formal.PERSON_CLASS
            ]
            matches, _, _ = formal.match_persons(pred_persons, gt_persons)
            gt_to_pred = {gi: pi for pi, gi, _ in matches}
            human_rows = by_audit[audit_id]
            if len(human_rows) != len(gt_persons):
                raise ValueError(f"{audit_id}: human/ground-truth person count mismatch")
            person_ids = sorted(human_rows, key=lambda x: int(x[1:]))
            if len(person_ids) != len(gt_persons):
                raise ValueError(f"{audit_id}: non-contiguous or incomplete person IDs")
            for mode in ASSIGNMENT_MODES:
                policy = policy_by_mode[mode]
                states = infer_for_mode(
                    record, policy["threshold"], mode, policy["fallback"]
                )
                for index, person_id in enumerate(person_ids):
                    h = human_rows[person_id]
                    pi = gt_to_pred.get(index)
                    pred_state = "UNMATCHED" if pi is None else states[pi]["state"].upper()
                    raw.append({
                        "seed": seed, "audit_id": audit_id,
                        "source_group": image["source_group"], "fold": fold,
                        "image_name": image["image_name"], "person_id": person_id,
                        "method": mode, "cell": policy["tag"],
                        "threshold": policy["threshold"],
                        "fallback": int(policy["fallback"]),
                        "human_helmet": h["helmet_state"],
                        "human_vest": h["vest_state"],
                        "human_overall": h["overall_state"],
                        "human_confidence": h["annotator_confidence"],
                        "predicted_state": pred_state,
                        "matched": int(pi is not None),
                        "exact_overall": int(
                            pi is not None and h["overall_state"] == pred_state
                        ),
                    })

    summary = []
    group_rows = []
    for (seed, fold), (_, _, _) in sorted(caches.items()):
        for mode in ASSIGNMENT_MODES:
            rows = [
                r for r in raw
                if r["method"] == mode and int(r["seed"]) == seed and int(r["fold"]) == fold
            ]
            if not rows:
                raise ValueError(f"empty human audit cell: seed={seed}, fold={fold}, method={mode}")
            matched = [r for r in rows if int(r["matched"]) == 1]
            determined = [r for r in matched if r["predicted_state"] in {"SAFE", "UNSAFE"}]
            confusion = Counter((r["human_overall"], r["predicted_state"]) for r in matched)
            unsafe = [r for r in rows if r["human_overall"] == "UNSAFE"]
            unsafe_auto_safe = sum(r["predicted_state"] == "SAFE" for r in unsafe)
            unsafe_composite = sum(r["predicted_state"] in {"SAFE", "UNMATCHED"} for r in unsafe)
            def rate(n, d): return n / d if d else ""
            summary.append({
                "seed": seed, "fold": fold, "cell": policies[(seed, fold, mode)]["tag"],
                "method": mode, "human_worker_rows": len(rows), "matched_rows": len(matched), "unmatched_rows": len(rows)-len(matched),
                "matched_overall_exact": rate(sum(int(r["exact_overall"]) for r in rows), len(matched)),
                "decision_rate_all_human_rows": rate(len(determined), len(rows)), "review_or_unmatched_rate": rate(len(rows)-len(determined), len(rows)),
                "human_unsafe_denominator": len(unsafe), "unsafe_auto_safe_errors": unsafe_auto_safe, "unsafe_auto_safe_rate": rate(unsafe_auto_safe, len(unsafe)),
                "unsafe_composite_errors": unsafe_composite, "unsafe_composite_rate": rate(unsafe_composite, len(unsafe)),
                "human_safe_to_unsafe": confusion[("SAFE", "UNSAFE")], "human_unsafe_to_safe": confusion[("UNSAFE", "SAFE")],
                "human_review_to_safe": confusion[("REVIEW", "SAFE")], "human_review_to_unsafe": confusion[("REVIEW", "UNSAFE")],
            })
            for group in sorted({r["source_group"] for r in rows}):
                gr = [r for r in rows if r["source_group"] == group]
                gm = [r for r in gr if int(r["matched"]) == 1]
                gu = [r for r in gr if r["human_overall"] == "UNSAFE"]
                group_rows.append({
                    "seed": seed, "fold": fold,
                    "cell": policies[(seed, fold, mode)]["tag"], "method": mode,
                    "source_group": group, "rows": len(gr), "matched": len(gm),
                    "unmatched": len(gr) - len(gm),
                    "matched_exact": rate(sum(int(r["exact_overall"]) for r in gr), len(gm)),
                    "unsafe_n": len(gu),
                    "unsafe_auto_safe_errors": sum(r["predicted_state"] == "SAFE" for r in gu),
                    "unsafe_composite_errors": sum(r["predicted_state"] in {"SAFE", "UNMATCHED"} for r in gu),
                    "fallback_cells": sum(int(r["fallback"]) for r in gr),
                })

    aggregate = []
    for mode in ASSIGNMENT_MODES:
        cell_rows = [r for r in summary if r["method"] == mode]
        group_mode_rows = [r for r in group_rows if r["method"] == mode]
        unsafe_n = sum(int(r["human_unsafe_denominator"]) for r in cell_rows)
        auto_errors = sum(int(r["unsafe_auto_safe_errors"]) for r in cell_rows)
        composite_errors = sum(int(r["unsafe_composite_errors"]) for r in cell_rows)
        matched_n = sum(int(r["matched_rows"]) for r in cell_rows)
        exact_n = sum(int(r["exact_overall"]) for r in raw if r["method"] == mode)
        group_composites = [
            int(r["unsafe_composite_errors"]) / int(r["unsafe_n"])
            for r in group_mode_rows if int(r["unsafe_n"]) > 0
        ]
        aggregate.append({
            "method": mode,
            "cells": len(cell_rows),
            "cell_mean_matched_exact": mean([float(r["matched_overall_exact"]) for r in cell_rows]),
            "cell_mean_decision_rate": mean([float(r["decision_rate_all_human_rows"]) for r in cell_rows]),
            "repeated_seed_pooled_matched_exact": exact_n / matched_n if matched_n else "",
            "repeated_seed_pooled_unsafe_n": unsafe_n,
            "repeated_seed_pooled_auto_safe_errors": auto_errors,
            "repeated_seed_pooled_auto_safe_rate": auto_errors / unsafe_n if unsafe_n else "",
            "repeated_seed_pooled_composite_errors": composite_errors,
            "repeated_seed_pooled_composite_rate": composite_errors / unsafe_n if unsafe_n else "",
            "filename_group_seed_macro_composite": mean(group_composites),
            "filename_group_seed_units": len(group_composites),
            "interpretation": "Repeated-seed summaries reuse the same sampled human rows; they are not independent worker samples.",
        })
    human_state_summary = []
    for state, count in sorted(Counter(r["overall_state"] for r in human).items()):
        human_state_summary.append({"human_overall": state, "rows": count, "fraction": count / len(human)})

    args.out.mkdir(parents=True)
    write_csv(args.out / "human_worker_state_method_rows.csv", raw)
    write_csv(args.out / "human_worker_state_method_summary.csv", summary)
    write_csv(args.out / "human_worker_state_group_summary.csv", group_rows)
    write_csv(args.out / "human_worker_state_aggregate_summary.csv", aggregate)
    write_csv(args.out / "human_state_distribution.csv", human_state_summary)
    write_csv(args.out / "selected_policy_audit.csv", [{"seed":seed,"fold":fold,"method":mode,**p} for (seed, fold, mode),p in sorted(policies.items())])
    seed_count = len({seed for seed, _ in caches})
    report = {"status":"complete_sampled_human_worker_state_evaluation","audit_images":len(manifest),"human_worker_rows":len(human),"raw_method_rows":len(raw),"expected_raw_method_rows":len(human)*len(ASSIGNMENT_MODES)*seed_count,"seed_count":seed_count,"cell_count":len(caches),"filename_groups":len({r["source_group"] for r in read_csv(args.audit_root / "audit_image_manifest.csv")}),"cache_cells":sorted({p["tag"] for p in policies.values()}),"alpha":ALPHA,"delta":DELTA,"scope":"sampled independent open-set human worker-state audit; not full-corpus prevalence, cluster-valid inference, or deployment safety guarantee","human_consensus_sha256":hashlib.sha256(args.human_consensus.read_bytes()).hexdigest()}
    if len(raw) != report["expected_raw_method_rows"]:
        raise ValueError(f"raw row count mismatch: {len(raw)} != {report['expected_raw_method_rows']}")
    (args.out / "evaluation_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({**report,"summary":summary},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
