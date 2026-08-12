"""CPU-only robustness analyses from RC-WSSI prediction caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import formal_worker_state_experiment as formal


ASSIGNMENT_MODES = (
    "proposed_lexicographic",
    "max_iou",
    "center_inside",
    "hungarian_iou",
    "global_pooling",
)
RISK_SCOPES = (
    "automatic_safe_all_unsafe",
    "automatic_safe_matched_unsafe",
    "miss_or_safe_all_unsafe",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_assignment(persons: list[dict], evidence: list[dict]) -> dict[int, list[dict]]:
    assigned = {index: [] for index in range(len(persons))}
    for item in evidence:
        candidates = []
        ex, ey = center(item["xyxy"])
        for index, person in enumerate(persons):
            support = formal.expand_box(person["xyxy"], formal.PERSON_SCALE)
            if not formal.center_inside(item["xyxy"], support):
                continue
            px, py = center(support)
            width = max(1.0, support[2] - support[0])
            height = max(1.0, support[3] - support[1])
            distance = ((ex - px) / width) ** 2 + ((ey - py) / height) ** 2
            candidates.append((distance, formal.box_area(support), index))
        if candidates:
            assigned[min(candidates)[2]].append(item)
    return assigned


def max_iou_assignment(persons: list[dict], evidence: list[dict]) -> dict[int, list[dict]]:
    assigned = {index: [] for index in range(len(persons))}
    for item in evidence:
        candidates = []
        for index, person in enumerate(persons):
            support = formal.expand_box(person["xyxy"], formal.PERSON_SCALE)
            score = formal.iou(item["xyxy"], support)
            if score > 0:
                candidates.append((score, -index, index))
        if candidates:
            assigned[max(candidates)[2]].append(item)
    return assigned


def hungarian_assignment(persons: list[dict], evidence: list[dict]) -> dict[int, list[dict]]:
    assigned = {index: [] for index in range(len(persons))}
    if not persons:
        return assigned
    for class_id in sorted(formal.SAFETY_CLASSES):
        class_items = [item for item in evidence if item["cls"] == class_id]
        if not class_items:
            continue
        scores = np.zeros((len(persons), len(class_items)), dtype=float)
        for person_index, person in enumerate(persons):
            support = formal.expand_box(person["xyxy"], formal.PERSON_SCALE)
            for item_index, item in enumerate(class_items):
                scores[person_index, item_index] = formal.iou(item["xyxy"], support)
        person_indices, item_indices = linear_sum_assignment(-scores)
        for person_index, item_index in zip(person_indices, item_indices):
            if scores[person_index, item_index] > 0:
                assigned[int(person_index)].append(class_items[int(item_index)])
    return assigned


def assign(persons: list[dict], evidence: list[dict], mode: str) -> dict[int, list[dict]]:
    if mode == "proposed_lexicographic":
        return formal.assign_safety_to_persons(persons, evidence)
    if mode == "max_iou":
        return max_iou_assignment(persons, evidence)
    if mode == "center_inside":
        return center_assignment(persons, evidence)
    if mode == "hungarian_iou":
        return hungarian_assignment(persons, evidence)
    if mode == "global_pooling":
        return {index: list(evidence) for index in range(len(persons))}
    raise ValueError(mode)


def infer_worker_state(
    persons: list[dict], evidence: list[dict], threshold: float, mode: str
) -> list[dict]:
    assigned = assign(persons, evidence, mode)
    states = []
    for index, person in enumerate(persons):
        items = assigned.get(index, [])
        head = formal.infer_component_state(
            formal.best_conf(items, {formal.HELMET_SAFE}),
            formal.best_conf(items, {formal.HELMET_UNSAFE}),
            threshold,
        )
        vest = formal.infer_component_state(
            formal.best_conf(items, {formal.VEST_SAFE}),
            formal.best_conf(items, {formal.VEST_UNSAFE}),
            threshold,
        )
        if head == "unsafe" or vest == "unsafe":
            state = "unsafe"
        elif head == "safe" and vest == "safe":
            state = "safe"
        else:
            state = "review"
        states.append(
            {
                "person": person,
                "head": head,
                "vest": vest,
                "state": state,
                "assigned_safety_count": len(items),
            }
        )
    return states


def evaluate_record(record: dict, threshold: float, mode: str, review_all: bool = False) -> dict:
    original_infer = formal.infer_worker_state
    original_collapse = formal.collapse_state

    def patched_infer(persons, evidence, thr, assignment="worker"):
        selected_mode = "global_pooling" if assignment == "global" else mode
        return infer_worker_state(persons, evidence, thr, selected_mode)

    def patched_collapse(state, policy):
        if policy == "review_all":
            return "review"
        return original_collapse(state, policy)

    formal.infer_worker_state = patched_infer
    formal.collapse_state = patched_collapse
    try:
        return formal.evaluate_image(
            record["predictions"],
            record["ground_truth"],
            threshold,
            policy="review_all" if review_all else "separate",
            pred_assignment="worker",
        )
    finally:
        formal.infer_worker_state = original_infer
        formal.collapse_state = original_collapse


def extended_metrics(counts: dict, delta: float) -> dict:
    metrics = formal.metrics_from_counts(counts, risk_delta=delta)
    matched_unsafe = counts["gt_unsafe"] - counts["unsafe_unmatched_gt"]
    metrics.update(
        {
            "matched_unsafe": matched_unsafe,
            "automatic_safe_matched_unsafe": formal.safe_div(
                counts["unsafe_to_safe"], matched_unsafe
            ),
            "automatic_safe_matched_unsafe_ucb": formal.risk_upper_bound(
                counts["unsafe_to_safe"], matched_unsafe, delta
            ),
            "miss_or_safe_all_unsafe": formal.safe_div(
                counts["unsafe_to_safe"] + counts["unsafe_unmatched_gt"],
                counts["gt_unsafe"],
            ),
            "miss_or_safe_all_unsafe_ucb": formal.risk_upper_bound(
                counts["unsafe_to_safe"] + counts["unsafe_unmatched_gt"],
                counts["gt_unsafe"],
                delta,
            ),
        }
    )
    return metrics


def risk_values(counts: dict, scope: str, delta: float) -> tuple[int, int, float, float]:
    if scope == "automatic_safe_all_unsafe":
        errors = counts["unsafe_to_safe"]
        trials = counts["gt_unsafe"]
    elif scope == "automatic_safe_matched_unsafe":
        errors = counts["unsafe_to_safe"]
        trials = counts["gt_unsafe"] - counts["unsafe_unmatched_gt"]
    elif scope == "miss_or_safe_all_unsafe":
        errors = counts["unsafe_to_safe"] + counts["unsafe_unmatched_gt"]
        trials = counts["gt_unsafe"]
    else:
        raise ValueError(scope)
    return errors, trials, formal.safe_div(errors, trials), formal.risk_upper_bound(errors, trials, delta)


def split_cache(cache: dict) -> tuple[list[dict], list[dict]]:
    images = cache["images"]
    if images and "role" in images[0]:
        calibration = [item for item in images if item["role"] == "calibration"]
        test = [item for item in images if item["role"] == "test"]
        return calibration, test
    metadata = cache["split_metadata"]
    calibration_sources = set(filter(None, metadata["calibration_source_groups"].split("|")))
    calibration = [item for item in images if item["source_group"] in calibration_sources]
    test = [item for item in images if item["source_group"] not in calibration_sources]
    return calibration, test


def cell_identity(cache: dict, path: Path) -> tuple[int, int, int, str]:
    tag = cache.get("tag", path.stem)
    parts = tag.replace("r", "", 1).replace("s", "").replace("f", "").split("_")
    if len(parts) != 3:
        import re

        match = re.search(r"r(\d+)_s(\d+)_f(\d+)", tag)
        if not match:
            raise ValueError(f"cannot parse cache tag: {tag}")
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), tag
    return int(parts[0]), int(parts[1]), int(parts[2]), tag


def evaluate_counts(records: list[dict], threshold: float, mode: str, review_all=False) -> dict:
    return formal.aggregate_counts(
        [evaluate_record(record, threshold, mode, review_all) for record in records]
    )


def select_ordered(calibration_rows: list[dict], alpha: float) -> dict | None:
    certified = []
    for row in sorted(calibration_rows, key=lambda item: item["threshold"], reverse=True):
        if row["selection_ucb"] > alpha:
            break
        certified.append(row)
    if not certified:
        return None
    return max(certified, key=lambda item: (item["coverage"], -item["threshold"]))


def bootstrap_summary(rows: list[dict], metric: str, draws: int, seed: int) -> tuple[float, float, float]:
    values = [float(row[metric]) for row in rows]
    estimate = statistics.mean(values)
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return estimate, estimate, estimate
    rng = random.Random(seed)
    sampled = []
    for _ in range(draws):
        chosen = [rng.choice(seeds) for _ in seeds]
        draw_values = [float(row[metric]) for selected in chosen for row in by_seed[selected]]
        sampled.append(statistics.mean(draw_values))
    sampled.sort()
    low = sampled[int(0.025 * (len(sampled) - 1))]
    high = sampled[int(0.975 * (len(sampled) - 1))]
    return estimate, low, high


def summarize(rows: list[dict], draws: int) -> list[dict]:
    keys = (
        "rate",
        "assignment_mode",
        "risk_scope",
        "alpha",
        "delta",
    )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    metrics = (
        "selection_risk",
        "unsafe_f1",
        "coverage",
        "review_rate",
        "prediction_decision_rate",
        "unsafe_unmatched_gt_rate",
        "miss_or_safe_all_unsafe",
    )
    for key, group in sorted(groups.items()):
        item = dict(zip(keys, key))
        item.update(
            {
                "cells": len(group),
                "fallbacks": sum(int(row["review_all_fallback"]) for row in group),
                "selected_threshold_mean_nonfallback": statistics.mean(
                    [row["threshold"] for row in group if not row["review_all_fallback"]]
                )
                if any(not row["review_all_fallback"] for row in group)
                else "",
            }
        )
        stable_seed = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)
        for metric in metrics:
            mean, low, high = bootstrap_summary(group, metric, draws, stable_seed)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci_low"] = low
            item[f"{metric}_ci_high"] = high
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    result_rows = []
    calibration_rows = []
    cache_paths = sorted(args.cache_root.glob("*.json"))
    if not cache_paths:
        raise FileNotFoundError(f"no cache JSON files under {args.cache_root}")
    for cache_path in cache_paths:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        rate, seed, fold, tag = cell_identity(cache, cache_path)
        calibration, test = split_cache(cache)
        for mode in ASSIGNMENT_MODES:
            count_cache = {
                ("calibration", threshold): evaluate_counts(calibration, threshold, mode)
                for threshold in args.thresholds
            }
            test_count_cache = {
                threshold: evaluate_counts(test, threshold, mode) for threshold in args.thresholds
            }
            fallback_test_counts = evaluate_counts(test, max(args.thresholds), mode, review_all=True)
            for delta in args.deltas:
                for scope in RISK_SCOPES:
                    scoped_calibration = []
                    for threshold in args.thresholds:
                        counts = count_cache[("calibration", threshold)]
                        errors, trials, risk, ucb = risk_values(counts, scope, delta)
                        metrics = extended_metrics(counts, delta)
                        row = {
                            "rate": rate,
                            "seed": seed,
                            "fold": fold,
                            "tag": tag,
                            "assignment_mode": mode,
                            "risk_scope": scope,
                            "delta": delta,
                            "threshold": threshold,
                            "selection_errors": errors,
                            "selection_trials": trials,
                            "selection_risk": risk,
                            "selection_ucb": ucb,
                            **metrics,
                        }
                        scoped_calibration.append(row)
                        calibration_rows.append(row)
                    for alpha in args.alphas:
                        selected = select_ordered(scoped_calibration, alpha)
                        fallback = selected is None
                        threshold = max(args.thresholds) if fallback else selected["threshold"]
                        test_counts = fallback_test_counts if fallback else test_count_cache[threshold]
                        errors, trials, risk, ucb = risk_values(test_counts, scope, delta)
                        result_rows.append(
                            {
                                "rate": rate,
                                "seed": seed,
                                "fold": fold,
                                "tag": tag,
                                "assignment_mode": mode,
                                "risk_scope": scope,
                                "alpha": alpha,
                                "delta": delta,
                                "threshold": threshold,
                                "review_all_fallback": int(fallback),
                                "calibration_selected_ucb": 1.0 if fallback else selected["selection_ucb"],
                                "selection_errors": errors,
                                "selection_trials": trials,
                                "selection_risk": risk,
                                "selection_ucb": ucb,
                                **extended_metrics(test_counts, delta),
                            }
                        )
        print(f"[done] {tag}", flush=True)

    summary = summarize(result_rows, args.bootstrap_draws)
    write_csv(args.out / "robustness_cell_rows.csv", result_rows)
    write_csv(args.out / "robustness_calibration_rows.csv", calibration_rows)
    write_csv(args.out / "robustness_summary.csv", summary)
    assignment = [
        row
        for row in summary
        if row["risk_scope"] == "automatic_safe_all_unsafe"
        and float(row["alpha"]) == 0.30
        and float(row["delta"]) == 0.10
    ]
    sensitivity = [row for row in summary if row["assignment_mode"] == "proposed_lexicographic"]
    write_csv(args.out / "assignment_baseline_summary.csv", assignment)
    write_csv(args.out / "alpha_delta_risk_scope_summary.csv", sensitivity)

    report = [
        "# RC-WSSI cached robustness analysis",
        "",
        f"- Cache cells: {len(cache_paths)}.",
        f"- Assignment modes: {', '.join(ASSIGNMENT_MODES)}.",
        f"- Risk scopes: {', '.join(RISK_SCOPES)}.",
        f"- Alpha values: {args.alphas}.",
        f"- Delta values: {args.deltas}.",
        "- All analyses use cached detector predictions and therefore require no GPU inference.",
        "- The legacy cache remains affected by detector-validation reuse; these analyses are diagnostic until rerun on the four-way source-disjoint protocol.",
        "",
        "## Outputs",
        "",
        "- `robustness_cell_rows.csv`: one row per protocol cell and setting.",
        "- `robustness_summary.csv`: seed-cluster bootstrap summaries.",
        "- `assignment_baseline_summary.csv`: assignment comparison at alpha=0.30, delta=0.10.",
        "- `alpha_delta_risk_scope_summary.csv`: proposed assignment sensitivity.",
    ]
    (args.out / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "cache_root": str(args.cache_root),
        "cache_cells": len(cache_paths),
        "alphas": args.alphas,
        "deltas": args.deltas,
        "thresholds": args.thresholds,
        "assignment_modes": ASSIGNMENT_MODES,
        "risk_scopes": RISK_SCOPES,
        "bootstrap_draws": args.bootstrap_draws,
    }
    (args.out / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[complete] {args.out}", flush=True)


if __name__ == "__main__":
    main()
