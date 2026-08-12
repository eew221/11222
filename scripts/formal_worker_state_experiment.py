import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import beta

from evaluate_person_conditioned_gate import (
    box_area,
    center_inside,
    expand_box,
    intersection,
    iou,
    read_yolo_boxes,
)
from run_person_gate_batch import (
    PERSON_CLASS,
    SAFETY_CLASSES,
    collect_val_gt,
    ensure_background_crops,
    eval_bg_at_threshold,
    eval_val_at_threshold,
    predict_paths,
)


HELMET_SAFE = 0
HELMET_UNSAFE = 1
VEST_UNSAFE = 2
VEST_SAFE = 4

PERSON_SCALE = 1.15
MIN_INTER_OVER_SAFETY = 0.15
STATE_VARIANTS = [
    {
        "method": "state_selected_separate",
        "policy": "separate",
        "pred_assignment": "worker",
    },
    {
        "method": "state_selected_force_unsafe",
        "policy": "force_unsafe",
        "pred_assignment": "worker",
    },
    {
        "method": "state_selected_force_safe",
        "policy": "force_safe",
        "pred_assignment": "worker",
    },
    {
        "method": "state_selected_no_assignment",
        "policy": "separate",
        "pred_assignment": "global",
    },
]


def stable_half(path):
    key = Path(path).name.encode("utf-8", errors="ignore")
    return int(hashlib.md5(key).hexdigest()[:8], 16) % 2


def imread(path):
    data = Path(path).read_bytes()
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def brightness(path):
    image = imread(path)
    if image is None:
        return 1.0
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()) / 255.0


def darkest_subset(paths, ratio=0.25):
    if not paths:
        return []
    ranked = sorted(paths, key=brightness)
    return ranked[: max(1, round(len(ranked) * ratio))]


def subset_dict(dct, keys):
    return {p: dct[p] for p in keys if p in dct}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def support_score(safety_box, person_box):
    support_box = expand_box(person_box, PERSON_SCALE)
    safety_area = max(1e-9, box_area(safety_box))
    overlap = intersection(safety_box, support_box) / safety_area
    return (
        overlap,
        iou(safety_box, support_box),
        1.0 if center_inside(safety_box, support_box) else 0.0,
    )


def best_conf(preds, cls_ids):
    vals = [p["conf"] for p in preds if p["cls"] in cls_ids]
    return max(vals) if vals else 0.0


def assign_safety_to_persons(persons, safety_preds):
    assigned = {i: [] for i in range(len(persons))}
    for pred in safety_preds:
        best_idx = None
        best_score = None
        for i, person in enumerate(persons):
            score = support_score(pred["xyxy"], person["xyxy"])
            if score[0] <= 0 and score[2] <= 0:
                continue
            # Exact geometric ties resolve to the smallest stable person index.
            if best_score is None or score > best_score or (
                score == best_score and i < best_idx
            ):
                best_score = score
                best_idx = i
        if best_idx is not None:
            assigned[best_idx].append(pred)
    return assigned


def infer_component_state(conf_safe, conf_unsafe, thr):
    if conf_safe < thr and conf_unsafe < thr:
        return "review"
    if conf_unsafe >= thr and conf_unsafe >= conf_safe:
        return "unsafe"
    if conf_safe >= thr:
        return "safe"
    return "review"


def infer_worker_state(persons, safety_preds, thr, assignment="worker"):
    if assignment == "worker":
        assigned = assign_safety_to_persons(persons, safety_preds)
    elif assignment == "global":
        assigned = {i: list(safety_preds) for i in range(len(persons))}
    else:
        raise ValueError(assignment)
    per_person = []
    for i, person in enumerate(persons):
        preds = assigned.get(i, [])
        head = infer_component_state(
            best_conf(preds, {HELMET_SAFE}),
            best_conf(preds, {HELMET_UNSAFE}),
            thr,
        )
        vest = infer_component_state(
            best_conf(preds, {VEST_SAFE}),
            best_conf(preds, {VEST_UNSAFE}),
            thr,
        )
        if head == "unsafe" or vest == "unsafe":
            overall = "unsafe"
        elif head == "safe" and vest == "safe":
            overall = "safe"
        else:
            overall = "review"
        per_person.append(
            {
                "person": person,
                "head": head,
                "vest": vest,
                "state": overall,
                "assigned_safety_count": len(preds),
            }
        )
    return per_person


def infer_gt_worker_state(gt_persons, gt_boxes):
    assigned = assign_safety_to_persons(gt_persons, gt_boxes)
    per_person = []
    for i, person in enumerate(gt_persons):
        preds = assigned.get(i, [])
        head_has_safe = any(p["cls"] == HELMET_SAFE for p in preds)
        head_has_unsafe = any(p["cls"] == HELMET_UNSAFE for p in preds)
        vest_has_safe = any(p["cls"] == VEST_SAFE for p in preds)
        vest_has_unsafe = any(p["cls"] == VEST_UNSAFE for p in preds)
        head = "unsafe" if head_has_unsafe and not head_has_safe else "safe" if head_has_safe and not head_has_unsafe else "review"
        vest = "unsafe" if vest_has_unsafe and not vest_has_safe else "safe" if vest_has_safe and not vest_has_unsafe else "review"
        if head == "unsafe" or vest == "unsafe":
            overall = "unsafe"
        elif head == "safe" and vest == "safe":
            overall = "safe"
        else:
            overall = "review"
        per_person.append(
            {
                "person": person,
                "head": head,
                "vest": vest,
                "state": overall,
                "assigned_safety_count": len(preds),
            }
        )
    return per_person


def match_persons(pred_persons, gt_persons, iou_thr=0.5):
    pred_order = sorted(range(len(pred_persons)), key=lambda i: pred_persons[i]["conf"], reverse=True)
    used_gt = set()
    matches = []
    unmatched_pred = []
    for pi in pred_order:
        pred_box = pred_persons[pi]["xyxy"]
        best_gi = None
        best_iou = 0.0
        for gi, gt in enumerate(gt_persons):
            if gi in used_gt:
                continue
            score = iou(pred_box, gt["xyxy"])
            if score > best_iou:
                best_iou = score
                best_gi = gi
        if best_gi is not None and best_iou >= iou_thr:
            used_gt.add(best_gi)
            matches.append((pi, best_gi, best_iou))
        else:
            unmatched_pred.append(pi)
    unmatched_gt = [gi for gi in range(len(gt_persons)) if gi not in used_gt]
    return matches, unmatched_pred, unmatched_gt


def collapse_state(state, policy):
    if policy == "separate":
        return state
    if policy == "force_unsafe":
        return "unsafe" if state == "review" else state
    if policy == "force_safe":
        return "safe" if state == "review" else state
    raise ValueError(policy)


def evaluate_image(preds, gt_boxes, thr, policy="separate", pred_assignment="worker"):
    persons_pred = [p for p in preds if p["cls"] == PERSON_CLASS]
    safety_pred = [p for p in preds if p["cls"] in SAFETY_CLASSES]
    pred_states = [dict(x) for x in infer_worker_state(persons_pred, safety_pred, thr, pred_assignment)]
    for item in pred_states:
        item["state"] = collapse_state(item["state"], policy)

    gt_persons = [g for g in gt_boxes if g["cls"] == PERSON_CLASS]
    gt_states = infer_gt_worker_state(gt_persons, gt_boxes)

    matches, unmatched_pred, unmatched_gt = match_persons(persons_pred, gt_persons)

    counts = {
        "gt_safe": 0,
        "gt_unsafe": 0,
        "gt_review": 0,
        "pred_safe": 0,
        "pred_unsafe": 0,
        "pred_review": 0,
        "tp_unsafe": 0,
        "fp_unsafe": 0,
        "fn_unsafe": 0,
        "tp_safe": 0,
        "fp_safe": 0,
        "fn_safe": 0,
        "unsafe_to_safe": 0,
        "unsafe_to_review": 0,
        "unsafe_unmatched_gt": 0,
        "safe_to_unsafe": 0,
        "safe_to_review": 0,
        "safe_unmatched_gt": 0,
        "review_to_safe": 0,
        "review_to_unsafe": 0,
        "review_to_review": 0,
        "matched": 0,
        "matched_decided": 0,
        "pred_decided": 0,
        "unmatched_pred": 0,
        "unmatched_gt": 0,
        "unmatched_pred_decided": 0,
    }

    def add_gt(label):
        counts[f"gt_{label}"] += 1

    def add_pred(label):
        counts[f"pred_{label}"] += 1

    for pi, gi, _ in matches:
        gt_state = gt_states[gi]["state"]
        pred_state = pred_states[pi]["state"]
        counts["matched"] += 1
        add_gt(gt_state)
        if pred_state in {"safe", "unsafe"}:
            add_pred(pred_state)
            counts["matched_decided"] += 1
            counts["pred_decided"] += 1
        else:
            add_pred("review")
        if gt_state == "unsafe":
            if pred_state == "unsafe":
                counts["tp_unsafe"] += 1
            elif pred_state == "safe":
                counts["fn_unsafe"] += 1
                counts["fp_safe"] += 1
                counts["unsafe_to_safe"] += 1
            else:
                counts["fn_unsafe"] += 1
                counts["unsafe_to_review"] += 1
        elif gt_state == "safe":
            if pred_state == "safe":
                counts["tp_safe"] += 1
            elif pred_state == "unsafe":
                counts["fn_safe"] += 1
                counts["fp_unsafe"] += 1
                counts["safe_to_unsafe"] += 1
            else:
                counts["fn_safe"] += 1
                counts["safe_to_review"] += 1
        elif gt_state == "review":
            if pred_state == "safe":
                counts["review_to_safe"] += 1
            elif pred_state == "unsafe":
                counts["review_to_unsafe"] += 1
            else:
                counts["review_to_review"] += 1

    for gi in unmatched_gt:
        counts["unmatched_gt"] += 1
        gt_state = gt_states[gi]["state"]
        add_gt(gt_state)
        if gt_state == "unsafe":
            counts["fn_unsafe"] += 1
            counts["unsafe_unmatched_gt"] += 1
        elif gt_state == "safe":
            counts["fn_safe"] += 1
            counts["safe_unmatched_gt"] += 1

    for pi in unmatched_pred:
        counts["unmatched_pred"] += 1
        pred_state = pred_states[pi]["state"]
        if pred_state in {"safe", "unsafe"}:
            add_pred(pred_state)
            counts["pred_decided"] += 1
            counts["unmatched_pred_decided"] += 1
            if pred_state == "unsafe":
                counts["fp_unsafe"] += 1
            elif pred_state == "safe":
                counts["fp_safe"] += 1
        else:
            add_pred("review")

    return counts


COUNT_KEYS = [
    "gt_safe",
    "gt_unsafe",
    "gt_review",
    "pred_safe",
    "pred_unsafe",
    "pred_review",
    "tp_unsafe",
    "fp_unsafe",
    "fn_unsafe",
    "tp_safe",
    "fp_safe",
    "fn_safe",
    "unsafe_to_safe",
    "unsafe_to_review",
    "unsafe_unmatched_gt",
    "safe_to_unsafe",
    "safe_to_review",
    "safe_unmatched_gt",
    "review_to_safe",
    "review_to_unsafe",
    "review_to_review",
    "matched",
    "matched_decided",
    "pred_decided",
    "unmatched_pred",
    "unmatched_gt",
    "unmatched_pred_decided",
]


def aggregate_counts(counts_list):
    return {key: sum(c.get(key, 0) for c in counts_list) for key in COUNT_KEYS}


def safe_div(num, den):
    return num / den if den else 0.0


def risk_upper_bound(violations, trials, delta=0.10):
    """Exact one-sided Clopper-Pearson upper bound for unsafe->safe risk."""
    if trials <= 0:
        return 1.0
    delta = min(1.0 - 1e-12, max(1e-12, float(delta)))
    violations = max(0, min(int(violations), int(trials)))
    if violations >= trials:
        return 1.0
    return float(beta.ppf(1.0 - delta, violations + 1, trials - violations))


def f1(precision, recall):
    return safe_div(2 * precision * recall, precision + recall)


def metrics_from_counts(counts, bg=None, risk_delta=0.10, risk_alpha=None):
    gt_determined = counts["gt_safe"] + counts["gt_unsafe"]
    gt_total = gt_determined + counts["gt_review"]
    pred_determined = counts["pred_safe"] + counts["pred_unsafe"]
    pred_total = pred_determined + counts["pred_review"]
    unsafe_precision = safe_div(counts["tp_unsafe"], counts["pred_unsafe"])
    unsafe_recall = safe_div(counts["tp_unsafe"], counts["gt_unsafe"])
    safe_precision = safe_div(counts["tp_safe"], counts["pred_safe"])
    safe_recall = safe_div(counts["tp_safe"], counts["gt_safe"])
    unsafe_risk = safe_div(counts["unsafe_to_safe"], counts["gt_unsafe"])
    unsafe_risk_ucb = risk_upper_bound(
        counts["unsafe_to_safe"], counts["gt_unsafe"], risk_delta
    )
    out = {
        **counts,
        "gt_determined": gt_determined,
        "gt_total": gt_total,
        "pred_determined": pred_determined,
        "pred_total": pred_total,
        "unsafe_recall": unsafe_recall,
        "unsafe_precision": unsafe_precision,
        "unsafe_f1": f1(unsafe_precision, unsafe_recall),
        "unsafe_miss_rate": safe_div(counts["fn_unsafe"], counts["gt_unsafe"]),
        "unsafe_auto_safe_rate": unsafe_risk,
        "unsafe_auto_safe_rate_ucb": unsafe_risk_ucb,
        "unsafe_risk_delta": risk_delta,
        "unsafe_risk_alpha": risk_alpha if risk_alpha is not None else "",
        "unsafe_risk_constraint_satisfied": (
            int(unsafe_risk_ucb <= risk_alpha) if risk_alpha is not None else ""
        ),
        "unsafe_review_rate": safe_div(counts["unsafe_to_review"], counts["gt_unsafe"]),
        "unsafe_unmatched_gt_rate": safe_div(counts["unsafe_unmatched_gt"], counts["gt_unsafe"]),
        "safe_auto_unsafe_rate": safe_div(counts["safe_to_unsafe"], counts["gt_safe"]),
        "safe_review_rate": safe_div(counts["safe_to_review"], counts["gt_safe"]),
        "safe_unmatched_gt_rate": safe_div(counts["safe_unmatched_gt"], counts["gt_safe"]),
        "safe_accept_unsafe_risk": safe_div(counts["unsafe_to_safe"], counts["pred_safe"]),
        "safe_recall": safe_recall,
        "safe_precision": safe_precision,
        "safe_f1": f1(safe_precision, safe_recall),
        "coverage": safe_div(counts["matched_decided"], counts["matched"]),
        "prediction_decision_rate": safe_div(counts["pred_decided"], pred_total),
        "review_rate": safe_div(counts["pred_review"], pred_total),
        "gt_determined_rate": safe_div(gt_determined, gt_total),
        "unsafe_fp_per_100_matched": safe_div(counts["fp_unsafe"], counts["matched"]) * 100,
        "safe_fp_per_100_matched": safe_div(counts["fp_safe"], counts["matched"]) * 100,
        "unmatched_pred_decision_rate": safe_div(counts["unmatched_pred_decided"], counts["unmatched_pred"]),
    }
    if bg:
        out.update(bg)
    return out


def summarize(rows):
    out = []
    keys = sorted({(r["rate"], r["region"], r["method"]) for r in rows if "unsafe_recall" in r})
    for rate, region, method in keys:
        group = [r for r in rows if (r["rate"], r["region"], r["method"]) == (rate, region, method) and "unsafe_recall" in r]

        def mean(key):
            vals = [float(r.get(key, 0.0) or 0.0) for r in group]
            return statistics.mean(vals) if vals else 0.0

        def sd(key):
            vals = [float(r.get(key, 0.0) or 0.0) for r in group]
            return statistics.stdev(vals) if len(vals) > 1 else 0.0

        out.append(
            {
                "rate": rate,
                "region": region,
                "method": method,
                "splits": len(group),
                "threshold_mean": mean("threshold"),
                "unsafe_recall_mean": mean("unsafe_recall"),
                "unsafe_recall_sd": sd("unsafe_recall"),
                "unsafe_precision_mean": mean("unsafe_precision"),
                "unsafe_precision_sd": sd("unsafe_precision"),
                "unsafe_f1_mean": mean("unsafe_f1"),
                "unsafe_miss_rate_mean": mean("unsafe_miss_rate"),
                "unsafe_auto_safe_rate_mean": mean("unsafe_auto_safe_rate"),
                "unsafe_auto_safe_rate_sd": sd("unsafe_auto_safe_rate"),
                "unsafe_auto_safe_rate_ucb_mean": mean("unsafe_auto_safe_rate_ucb"),
                "unsafe_risk_constraint_satisfied_mean": mean("unsafe_risk_constraint_satisfied"),
                "unsafe_review_rate_mean": mean("unsafe_review_rate"),
                "unsafe_unmatched_gt_rate_mean": mean("unsafe_unmatched_gt_rate"),
                "safe_auto_unsafe_rate_mean": mean("safe_auto_unsafe_rate"),
                "safe_review_rate_mean": mean("safe_review_rate"),
                "safe_accept_unsafe_risk_mean": mean("safe_accept_unsafe_risk"),
                "safe_accept_unsafe_risk_sd": sd("safe_accept_unsafe_risk"),
                "safe_recall_mean": mean("safe_recall"),
                "safe_precision_mean": mean("safe_precision"),
                "coverage_mean": mean("coverage"),
                "review_rate_mean": mean("review_rate"),
                "prediction_decision_rate_mean": mean("prediction_decision_rate"),
                "bg_any_state_fp_rate_mean": mean("bg_any_state_fp_rate"),
                "bg_any_state_fp_rate_sd": sd("bg_any_state_fp_rate"),
                "bg_unsafe_fp_rate_mean": mean("bg_unsafe_fp_rate"),
                "bg_unsafe_fp_rate_sd": sd("bg_unsafe_fp_rate"),
                "bg_review_rate_mean": mean("bg_review_rate"),
                "gt_determined_rate_mean": mean("gt_determined_rate"),
                "unmatched_pred_decision_rate_mean": mean("unmatched_pred_decision_rate"),
            }
        )
    return out


def summarize_threshold_sweep(rows):
    out = []
    keys = sorted(
        {(r["rate"], r["region"], r["method"], r["threshold"]) for r in rows if "unsafe_recall" in r},
        key=lambda x: (int(x[0]), x[1], x[2], float(x[3])),
    )
    for rate, region, method, threshold in keys:
        group = [
            r
            for r in rows
            if (r["rate"], r["region"], r["method"], r["threshold"]) == (rate, region, method, threshold)
            and "unsafe_recall" in r
        ]

        def mean(key):
            vals = [float(r.get(key, 0.0) or 0.0) for r in group]
            return statistics.mean(vals) if vals else 0.0

        def sd(key):
            vals = [float(r.get(key, 0.0) or 0.0) for r in group]
            return statistics.stdev(vals) if len(vals) > 1 else 0.0

        out.append(
            {
                "rate": rate,
                "region": region,
                "method": method,
                "threshold": threshold,
                "splits": len(group),
                "unsafe_auto_safe_rate_mean": mean("unsafe_auto_safe_rate"),
                "unsafe_auto_safe_rate_sd": sd("unsafe_auto_safe_rate"),
                "unsafe_auto_safe_rate_ucb_mean": mean("unsafe_auto_safe_rate_ucb"),
                "unsafe_risk_constraint_satisfied_mean": mean("unsafe_risk_constraint_satisfied"),
                "unsafe_review_rate_mean": mean("unsafe_review_rate"),
                "unsafe_unmatched_gt_rate_mean": mean("unsafe_unmatched_gt_rate"),
                "safe_accept_unsafe_risk_mean": mean("safe_accept_unsafe_risk"),
                "safe_accept_unsafe_risk_sd": sd("safe_accept_unsafe_risk"),
                "unsafe_recall_mean": mean("unsafe_recall"),
                "unsafe_precision_mean": mean("unsafe_precision"),
                "unsafe_f1_mean": mean("unsafe_f1"),
                "coverage_mean": mean("coverage"),
                "review_rate_mean": mean("review_rate"),
                "prediction_decision_rate_mean": mean("prediction_decision_rate"),
                "bg_any_state_fp_rate_mean": mean("bg_any_state_fp_rate"),
                "bg_unsafe_fp_rate_mean": mean("bg_unsafe_fp_rate"),
                "bg_review_rate_mean": mean("bg_review_rate"),
            }
        )
    return out


def select_threshold(cal_rows, risk_alpha):
    candidates = []
    for row in sorted(cal_rows, key=lambda item: float(item["threshold"]), reverse=True):
        if float(row["unsafe_auto_safe_rate_ucb"]) > float(risk_alpha):
            break
        candidates.append(row)
    if not candidates:
        return None
    # The retained prefix is monotone. Coverage is the primary criterion and
    # threshold is a stable secondary tie-breaker for reproducibility.
    selected = dict(
        max(candidates, key=lambda row: (float(row["coverage"]), -float(row["threshold"])))
    )
    selected["selection_rule"] = "highest_calibration_coverage_subject_to_unsafe_to_safe_ucb"
    selected["risk_alpha"] = float(risk_alpha)
    return selected


def eval_background(bg_predictions, thr, policy, pred_assignment="worker"):
    bg_state_fp = 0
    bg_unsafe_fp = 0
    bg_review = 0
    bg_n = 0
    for preds in bg_predictions.values():
        counts = evaluate_image(preds, [], thr, policy=policy, pred_assignment=pred_assignment)
        bg_n += 1
        if counts["pred_safe"] + counts["pred_unsafe"] > 0:
            bg_state_fp += 1
        if counts["pred_unsafe"] > 0:
            bg_unsafe_fp += 1
        if counts["pred_review"] > 0:
            bg_review += 1
    return {
        "bg_any_state_fp_rate": bg_state_fp / max(1, bg_n),
        "bg_unsafe_fp_rate": bg_unsafe_fp / max(1, bg_n),
        "bg_review_rate": bg_review / max(1, bg_n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=Path(r"D:\ppe_pilot"))
    parser.add_argument("--runs-root", type=Path, default=Path(r"D:\ppe_pilot\runs"))
    parser.add_argument("--out", type=Path, default=Path(r"D:\ppe_pilot\person_conditioned_gate\worker_state_formal_v1"))
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--raw-thr", type=float, default=0.25)
    parser.add_argument(
        "--risk-alpha",
        type=float,
        default=0.30,
        help="Maximum allowed unsafe->safe conditional risk.",
    )
    parser.add_argument(
        "--risk-delta",
        type=float,
        default=0.10,
        help="Failure probability for the one-sided Clopper-Pearson risk bound.",
    )
    parser.add_argument("--crop-size", type=int, default=640)
    args = parser.parse_args()
    if not 0.0 < args.risk_alpha < 1.0:
        parser.error("--risk-alpha must be in (0, 1)")
    if not 0.0 < args.risk_delta < 1.0:
        parser.error("--risk-delta must be in (0, 1)")
    selection_delta = args.risk_delta / max(1, len(args.thresholds))

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    eval_sweep_rows = []
    calibration_rows = []
    failures = []

    for rate in args.rates:
        for seed in args.seeds:
            for fold in args.folds:
                tag = f"r{rate}_s{seed}_f{fold}"
                yolo_root = args.split_root / f"target_{rate}pct" / f"seed{seed}" / f"fold{fold}"
                weights = args.runs_root / f"FT_r{rate}_s{seed}_f{fold}" / "weights" / "best.pt"
                if not weights.exists():
                    failures.append({"tag": tag, "reason": f"missing weights: {weights}"})
                    continue
                print(f"[start] {tag}", flush=True)

                gt_by_image = collect_val_gt(yolo_root, "val")
                val_paths = sorted(gt_by_image)
                cal_paths = [p for p in val_paths if stable_half(p) == 0]
                eval_paths = [p for p in val_paths if stable_half(p) == 1]
                bg_paths = ensure_background_crops(
                    yolo_root,
                    args.out / "strict_crops" / tag,
                    "val",
                    args.crop_size,
                    20260731 + rate * 100 + seed * 10 + fold,
                )
                cal_bg_paths = [p for p in bg_paths if stable_half(p) == 0]
                eval_bg_paths = [p for p in bg_paths if stable_half(p) == 1]

                model = YOLO(str(weights))
                all_val_pred = predict_paths(model, val_paths, args.imgsz, 0.05, args.device)
                all_bg_pred = predict_paths(model, bg_paths, args.imgsz, 0.05, args.device)

                cal_gt = subset_dict(gt_by_image, cal_paths)
                eval_gt = subset_dict(gt_by_image, eval_paths)
                cal_pred = subset_dict(all_val_pred, cal_paths)
                eval_pred = subset_dict(all_val_pred, eval_paths)
                cal_bg_pred = subset_dict(all_bg_pred, cal_bg_paths)
                eval_bg_pred = subset_dict(all_bg_pred, eval_bg_paths)

                ref_counts = [evaluate_image(cal_pred[p], cal_gt[p], args.raw_thr, policy="separate") for p in cal_paths]
                raw_ref_unsafe_recall = sum(c["tp_unsafe"] for c in ref_counts) / max(1, sum(c["gt_unsafe"] for c in ref_counts))

                cal_rows = []
                for thr in args.thresholds:
                    cal_counts = [evaluate_image(cal_pred[p], cal_gt[p], thr, policy="separate") for p in cal_paths]
                    counts = aggregate_counts(cal_counts)
                    bg = eval_background(cal_bg_pred, thr, "separate")
                    cal_row = {
                        "rate": rate,
                        "seed": seed,
                        "fold": fold,
                        "tag": tag,
                        "split": "calibration",
                        "method": "state_threshold_sweep",
                        "threshold": thr,
                        "raw_ref_unsafe_recall_at_025": raw_ref_unsafe_recall,
                        "risk_familywise_delta": args.risk_delta,
                        "risk_per_threshold_delta": selection_delta,
                        **metrics_from_counts(
                            counts,
                            bg,
                            selection_delta,
                            args.risk_alpha,
                        ),
                    }
                    cal_rows.append(cal_row)
                    calibration_rows.append(cal_row)

                selected = select_threshold(cal_rows, args.risk_alpha)
                if selected is None:
                    failures.append(
                        {
                            "tag": tag,
                            "reason": "no threshold satisfies unsafe-to-safe risk bound",
                            "risk_alpha": args.risk_alpha,
                        }
                    )
                    continue

                regions = {
                    "all": (eval_paths, eval_bg_paths),
                    "darkest25": (darkest_subset(eval_paths), darkest_subset(eval_bg_paths)),
                }
                for region, (region_eval_paths, region_bg_paths) in regions.items():
                    region_gt = subset_dict(eval_gt, region_eval_paths)
                    region_pred = subset_dict(eval_pred, region_eval_paths)
                    region_bg_pred = subset_dict(eval_bg_pred, region_bg_paths)

                    for thr in args.thresholds:
                        sweep_counts = [
                            evaluate_image(region_pred[p], region_gt[p], thr, policy="separate", pred_assignment="worker")
                            for p in region_eval_paths
                        ]
                        sweep_bg = eval_background(region_bg_pred, thr, "separate", "worker")
                        eval_sweep_rows.append(
                            {
                                "rate": rate,
                                "seed": seed,
                                "fold": fold,
                                "tag": tag,
                                "region": region,
                                "split": "eval",
                                "method": "state_eval_threshold_sweep",
                                "policy": "separate",
                                "pred_assignment": "worker",
                                "threshold": thr,
                                **metrics_from_counts(
                                    aggregate_counts(sweep_counts),
                                    sweep_bg,
                                    args.risk_delta,
                                    args.risk_alpha,
                                ),
                            }
                        )

                    for variant in STATE_VARIANTS:
                        counts = [
                            evaluate_image(
                                region_pred[p],
                                region_gt[p],
                                selected["threshold"],
                                policy=variant["policy"],
                                pred_assignment=variant["pred_assignment"],
                            )
                            for p in region_eval_paths
                        ]
                        bg = eval_background(
                            region_bg_pred,
                            selected["threshold"],
                            variant["policy"],
                            variant["pred_assignment"],
                        )
                        rows.append(
                            {
                                "rate": rate,
                                "seed": seed,
                                "fold": fold,
                                "tag": tag,
                                "region": region,
                                "method": variant["method"],
                                "policy": variant["policy"],
                                "pred_assignment": variant["pred_assignment"],
                                "threshold": selected["threshold"],
                                "raw_ref_unsafe_recall_at_025": raw_ref_unsafe_recall,
                                "cal_selected_threshold": selected["threshold"],
                                "risk_alpha": args.risk_alpha,
                                "risk_delta": args.risk_delta,
                                "cal_risk_bound_delta": selection_delta,
                                "cal_unsafe_auto_safe_rate": selected["unsafe_auto_safe_rate"],
                                "cal_unsafe_auto_safe_rate_ucb": selected["unsafe_auto_safe_rate_ucb"],
                                "cal_coverage": selected["coverage"],
                                "selection_rule": selected["selection_rule"],
                                **metrics_from_counts(
                                    aggregate_counts(counts),
                                    bg,
                                    args.risk_delta,
                                    args.risk_alpha,
                                ),
                            }
                        )

                    # Raw threshold reference at 0.25 under separate policy.
                    raw_counts = [evaluate_image(region_pred[p], region_gt[p], args.raw_thr, policy="separate") for p in region_eval_paths]
                    raw_bg = eval_background(region_bg_pred, args.raw_thr, "separate")
                    rows.append(
                        {
                            "rate": rate,
                            "seed": seed,
                            "fold": fold,
                            "tag": tag,
                            "region": region,
                            "method": "state_raw_025",
                            "threshold": args.raw_thr,
                            "raw_ref_unsafe_recall_at_025": raw_ref_unsafe_recall,
                            **metrics_from_counts(
                                aggregate_counts(raw_counts),
                                raw_bg,
                                args.risk_delta,
                                args.risk_alpha,
                            ),
                        }
                    )

                    # External object-level reference for continuity.
                    gate_val = eval_val_at_threshold(region_pred, region_gt, args.raw_thr, "pred_person", PERSON_SCALE, MIN_INTER_OVER_SAFETY)
                    gate_bg = eval_bg_at_threshold(region_bg_pred, args.raw_thr, "pred_person", PERSON_SCALE, MIN_INTER_OVER_SAFETY)
                    rows.append(
                        {
                            "rate": rate,
                            "seed": seed,
                            "fold": fold,
                            "tag": tag,
                            "region": region,
                            "method": "worker_box_gate_025",
                            "threshold": args.raw_thr,
                            "val_recall": gate_val["recall"],
                            "val_precision": gate_val["precision"],
                            "bg_safety_fp_rate": gate_bg["image_safety_fp_rate"],
                            "bg_safety_prediction_count": gate_bg["safety_prediction_count"],
                        }
                    )

                write_csv(args.out / "worker_state_formal_rows.csv", rows)
                write_csv(args.out / "worker_state_eval_sweep_rows.csv", eval_sweep_rows)
                write_csv(args.out / "worker_state_eval_sweep_summary.csv", summarize_threshold_sweep(eval_sweep_rows))
                write_csv(args.out / "worker_state_calibration_rows.csv", calibration_rows)
                write_csv(args.out / "worker_state_formal_summary.csv", summarize(rows))

    write_csv(args.out / "worker_state_formal_rows.csv", rows)
    write_csv(args.out / "worker_state_eval_sweep_rows.csv", eval_sweep_rows)
    write_csv(args.out / "worker_state_eval_sweep_summary.csv", summarize_threshold_sweep(eval_sweep_rows))
    write_csv(args.out / "worker_state_calibration_rows.csv", calibration_rows)
    write_csv(args.out / "worker_state_formal_summary.csv", summarize(rows))
    (args.out / "worker_state_formal_summary.json").write_text(
        json.dumps(
            {"rows": rows, "eval_sweep_rows": eval_sweep_rows, "calibration_rows": calibration_rows, "failures": failures},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] rows={len(rows)} failures={len(failures)} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
