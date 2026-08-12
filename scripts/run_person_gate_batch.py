import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from evaluate_person_conditioned_gate import (
    DEFAULT_CLASS_NAMES,
    filter_predictions,
    imread,
    label_path_for,
    match_class,
    predict_image,
    read_yolo_boxes,
)


CLASS_NAMES = DEFAULT_CLASS_NAMES
PERSON_CLASS = 3
SAFETY_CLASSES = {0, 1, 2, 4}


def imwrite(path, image):
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    buf.tofile(str(path))


def expand(box, w, h, scale):
    _, x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    return (
        max(0, int(round(cx - bw / 2))),
        max(0, int(round(cy - bh / 2))),
        min(w - 1, int(round(cx + bw / 2))),
        min(h - 1, int(round(cy + bh / 2))),
    )


def clean_crop(image, boxes, crop_size, rng, scale=1.35, tries=80):
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return None
    occupied = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        ex1, ey1, ex2, ey2 = expand(box, w, h, scale)
        occupied[ey1:ey2, ex1:ex2] = 1
    xs = [0, max(0, (w - crop_size) // 2), w - crop_size]
    ys = [0, max(0, (h - crop_size) // 2), h - crop_size]
    candidates = [(x, y) for y in ys for x in xs]
    candidates += [(rng.randint(0, w - crop_size), rng.randint(0, h - crop_size)) for _ in range(tries)]
    for x, y in candidates:
        if not occupied[y:y + crop_size, x:x + crop_size].any():
            return image[y:y + crop_size, x:x + crop_size].copy()
    return None


def yolo_boxes_int(label_path, w, h):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, bw, bh = map(float, parts[1:5])
        x1 = int(round((x - bw / 2) * w))
        y1 = int(round((y - bh / 2) * h))
        x2 = int(round((x + bw / 2) * w))
        y2 = int(round((y + bh / 2) * h))
        boxes.append((cls, max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
    return boxes


def ensure_background_crops(yolo_root, out_dir, split, crop_size, seed):
    bg_dir = out_dir / "background_crops"
    existing = sorted(bg_dir.glob("*.jpg")) if bg_dir.exists() else []
    if existing:
        return existing
    bg_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    made = []
    img_dir = yolo_root / "images" / split
    lab_dir = yolo_root / "labels" / split
    for image_path in sorted(img_dir.glob("*.jpg")):
        image = imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes = yolo_boxes_int(lab_dir / f"{image_path.stem}.txt", w, h)
        if not boxes:
            continue
        crop = clean_crop(image, boxes, crop_size, rng)
        if crop is None:
            continue
        out_path = bg_dir / f"{len(made):04d}_{image_path.name}"
        imwrite(out_path, crop)
        made.append(out_path)
    return made


def threshold_preds(preds, threshold):
    return [p for p in preds if p["conf"] >= threshold]


def eval_val_at_threshold(predictions, gt_by_image, threshold, mode, person_scale, min_inter_over_safety):
    totals = {CLASS_NAMES[c]: {"tp": 0, "fp": 0, "fn": 0} for c in sorted(SAFETY_CLASSES)}
    person_instances = 0
    for image_path, preds in predictions.items():
        gts = gt_by_image[image_path]
        person_instances += sum(1 for g in gts if g["cls"] == PERSON_CLASS)
        filtered = filter_predictions(
            threshold_preds(preds, threshold),
            mode,
            gts,
            PERSON_CLASS,
            SAFETY_CLASSES,
            person_scale,
            min_inter_over_safety,
        )
        for cls in sorted(SAFETY_CLASSES):
            tp, fp, fn = match_class(filtered, gts, cls, iou_thr=0.5)
            row = totals[CLASS_NAMES[cls]]
            row["tp"] += tp
            row["fp"] += fp
            row["fn"] += fn
    tp = sum(v["tp"] for v in totals.values())
    fp = sum(v["fp"] for v in totals.values())
    fn = sum(v["fn"] for v in totals.values())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "fp_per_1000_person": fp / max(1, person_instances) * 1000,
        "person_instances": person_instances,
        "class_totals": totals,
    }


def eval_bg_at_threshold(predictions, threshold, mode, person_scale, min_inter_over_safety):
    image_safety_fp = 0
    image_person_fp = 0
    safety_prediction_count = 0
    counts = {name: 0 for name in CLASS_NAMES}
    for preds in predictions.values():
        filtered = filter_predictions(
            threshold_preds(preds, threshold),
            mode,
            [],
            PERSON_CLASS,
            SAFETY_CLASSES,
            person_scale,
            min_inter_over_safety,
        )
        has_safety = False
        has_person = False
        for pred in filtered:
            cls = pred["cls"]
            counts[CLASS_NAMES[cls]] += 1
            if cls == PERSON_CLASS:
                has_person = True
            if cls in SAFETY_CLASSES:
                has_safety = True
                safety_prediction_count += 1
        image_safety_fp += int(has_safety)
        image_person_fp += int(has_person)
    n = max(1, len(predictions))
    return {
        "images": len(predictions),
        "image_safety_fp_rate": image_safety_fp / n,
        "image_person_fp_rate": image_person_fp / n,
        "safety_prediction_count": safety_prediction_count,
        "class_counts": counts,
    }


def collect_val_gt(yolo_root, split):
    gt_by_image = {}
    for image_path in sorted((yolo_root / "images" / split).glob("*.jpg")):
        image = imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        gt_by_image[image_path] = read_yolo_boxes(label_path_for(image_path, yolo_root, split), w, h)
    return gt_by_image


def predict_paths(model, paths, imgsz, conf, device):
    predictions = {}
    for path in paths:
        image = imread(path)
        if image is None:
            continue
        predictions[path] = predict_image(model, image, imgsz, conf, device)
    return predictions


def mean_sd(values):
    if not values:
        return {"mean": None, "sd": None}
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(rows):
    summary = []
    keys = sorted({(r["rate"], r["method"], r["threshold"], r["mode"]) for r in rows})
    for rate, method, threshold, mode in keys:
        group = [r for r in rows if (r["rate"], r["method"], r["threshold"], r["mode"]) == (rate, method, threshold, mode)]
        rec = mean_sd([r["val_recall"] for r in group])
        prec = mean_sd([r["val_precision"] for r in group])
        bg = mean_sd([r["bg_safety_fp_rate"] for r in group])
        summary.append({
            "rate": rate,
            "method": method,
            "threshold": threshold,
            "mode": mode,
            "splits": len(group),
            "val_recall_mean": rec["mean"],
            "val_recall_sd": rec["sd"],
            "val_precision_mean": prec["mean"],
            "val_precision_sd": prec["sd"],
            "bg_safety_fp_rate_mean": bg["mean"],
            "bg_safety_fp_rate_sd": bg["sd"],
        })
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bgneg_name(rate, seed, fold):
    if rate == 5:
        return f"BGNegFT_r5_s{seed}_f{fold}_6ep"
    if rate == 10:
        return f"BGNegFT_r10_s{seed}_f{fold}_6ep_ampoff"
    raise ValueError(rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=Path(r"D:\ppe_pilot"))
    parser.add_argument("--runs-root", type=Path, default=Path(r"D:\ppe_pilot\runs"))
    parser.add_argument("--out", type=Path, default=Path(r"D:\ppe_pilot\person_conditioned_gate\batch_v1"))
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--person-scale", type=float, default=1.15)
    parser.add_argument("--min-inter-over-safety", type=float, default=0.15)
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    cell_results = []
    min_conf = min(args.thresholds)
    total_cells = len(args.rates) * len(args.seeds) * len(args.folds)
    done = 0
    for rate in args.rates:
        for seed in args.seeds:
            for fold in args.folds:
                done += 1
                tag = f"r{rate}_s{seed}_f{fold}"
                yolo_root = args.split_root / f"target_{rate}pct" / f"seed{seed}" / f"fold{fold}"
                bg_dir = args.out / "strict_crops" / tag
                bg_paths = ensure_background_crops(yolo_root, bg_dir, args.split, args.crop_size, 20260731 + rate * 100 + seed * 10 + fold)
                gt_by_image = collect_val_gt(yolo_root, args.split)
                val_paths = list(gt_by_image)
                methods = {
                    "base": args.runs_root / f"FT_r{rate}_s{seed}_f{fold}" / "weights" / "best.pt",
                    "bgneg": args.runs_root / bgneg_name(rate, seed, fold) / "weights" / "best.pt",
                }
                print(f"[{done}/{total_cells}] {tag}: val={len(val_paths)} bg={len(bg_paths)}", flush=True)
                cell = {"tag": tag, "rate": rate, "seed": seed, "fold": fold, "methods": {}}
                for method, weights in methods.items():
                    if not weights.is_file():
                        print(f"[missing] {method} {weights}", flush=True)
                        continue
                    model = YOLO(str(weights))
                    val_pred = predict_paths(model, val_paths, args.imgsz, min_conf, args.device)
                    bg_pred = predict_paths(model, bg_paths, args.imgsz, min_conf, args.device)
                    cell["methods"][method] = {"weights": str(weights), "thresholds": {}}
                    for threshold in args.thresholds:
                        for mode in ["raw", "pred_person", "gt_person"]:
                            val_metrics = eval_val_at_threshold(
                                val_pred, gt_by_image, threshold, mode, args.person_scale, args.min_inter_over_safety
                            )
                            bg_metrics = eval_bg_at_threshold(
                                bg_pred, threshold, mode, args.person_scale, args.min_inter_over_safety
                            )
                            row = {
                                "rate": rate,
                                "seed": seed,
                                "fold": fold,
                                "tag": tag,
                                "method": method,
                                "threshold": threshold,
                                "mode": mode,
                                "val_precision": val_metrics["precision"],
                                "val_recall": val_metrics["recall"],
                                "val_fp_per_1000_person": val_metrics["fp_per_1000_person"],
                                "bg_safety_fp_rate": bg_metrics["image_safety_fp_rate"],
                                "bg_safety_prediction_count": bg_metrics["safety_prediction_count"],
                                "bg_person_fp_rate": bg_metrics["image_person_fp_rate"],
                                "bg_images": bg_metrics["images"],
                            }
                            rows.append(row)
                            cell["methods"][method]["thresholds"][f"{threshold}_{mode}"] = {
                                "val": val_metrics,
                                "background": bg_metrics,
                            }
                cell_results.append(cell)
                write_csv(args.out / "person_gate_rows.csv", rows)
                write_csv(args.out / "person_gate_summary.csv", summarize(rows))
                (args.out / "person_gate_cells.json").write_text(
                    json.dumps(cell_results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    summary = summarize(rows)
    write_csv(args.out / "person_gate_rows.csv", rows)
    write_csv(args.out / "person_gate_summary.csv", summary)
    (args.out / "person_gate_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rows={len(rows)} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
