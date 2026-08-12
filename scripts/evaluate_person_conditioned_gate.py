import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASS_NAMES = ["helmet", "no_helmet", "no_reflective_vest", "person", "reflective_vest"]


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_yolo_boxes(label_path, w, h):
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, bw, bh = map(float, parts[1:5])
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        rows.append({"cls": cls, "conf": 1.0, "xyxy": [x1, y1, x2, y2]})
    return rows


def label_path_for(image_path, yolo_root, split):
    return yolo_root / "labels" / split / f"{image_path.stem}.txt"


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_box(box, scale):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def center_inside(inner, outer):
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def has_person_support(safety_box, person_boxes, person_scale, min_inter_over_safety):
    safety_area = box_area(safety_box)
    if safety_area <= 0:
        return False
    for person_box in person_boxes:
        support_box = expand_box(person_box, person_scale)
        if center_inside(safety_box, support_box):
            return True
        if intersection(safety_box, support_box) / safety_area >= min_inter_over_safety:
            return True
    return False


def filter_predictions(preds, mode, gt_persons, person_class, safety_class_ids, person_scale, min_inter_over_safety):
    if mode == "raw":
        return preds
    if mode == "pred_person":
        persons = [p["xyxy"] for p in preds if p["cls"] == person_class]
    elif mode == "gt_person":
        persons = [p["xyxy"] for p in gt_persons if p["cls"] == person_class]
    else:
        raise ValueError(mode)

    kept = []
    for pred in preds:
        if pred["cls"] == person_class:
            kept.append(pred)
        elif pred["cls"] in safety_class_ids:
            if has_person_support(pred["xyxy"], persons, person_scale, min_inter_over_safety):
                kept.append(pred)
        else:
            kept.append(pred)
    return kept


def predict_image(model, image, imgsz, conf, device):
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        device=device,
        verbose=False,
        stream=False,
    )[0]
    rows = []
    if result.boxes is None or len(result.boxes) == 0:
        return rows
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for box, c, s in zip(xyxy, cls, confs):
        rows.append({"cls": int(c), "conf": float(s), "xyxy": [float(v) for v in box]})
    return rows


def match_class(preds, gts, cls, iou_thr):
    gt_cls = [g["xyxy"] for g in gts if g["cls"] == cls]
    pred_cls = sorted([p["xyxy"] for p in preds if p["cls"] == cls], key=box_area, reverse=True)
    used = set()
    tp = 0
    for pred in pred_cls:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_cls):
            if idx in used:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= iou_thr:
            used.add(best_idx)
            tp += 1
    fp = len(pred_cls) - tp
    fn = len(gt_cls) - tp
    return tp, fp, fn


def evaluate_val(
    model,
    yolo_root,
    split,
    imgsz,
    conf,
    device,
    modes,
    class_names,
    person_class,
    safety_class_ids,
    person_scale,
    min_inter_over_safety,
):
    image_dir = yolo_root / "images" / split
    totals = {
        mode: {
            "classes": {
                class_names[c]: {"tp": 0, "fp": 0, "fn": 0}
                for c in sorted(safety_class_ids)
            },
            "person_gt_instances": 0,
            "images": 0,
        }
        for mode in modes
    }
    for image_path in sorted(image_dir.glob("*.jpg")):
        image = imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        gts = read_yolo_boxes(label_path_for(image_path, yolo_root, split), w, h)
        preds = predict_image(model, image, imgsz, conf, device)
        for mode in modes:
            filtered = filter_predictions(
                preds, mode, gts, person_class, safety_class_ids, person_scale, min_inter_over_safety
            )
            totals[mode]["images"] += 1
            totals[mode]["person_gt_instances"] += sum(1 for g in gts if g["cls"] == person_class)
            for cls in sorted(safety_class_ids):
                tp, fp, fn = match_class(filtered, gts, cls, iou_thr=0.5)
                item = totals[mode]["classes"][class_names[cls]]
                item["tp"] += tp
                item["fp"] += fp
                item["fn"] += fn
    for mode in modes:
        safety_tp = safety_fp = safety_fn = 0
        for item in totals[mode]["classes"].values():
            item["precision"] = item["tp"] / max(1, item["tp"] + item["fp"])
            item["recall"] = item["tp"] / max(1, item["tp"] + item["fn"])
            safety_tp += item["tp"]
            safety_fp += item["fp"]
            safety_fn += item["fn"]
        person_n = totals[mode]["person_gt_instances"]
        totals[mode]["safety_macro"] = {
            "tp": safety_tp,
            "fp": safety_fp,
            "fn": safety_fn,
            "precision": safety_tp / max(1, safety_tp + safety_fp),
            "recall": safety_tp / max(1, safety_tp + safety_fn),
            "fp_per_1000_person": safety_fp / max(1, person_n) * 1000,
        }
    return totals


def evaluate_background(
    model,
    bg_dir,
    imgsz,
    conf,
    device,
    modes,
    class_names,
    person_class,
    safety_class_ids,
    person_scale,
    min_inter_over_safety,
):
    result = {
        mode: {
            "images": 0,
            "image_safety_fp": 0,
            "image_person_fp": 0,
            "safety_prediction_count": 0,
            "class_counts": {name: 0 for name in class_names},
        }
        for mode in modes
    }
    for image_path in sorted(bg_dir.glob("*.jpg")):
        image = imread(image_path)
        if image is None:
            continue
        preds = predict_image(model, image, imgsz, conf, device)
        for mode in modes:
            filtered = filter_predictions(
                preds, mode, [], person_class, safety_class_ids, person_scale, min_inter_over_safety
            )
            result[mode]["images"] += 1
            has_safety = False
            has_person = False
            for pred in filtered:
                cls = pred["cls"]
                if cls < len(class_names):
                    result[mode]["class_counts"][class_names[cls]] += 1
                if cls == person_class:
                    has_person = True
                if cls in safety_class_ids:
                    has_safety = True
                    result[mode]["safety_prediction_count"] += 1
            result[mode]["image_safety_fp"] += int(has_safety)
            result[mode]["image_person_fp"] += int(has_person)
    for mode in modes:
        n = max(1, result[mode]["images"])
        result[mode]["image_safety_fp_rate"] = result[mode]["image_safety_fp"] / n
        result[mode]["image_person_fp_rate"] = result[mode]["image_person_fp"] / n
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--yolo-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--background-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--person-scale", type=float, default=1.15)
    parser.add_argument("--min-inter-over-safety", type=float, default=0.15)
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--person-class", type=int, default=3)
    parser.add_argument("--safety-classes", type=int, nargs="+", default=[0, 1, 2, 4])
    args = parser.parse_args()

    from ultralytics import YOLO

    modes = ["raw", "pred_person", "gt_person"]
    model = YOLO(args.weights)
    out = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": args.weights,
        "yolo_root": args.yolo_root,
        "split": args.split,
        "background_dir": args.background_dir,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "class_names": args.class_names,
        "person_class": args.person_class,
        "safety_classes": args.safety_classes,
        "person_scale": args.person_scale,
        "min_inter_over_safety": args.min_inter_over_safety,
        "val": evaluate_val(
            model,
            Path(args.yolo_root),
            args.split,
            args.imgsz,
            args.conf,
            args.device,
            modes,
            args.class_names,
            args.person_class,
            set(args.safety_classes),
            args.person_scale,
            args.min_inter_over_safety,
        ),
        "background": evaluate_background(
            model,
            Path(args.background_dir),
            args.imgsz,
            args.conf,
            args.device,
            modes,
            args.class_names,
            args.person_class,
            set(args.safety_classes),
            args.person_scale,
            args.min_inter_over_safety,
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "val_safety_macro": {
            mode: out["val"][mode]["safety_macro"]
            for mode in modes
        },
        "background": {
            mode: {
                "image_safety_fp_rate": out["background"][mode]["image_safety_fp_rate"],
                "safety_prediction_count": out["background"][mode]["safety_prediction_count"],
                "image_person_fp_rate": out["background"][mode]["image_person_fp_rate"],
            }
            for mode in modes
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
