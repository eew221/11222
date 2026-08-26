"""Create a frozen detector-output PPE-owner audit package.

Unlike the annotation-box ownership audits, this protocol renders the detector
person and PPE boxes that actually enter the allocation rule.  Human reviewers
can identify an owner among detected persons, flag an owner outside the
detected-person set, flag a false PPE detection, or retain ambiguity.  Ground-
truth PPE boxes unmatched by a prediction are rendered as ``M`` rows solely to
make detector misses visible in the audit; they never become model evidence.

The package is deliberately local because source images and prediction caches
are restricted.  It must be frozen before any human labels or method outcomes
are inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import formal_worker_state_experiment as formal
from build_manual_assignment_audit import CLASS_COLORS, CLASS_LABELS, put_label


FIELDS = (
    "audit_id", "evidence_id", "evidence_class_id", "evidence_class",
    "candidate_person_ids", "all_visible_person_ids", "assigned_person_id",
    "assignment_confidence", "occluded_or_ambiguous", "notes",
)
PPE_CLASSES = {0, 1, 2, 4}
PERSON_CLASS = 3


def read_image(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return image


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def match_ppe(predictions: list[dict], truth: list[dict], iou_threshold: float) -> tuple[dict[int, int], set[int]]:
    """Greedily match same-class predicted PPE boxes to reference PPE boxes."""
    pairs = []
    for pi, prediction in enumerate(predictions):
        for gi, reference in enumerate(truth):
            if prediction["cls"] == reference["cls"]:
                pairs.append((box_iou(prediction["xyxy"], reference["xyxy"]), pi, gi))
    claimed_pred, claimed_truth, matches = set(), set(), {}
    for overlap, pi, gi in sorted(pairs, reverse=True):
        if overlap < iou_threshold or pi in claimed_pred or gi in claimed_truth:
            continue
        claimed_pred.add(pi); claimed_truth.add(gi); matches[pi] = gi
    return matches, set(range(len(truth))) - claimed_truth


def source_stratum(record: dict) -> str:
    person_count = sum(item["cls"] == PERSON_CLASS for item in record["predictions"])
    ppe_count = sum(item["cls"] in PPE_CLASSES for item in record["predictions"])
    people = "single" if person_count <= 1 else "2-3" if person_count <= 3 else "4+"
    evidence = "0-2" if ppe_count <= 2 else "3-6" if ppe_count <= 6 else "7+"
    return f"persons_{people}__predicted_ppe_{evidence}"


def choose_records(records: list[dict], count_per_source: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[record["source_group"]].append(record)
    selected = []
    for source, rows in sorted(by_source.items()):
        by_stratum: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_stratum[source_stratum(row)].append(row)
        for values in by_stratum.values():
            rng.shuffle(values)
        ordered = sorted(by_stratum, key=lambda key: (-len(by_stratum[key]), key))
        take = []
        while ordered and len(take) < min(count_per_source, len(rows)):
            for key in list(ordered):
                if by_stratum[key]:
                    take.append(by_stratum[key].pop())
                    if len(take) == count_per_source:
                        break
                if not by_stratum[key]:
                    ordered.remove(key)
        selected.extend(take)
    return sorted(selected, key=lambda row: (row["source_group"], Path(row["image_path"]).name))


def draw_box(canvas: np.ndarray, box: list[float], label: str, color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    put_label(canvas, label, (x1, y1), color)


def render(image: np.ndarray, people: list[dict], predicted_ppe: list[dict], missed_ppe: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for index, person in enumerate(people, 1):
        draw_box(canvas, person["xyxy"], f"P{index}", (0, 220, 255), 3)
    for index, item in enumerate(predicted_ppe, 1):
        draw_box(canvas, item["xyxy"], f"E{index}:{CLASS_LABELS[item['cls']]}", CLASS_COLORS[item["cls"]])
    for index, item in enumerate(missed_ppe, 1):
        draw_box(canvas, item["xyxy"], f"M{index}:{CLASS_LABELS[item['cls']]}", (150, 150, 150), 1)
    return canvas


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-cache", type=Path, required=True, help="Frozen JSON detector cache for one held-out evaluation cell.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--prediction-floor", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.50)
    args = parser.parse_args()
    cache, out = args.prediction_cache.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit package: {out}")
    payload = json.loads(cache.read_text(encoding="utf-8"))
    raw_records = payload.get("images", [])
    records = []
    for record in raw_records:
        if record.get("role") not in {None, "test"}:
            continue
        if not Path(record["image_path"]).is_file():
            continue
        predictions = [item for item in record.get("predictions", []) if item["conf"] >= args.prediction_floor]
        predicted_ppe = [item for item in predictions if item["cls"] in PPE_CLASSES]
        reference_ppe = [item for item in record.get("ground_truth", []) if item["cls"] in PPE_CLASSES]
        if not any(item["cls"] in PPE_CLASSES | {PERSON_CLASS} for item in predictions):
            continue
        # A person-only image contains no PPE ownership or miss event to audit.
        if not predicted_ppe and not reference_ppe:
            continue
        records.append({**record, "predictions": predictions})
    if not records:
        raise ValueError("no readable held-out records with detector outputs; regenerate or provide a final-test cache")
    selected = choose_records(records, args.count_per_source, args.seed)
    if len({record["source_group"] for record in selected}) < 2:
        raise ValueError("audit requires at least two declared source groups")

    out.mkdir(parents=True)
    images_dir = out / "annotated_images"; images_dir.mkdir()
    image_rows, audit_rows, truth_rows = [], [], []
    for image_index, record in enumerate(selected, 1):
        audit_id = f"D{image_index:03d}"
        people = sorted((item for item in record["predictions"] if item["cls"] == PERSON_CLASS), key=lambda item: (-item["conf"], item["xyxy"]))
        predicted_ppe = sorted((item for item in record["predictions"] if item["cls"] in PPE_CLASSES), key=lambda item: (-item["conf"], item["cls"], item["xyxy"]))
        reference_ppe = [item for item in record.get("ground_truth", []) if item["cls"] in PPE_CLASSES]
        matches, missed = match_ppe(predicted_ppe, reference_ppe, args.match_iou)
        assigned = formal.assign_safety_to_persons(people, predicted_ppe)
        rc_owner = {id(item): f"P{person_index + 1}" for person_index, items in assigned.items() for item in items}
        image = read_image(record["image_path"])
        rendered = render(image, people, predicted_ppe, [reference_ppe[index] for index in sorted(missed)])
        rendered_name = f"{audit_id}_{Path(record['image_path']).stem}.jpg"
        cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(str(images_dir / rendered_name))
        visible = "|".join(f"P{index}" for index in range(1, len(people) + 1))
        image_rows.append({"audit_id": audit_id, "source_group": record["source_group"], "fold": payload.get("tag", "frozen_cache"), "image": rendered_name, "image_name": Path(record["image_path"]).name, "person_count": len(people), "predicted_ppe_count": len(predicted_ppe), "missed_reference_ppe_count": len(missed), "sampling_stratum": source_stratum(record)})
        for index, item in enumerate(predicted_ppe, 1):
            candidate_ids = [f"P{pi + 1}" for pi, person in enumerate(people) if formal.support_score(item["xyxy"], person["xyxy"])[0] > 0 or formal.support_score(item["xyxy"], person["xyxy"])[2] > 0]
            audit_rows.append({"audit_id": audit_id, "evidence_id": f"E{index}", "evidence_class_id": item["cls"], "evidence_class": CLASS_LABELS[item["cls"]], "candidate_person_ids": "|".join(candidate_ids), "all_visible_person_ids": visible, "assigned_person_id": "", "assignment_confidence": "", "occluded_or_ambiguous": "", "notes": ""})
            truth_rows.append({"audit_id": audit_id, "evidence_id": f"E{index}", "row_type": "predicted_ppe", "prediction_confidence": item["conf"], "matched_reference_index": matches.get(index - 1, ""), "detector_status": "matched_reference" if index - 1 in matches else "unmatched_predicted_ppe", "rc_wssi_owner": rc_owner.get(id(item), "NONE"), "global_pool_owner_set": visible or "NONE"})
        for local_index, reference_index in enumerate(sorted(missed), 1):
            item = reference_ppe[reference_index]
            audit_rows.append({"audit_id": audit_id, "evidence_id": f"M{local_index}", "evidence_class_id": item["cls"], "evidence_class": CLASS_LABELS[item["cls"]], "candidate_person_ids": "", "all_visible_person_ids": visible, "assigned_person_id": "", "assignment_confidence": "", "occluded_or_ambiguous": "", "notes": ""})
            truth_rows.append({"audit_id": audit_id, "evidence_id": f"M{local_index}", "row_type": "reference_ppe_missed_by_detector", "prediction_confidence": "", "matched_reference_index": reference_index, "detector_status": "missed_reference_ppe", "rc_wssi_owner": "NONE", "global_pool_owner_set": "NONE"})

    with (out / "audit_image_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(image_rows[0])); writer.writeheader(); writer.writerows(image_rows)
    write_csv(out / "blinded_evidence_assignment_template.csv", audit_rows)
    for annotator in "ABC":
        directory = out / f"annotator_{annotator}"; directory.mkdir()
        write_csv(directory / "evidence_assignment.csv", audit_rows)
    with (out / "sealed_detector_audit_reference.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(truth_rows[0])); writer.writeheader(); writer.writerows(truth_rows)
    instructions = """# 检测输出端到端 PPE 归属审计（A/B/C 独立盲标）

本次审计展示的是**冻结检测器实际输出的人员框 P 和 PPE 框 E**，不是人工标注框。灰色 M 框表示人工标注中存在、但本次检测器没有匹配到的 PPE，用于记录漏检对端到端流程的影响。

每一行请判断它所画出的 PPE 是否真实存在以及属于谁：

1. 对 E（预测 PPE 框），从任意显示的检测人员框 P 中选择实际 owner；若真实 PPE 属于画面中但未被检测为 P 的人员，选择 `OUTSIDE_DETECTED_PERSON_SET`；若 E 是误检或并非真实 PPE，选择 `FALSE_DETECTION`；无法判断选择 `AMBIGUOUS`。
2. 对 M（漏检参考 PPE 框），选择其实际 owner；若实际 owner 未出现在任何 P 框中，选择 `OUTSIDE_DETECTED_PERSON_SET`；无法判断选择 `AMBIGUOUS`。M 不是模型证据，而是漏检核查行。
3. `NONE` 仅用于确实不属于任何可见人员、且不应作为 PPE 证据的异常情形；通常误检请选 `FALSE_DETECTION`。
4. 三位专家必须独立完成并冻结，不得查看检测结果外的算法输出、阈值、其他专家答案或密封参考文件。

本协议评估检测器坐标、误检、重复框、PPE 漏检和人员漏检共同构成的端到端归属边界；它不是部署安全认证。
"""
    (out / "ANNOTATION_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    for annotator in "ABC":
        (out / f"annotator_{annotator}" / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest = {"protocol": "pre-frozen_end_to_end_detector_output_human_owner_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(), "cache_sha256": sha256(cache), "cache_tag": payload.get("tag", "unknown"), "selection_seed": args.seed, "count_per_source": args.count_per_source, "prediction_floor": args.prediction_floor, "same_class_iou_for_miss_rows": args.match_iou, "images": len(image_rows), "declared_source_groups": sorted({row["source_group"] for row in image_rows}), "predicted_ppe_rows": sum(row["evidence_id"].startswith("E") for row in audit_rows), "missed_reference_ppe_rows": sum(row["evidence_id"].startswith("M") for row in audit_rows), "blinding": "A/B/C independent blind passes; no method outputs, thresholds, or other responses are disclosed", "annotation_status": "templates_created_not_yet_human_signed", "scope": "sampled detector-output association and error decomposition; not a full-corpus or deployment guarantee"}
    (out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
