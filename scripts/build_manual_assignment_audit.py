"""Create a blinded visual audit set for worker-PPE association."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import formal_worker_state_experiment as formal
from evaluate_person_conditioned_gate import imread, read_yolo_boxes


CLASS_LABELS = {0: "H+", 1: "H-", 2: "V-", 4: "V+"}
CLASS_COLORS = {
    0: (30, 190, 30),
    1: (30, 30, 230),
    2: (20, 120, 240),
    4: (230, 120, 20),
}
PERSON_COLOR = (0, 220, 255)


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


def load_manifest(cell_root: Path) -> dict[str, dict]:
    with (cell_root / "manifest.csv").open("r", encoding="utf-8", newline="") as stream:
        return {
            row["image_name"]: row
            for row in csv.DictReader(stream)
            if row["role"] == "test"
        }


def load_candidates(protocol_root: Path) -> list[dict]:
    candidates = []
    for fold in range(3):
        cell_root = protocol_root / "rate_10pct" / "seed0" / f"fold{fold}"
        manifest = load_manifest(cell_root)
        for image_path in sorted((cell_root / "images" / "test").glob("*.jpg")):
            image = imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            label_path = cell_root / "labels" / "test" / f"{image_path.stem}.txt"
            boxes = read_yolo_boxes(label_path, width, height)
            persons = [box for box in boxes if box["cls"] == formal.PERSON_CLASS]
            evidence = [box for box in boxes if box["cls"] in formal.SAFETY_CLASSES]
            if len(persons) < 2 or len(persons) > 8 or not evidence:
                continue
            person_heights = [box["xyxy"][3] - box["xyxy"][1] for box in persons]
            person_area_fractions = [
                formal.box_area(box["xyxy"]) / max(1.0, width * height) for box in persons
            ]
            median_person_height = statistics.median(person_heights)
            median_person_area_fraction = statistics.median(person_area_fractions)
            if median_person_height < 35:
                continue
            overlap = 0.0
            for left in range(len(persons)):
                for right in range(left + 1, len(persons)):
                    overlap = max(overlap, formal.iou(persons[left]["xyxy"], persons[right]["xyxy"]))
            ambiguous = 0
            for item in evidence:
                support_count = sum(
                    formal.support_score(item["xyxy"], person["xyxy"])[0] > 0
                    for person in persons
                )
                ambiguous += int(support_count > 1)
            candidates.append(
                {
                    "fold": fold,
                    "image_path": image_path,
                    "label_path": label_path,
                    "source": manifest[image_path.name]["source"],
                    "persons": len(persons),
                    "evidence": len(evidence),
                    "max_person_iou": overlap,
                    "ambiguous_evidence": ambiguous,
                    "median_person_height": median_person_height,
                    "median_person_area_fraction": median_person_area_fraction,
                    "score": (
                        ambiguous * 12
                        + overlap * 10
                        + min(20.0, median_person_area_fraction * 1500)
                        + len(persons) * 0.5
                        + len(evidence) * 0.05
                    ),
                }
            )
    return candidates


def select_balanced(candidates: list[dict], count: int, minimum_per_source: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[item["source"]].append(item)
    for rows in grouped.values():
        rows.sort(key=lambda item: (-item["score"], item["image_path"].name))
    selected = []
    selected_names = set()
    for source in sorted(grouped):
        for item in grouped[source][:minimum_per_source]:
            selected.append(item)
            selected_names.add(item["image_path"].name)
    remaining = sorted(
        (item for item in candidates if item["image_path"].name not in selected_names),
        key=lambda item: (-item["score"], item["source"], item["image_path"].name),
    )
    selected.extend(remaining[: max(0, count - len(selected))])
    return sorted(selected[:count], key=lambda item: (item["source"], -item["score"], item["image_path"].name))


def put_label(image, text: str, origin: tuple[int, int], color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(image.shape[:2]) / 1600.0)
    thickness = max(1, round(scale * 2))
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(origin[0], image.shape[1] - width - 3))
    y = max(height + 3, min(origin[1], image.shape[0] - baseline - 3))
    cv2.rectangle(image, (x, y - height - 3), (x + width + 3, y + baseline + 2), (255, 255, 255), -1)
    cv2.putText(image, text, (x + 1, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_audit_image(image, persons, evidence):
    canvas = image.copy()
    person_rows = sorted(persons, key=lambda box: (box["xyxy"][0], box["xyxy"][1]))
    evidence_rows = sorted(evidence, key=lambda box: (box["cls"], box["xyxy"][1], box["xyxy"][0]))
    for index, person in enumerate(person_rows, 1):
        x1, y1, x2, y2 = map(lambda value: int(round(value)), person["xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PERSON_COLOR, 3)
        put_label(canvas, f"P{index}", (x1, y1), PERSON_COLOR)
    for index, item in enumerate(evidence_rows, 1):
        x1, y1, x2, y2 = map(lambda value: int(round(value)), item["xyxy"])
        color = CLASS_COLORS[item["cls"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        put_label(canvas, f"E{index}:{CLASS_LABELS[item['cls']]}", (x1, y2), color)
    all_boxes = [box["xyxy"] for box in person_rows + evidence_rows]
    x1 = max(0, int(min(box[0] for box in all_boxes)))
    y1 = max(0, int(min(box[1] for box in all_boxes)))
    x2 = min(canvas.shape[1], int(max(box[2] for box in all_boxes)) + 1)
    y2 = min(canvas.shape[0], int(max(box[3] for box in all_boxes)) + 1)
    margin_x = max(20, int((x2 - x1) * 0.08))
    margin_y = max(20, int((y2 - y1) * 0.08))
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(canvas.shape[1], x2 + margin_x), min(canvas.shape[0], y2 + margin_y)
    return canvas[y1:y2, x1:x2], person_rows, evidence_rows


def candidate_person_ids(item, persons) -> list[str]:
    output = []
    for index, person in enumerate(persons, 1):
        score = formal.support_score(item["xyxy"], person["xyxy"])
        if score[0] > 0 or score[2] > 0:
            output.append(f"P{index}")
    return output


def make_contact_sheet(items: list[tuple[str, np.ndarray]], path: Path) -> None:
    tile_width, tile_height = 900, 650
    sheet = np.full((tile_height * 2, tile_width * 2, 3), 255, dtype=np.uint8)
    for slot, (title, image) in enumerate(items):
        scale = min((tile_width - 20) / image.shape[1], (tile_height - 55) / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        row, column = divmod(slot, 2)
        x = column * tile_width + (tile_width - resized.shape[1]) // 2
        y = row * tile_height + 45 + (tile_height - 45 - resized.shape[0]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            title,
            (column * tile_width + 10, row * tile_height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--minimum-per-source", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    image_out = args.out / "annotated_images"
    sheet_out = args.out / "contact_sheets"
    image_out.mkdir()
    sheet_out.mkdir()

    selected = select_balanced(load_candidates(args.protocol_root), args.count, args.minimum_per_source)
    image_rows = []
    evidence_rows = []
    worker_rows = []
    reference_rows = []
    sheet_items = []
    for audit_index, item in enumerate(selected, 1):
        audit_id = f"A{audit_index:03d}"
        image = imread(item["image_path"])
        height, width = image.shape[:2]
        boxes = read_yolo_boxes(item["label_path"], width, height)
        persons = [box for box in boxes if box["cls"] == formal.PERSON_CLASS]
        evidence = [box for box in boxes if box["cls"] in formal.SAFETY_CLASSES]
        rendered, persons, evidence = draw_audit_image(image, persons, evidence)
        rendered_path = image_out / f"{audit_id}_{item['image_path'].stem}.jpg"
        cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(str(rendered_path))
        sheet_items.append((f"{audit_id} | {item['source']} | {item['image_path'].name}", rendered))
        image_rows.append(
            {
                "audit_id": audit_id,
                "fold": item["fold"],
                "source_group": item["source"],
                "image_name": item["image_path"].name,
                "original_image": str(item["image_path"]),
                "rendered_image": str(rendered_path),
                "person_count": len(persons),
                "evidence_count": len(evidence),
                "max_person_iou": item["max_person_iou"],
                "ambiguous_evidence_count": item["ambiguous_evidence"],
                "median_person_height": item["median_person_height"],
                "median_person_area_fraction": item["median_person_area_fraction"],
                "selection_score": item["score"],
            }
        )
        for person_index, _ in enumerate(persons, 1):
            worker_rows.append(
                {
                    "audit_id": audit_id,
                    "person_id": f"P{person_index}",
                    "helmet_state": "",
                    "vest_state": "",
                    "overall_state": "",
                    "annotator_confidence": "",
                    "notes": "",
                }
            )
        proposed = formal.assign_safety_to_persons(persons, evidence)
        proposed_by_object = {
            id(box): f"P{person_index + 1}"
            for person_index, assigned in proposed.items()
            for box in assigned
        }
        for evidence_index, box in enumerate(evidence, 1):
            row = {
                "audit_id": audit_id,
                "evidence_id": f"E{evidence_index}",
                "evidence_class_id": box["cls"],
                "evidence_class": CLASS_LABELS[box["cls"]],
                "candidate_person_ids": "|".join(candidate_person_ids(box, persons)),
                "assigned_person_id": "",
                "assignment_confidence": "",
                "occluded_or_ambiguous": "",
                "notes": "",
            }
            evidence_rows.append(row)
            reference_rows.append(
                {
                    "audit_id": audit_id,
                    "evidence_id": f"E{evidence_index}",
                    "proposed_assigned_person_id": proposed_by_object.get(id(box), "NONE"),
                }
            )

    for start in range(0, len(sheet_items), 4):
        make_contact_sheet(
            sheet_items[start : start + 4],
            sheet_out / f"sheet_{start // 4 + 1:02d}.jpg",
        )
    write_csv(args.out / "audit_image_manifest.csv", image_rows)
    write_csv(args.out / "blinded_evidence_assignment_template.csv", evidence_rows)
    write_csv(args.out / "blinded_worker_state_template.csv", worker_rows)
    write_csv(args.out / "sealed_proposed_assignment_reference.csv", reference_rows)
    instructions = """# Independent worker-PPE association audit

This set is sampled only from the source-disjoint outer test folds. Images were
selected before examining method outcomes, using source balance, worker count,
person overlap, and the number of PPE boxes geometrically supported by more
than one worker.

Annotate `blinded_evidence_assignment_template.csv` without opening
`sealed_proposed_assignment_reference.csv`. For each evidence box, enter one
person ID, `NONE`, or `AMBIGUOUS`. Then complete the worker-state template from
the visually assigned helmet and vest evidence. A second independent annotator
should repeat the task in a separate copy before adjudication. Do not describe
the set as human-validated until the author team completes and signs both
passes.

Legend: H+ helmet, H- no helmet, V+ reflective vest, V- no reflective vest.
"""
    (args.out / "ANNOTATION_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest = {
        "protocol": "source_disjoint_four_way_v1",
        "selection": "source-balanced pre-outcome visual association audit",
        "images": len(image_rows),
        "evidence_boxes": len(evidence_rows),
        "worker_boxes": len(worker_rows),
        "sources": sorted({row["source_group"] for row in image_rows}),
        "annotation_status": "templates_created_not_yet_human_signed",
    }
    (args.out / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
