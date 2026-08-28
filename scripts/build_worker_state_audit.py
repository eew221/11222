"""Build a blinded, source-stratified random human worker-state audit.

The package exposes only the image and person boxes. It does not expose PPE
candidate ownership, model predictions, thresholds, or geometry references.
Two annotators must complete and freeze separate copies before analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

import formal_worker_state_experiment as formal
from build_manual_assignment_audit import imread, load_manifest, read_yolo_boxes, write_csv


def stratum(persons: int, evidence: int) -> str:
    p = "single" if persons == 1 else "2-3" if persons <= 3 else "4+"
    e = "none" if evidence == 0 else "1-4" if evidence <= 4 else "5-9" if evidence <= 9 else "10+"
    return f"people_{p}__evidence_{e}"


def read_excluded(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {(row["source_group"], row["image_name"]) for row in csv.DictReader(stream)}


def load_pool(protocol_root: Path, excluded: set[tuple[str, str]]) -> list[dict]:
    pool = []
    seen: set[tuple[str, str]] = set()
    for fold in range(3):
        cell = protocol_root / "rate_10pct" / "seed0" / f"fold{fold}"
        manifest = load_manifest(cell)
        for image_path in sorted((cell / "images" / "test").glob("*.jpg")):
            source = manifest[image_path.name]["source"]
            key = (source, image_path.name)
            if key in excluded or key in seen:
                continue
            image = imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            boxes = read_yolo_boxes(cell / "labels" / "test" / f"{image_path.stem}.txt", width, height)
            persons = [b for b in boxes if b["cls"] == formal.PERSON_CLASS]
            evidence = [b for b in boxes if b["cls"] in formal.SAFETY_CLASSES]
            if not persons:
                continue
            seen.add(key)
            pool.append({
                "fold": fold, "source": source, "image_path": image_path,
                "label_path": cell / "labels" / "test" / f"{image_path.stem}.txt",
                "persons": len(persons), "evidence": len(evidence),
                "stratum": stratum(len(persons), len(evidence)),
            })
    return pool


def choose(pool: list[dict], per_source: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        groups[row["source"]].append(row)
    chosen = []
    for source in sorted(groups):
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in groups[source]:
            strata[row["stratum"]].append(row)
        for rows in strata.values():
            rng.shuffle(rows)
        keys = sorted(strata, key=lambda k: (-len(strata[k]), k))
        selected = []
        while len(selected) < min(per_source, len(groups[source])) and keys:
            for key in list(keys):
                if strata[key]:
                    selected.append(strata[key].pop())
                    if len(selected) >= per_source:
                        break
                if not strata[key]:
                    keys.remove(key)
        chosen.extend(selected)
    return sorted(chosen, key=lambda r: (r["source"], r["image_path"].name))


def put_label(image, text, origin, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(image.shape[:2]) / 1500.0)
    thickness = max(1, round(scale * 2))
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(int(origin[0]), image.shape[1] - w - 4))
    y = max(h + base + 4, min(int(origin[1]), image.shape[0] - 4))
    cv2.rectangle(image, (x, y - h - 4), (x + w + 4, y + base + 2), (255, 255, 255), -1)
    cv2.putText(image, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)


def render_persons(image, persons):
    canvas = image.copy()
    ordered = sorted(persons, key=lambda b: (b["xyxy"][0], b["xyxy"][1]))
    color = (0, 220, 255)
    for index, person in enumerate(ordered, 1):
        x1, y1, x2, y2 = map(lambda v: int(round(v)), person["xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        put_label(canvas, f"P{index}", (x1, y1), color)
    return canvas, ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count-per-source", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    excluded = set().union(*(read_excluded(path) for path in args.exclude))
    pool = load_pool(args.protocol_root, excluded)
    selected = choose(pool, args.count_per_source, args.seed)
    sources = sorted({r["source"] for r in selected})
    if len(sources) < 11 or any(sum(r["source"] == s for r in selected) != args.count_per_source for s in sources):
        raise RuntimeError("could not obtain the requested per-source sample")
    args.out.mkdir(parents=True)
    images_dir = args.out / "annotated_images"
    images_dir.mkdir()
    image_rows, worker_rows = [], []
    for number, item in enumerate(selected, 1):
        audit_id = f"W{number:04d}"
        image = imread(item["image_path"])
        h, w = image.shape[:2]
        boxes = read_yolo_boxes(item["label_path"], w, h)
        persons = [b for b in boxes if b["cls"] == formal.PERSON_CLASS]
        rendered, persons = render_persons(image, persons)
        output = images_dir / f"{audit_id}_{item['image_path'].stem}.jpg"
        cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(output))
        image_rows.append({
            "audit_id": audit_id, "fold": item["fold"], "source_group": item["source"],
            "image_name": item["image_path"].name, "original_image": str(item["image_path"]),
            "rendered_image": str(output), "person_count": len(persons),
            "label_evidence_count": item["evidence"], "sampling_stratum": item["stratum"],
            "selection_seed": args.seed,
        })
        for index in range(len(persons)):
            worker_rows.append({
                "audit_id": audit_id, "person_id": f"P{index + 1}",
                "helmet_state": "", "vest_state": "", "overall_state": "",
                "annotator_confidence": "", "visibility_issue": "", "notes": "",
            })
    write_csv(args.out / "audit_image_manifest.csv", image_rows)
    write_csv(args.out / "blinded_worker_state_template.csv", worker_rows)
    instructions = """# 独立开放候选 worker-state 人工审计

本审计从源级隔离最终测试图像中按 filename group 分层随机抽样。页面只展示原图和用于定位的人员框 P1、P2、P3；不展示 PPE 框、几何候选人、模型输出、阈值或旧人工答案。请只依据原始图像判断每个可见工人的 PPE 状态。

## 标注步骤

1. 两位专家分别打开自己的 `annotator_A` 或 `annotator_B` 页面，不能查看对方文件、模型输出、几何 reference 或预测结果。
2. 对当前图像的每个 P 编号，依次判断安全帽、反光背心和总体状态。
3. `SAFE`：对应 PPE 清楚可见且符合要求；`UNSAFE`：明确未佩戴/不符合要求；`REVIEW`：被遮挡、太小、截断、光照不足或无法可靠判断。不要猜。
4. 总体状态按真实判断填写：只要安全帽或反光背心明确不合格，选 `UNSAFE`；两项都明确合格才选 `SAFE`；其余选 `REVIEW`。组件状态和总体状态都必须填写。
5. 置信度：高=证据清楚；中=基本能判断但有轻微问题；低=勉强判断。它是专家判断置信度，不是模型分数。
6. 若发现图中有明显可见、但没有 P 框的人员，在“可见性问题”选“有未框出人员”，并在备注写明大致位置；不要把该人员的状态强行套到某个 P 上。
7. 备注只记录必要事实，例如“背部被遮挡”“人太小”“两人重叠”。不为了完成数量而猜测。
8. 全部 P 行完成后先导出 CSV 备份，再点击“冻结本轮”。冻结后不能修改；它只锁定专家自己的原始答案，不会自动改答案。

## 特别说明

这是 worker-state 人工真值审计，不是 PPE 框归属审计。不要先数几何框，也不要根据“模型应该怎么判”反推答案。手上拿着的安全帽/背心不等于正在佩戴，按图像能否判断“该工人的 PPE 状态”填写；不确定就选 REVIEW。
"""
    (args.out / "ANNOTATION_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    for annotator in ("A", "B"):
        folder = args.out / f"annotator_{annotator}"
        folder.mkdir()
        shutil.copy2(args.out / "blinded_worker_state_template.csv", folder / "worker_state.csv")
        (folder / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest = {
        "protocol": "independent_open_set_worker_state_audit_v1",
        "selection": "pre-outcome source-stratified random sample from outer test; prior audits excluded",
        "selection_seed": args.seed, "count_per_source": args.count_per_source,
        "images": len(image_rows), "worker_rows": len(worker_rows),
        "sources": sources, "excluded_rows": len(excluded),
        "annotation_status": "templates_created_not_yet_human_signed",
        "reference_visibility": "person boxes only; no PPE boxes or method output shown",
    }
    (args.out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
