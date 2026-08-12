"""Build a blinded, source-stratified random PPE-owner audit set.

The sampler reads only the source-disjoint test manifest and human object
boxes. It does not read detector predictions, thresholds, or assignment
outputs. The generated package is intentionally incomplete until two
annotators finish and freeze their passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import cv2

import formal_worker_state_experiment as formal
from build_manual_assignment_audit import (
    CLASS_COLORS,
    CLASS_LABELS,
    PERSON_COLOR,
    candidate_person_ids,
    draw_audit_image,
    imread,
    load_manifest,
    put_label,
    read_yolo_boxes,
    write_csv,
)


def stratum(person_count: int, evidence_count: int) -> str:
    people = "single" if person_count == 1 else "2-3" if person_count <= 3 else "4+"
    evidence = "1-4" if evidence_count <= 4 else "5-9" if evidence_count <= 9 else "10+"
    return f"people_{people}__evidence_{evidence}"


def load_pool(protocol_root: Path, excluded: set[tuple[str, str]]) -> list[dict]:
    pool: list[dict] = []
    for fold in range(3):
        cell_root = protocol_root / "rate_10pct" / "seed0" / f"fold{fold}"
        manifest = load_manifest(cell_root)
        for image_path in sorted((cell_root / "images" / "test").glob("*.jpg")):
            source = manifest[image_path.name]["source"]
            key = (source, image_path.name)
            if key in excluded:
                continue
            image = imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            label_path = cell_root / "labels" / "test" / f"{image_path.stem}.txt"
            boxes = read_yolo_boxes(label_path, width, height)
            persons = [box for box in boxes if box["cls"] == formal.PERSON_CLASS]
            evidence = [box for box in boxes if box["cls"] in formal.SAFETY_CLASSES]
            if not persons or not evidence:
                continue
            pool.append({
                "fold": fold,
                "source": source,
                "image_path": image_path,
                "label_path": label_path,
                "persons": len(persons),
                "evidence": len(evidence),
                "stratum": stratum(len(persons), len(evidence)),
            })
    return pool


def read_excluded(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {(row["source_group"], row["image_name"]) for row in csv.DictReader(stream)}


def choose(pool: list[dict], count_per_source: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_source[row["source"]].append(row)
    selected: list[dict] = []
    for source in sorted(by_source):
        rows = by_source[source]
        by_stratum: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_stratum[row["stratum"]].append(row)
        for values in by_stratum.values():
            rng.shuffle(values)
        strata = sorted(by_stratum, key=lambda key: (-len(by_stratum[key]), key))
        take: list[dict] = []
        while len(take) < min(count_per_source, len(rows)) and strata:
            progressed = False
            for key in list(strata):
                if by_stratum[key]:
                    take.append(by_stratum[key].pop())
                    progressed = True
                    if len(take) == count_per_source:
                        break
                if not by_stratum[key]:
                    strata.remove(key)
            if not progressed:
                break
        selected.extend(take)
    return sorted(selected, key=lambda row: (row["source"], row["image_path"].name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--difficult-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count-per-source", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    excluded = read_excluded(args.difficult_manifest)
    pool = load_pool(args.protocol_root, excluded)
    selected = choose(pool, args.count_per_source, args.seed)
    sources = sorted({row["source"] for row in selected})
    if len(sources) != 11 or any(sum(row["source"] == source for row in selected) != args.count_per_source for source in sources):
        raise RuntimeError("the requested balanced sample could not cover all 11 source groups")

    args.out.mkdir(parents=True)
    image_out = args.out / "annotated_images"
    image_out.mkdir()
    for annotator in ("A", "B"):
        annotator_dir = args.out / f"annotator_{annotator}"
        annotator_dir.mkdir()
        shutil.copy2(
            Path(__file__).resolve().parents[1] / "experiments" / "manual_worker_ppe_association_audit_20260810_v1" / "annotation_app.html",
            args.out / "annotation_app.html",
        )

    image_rows: list[dict] = []
    evidence_rows: list[dict] = []
    worker_rows: list[dict] = []
    for audit_index, item in enumerate(selected, 1):
        audit_id = f"R{audit_index:03d}"
        image = imread(item["image_path"])
        height, width = image.shape[:2]
        boxes = read_yolo_boxes(item["label_path"], width, height)
        persons = [box for box in boxes if box["cls"] == formal.PERSON_CLASS]
        evidence = [box for box in boxes if box["cls"] in formal.SAFETY_CLASSES]
        rendered, persons, evidence = draw_audit_image(image, persons, evidence)
        rendered_path = image_out / f"{audit_id}_{item['image_path'].stem}.jpg"
        cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(str(rendered_path))
        image_rows.append({
            "audit_id": audit_id,
            "fold": item["fold"],
            "source_group": item["source"],
            "image_name": item["image_path"].name,
            "original_image": str(item["image_path"]),
            "rendered_image": str(rendered_path),
            "person_count": len(persons),
            "evidence_count": len(evidence),
            "sampling_stratum": item["stratum"],
            "selection_seed": args.seed,
        })
        for person_index in range(len(persons)):
            worker_rows.append({
                "audit_id": audit_id,
                "person_id": f"P{person_index + 1}",
                "helmet_state": "",
                "vest_state": "",
                "overall_state": "",
                "annotator_confidence": "",
                "notes": "",
            })
        for evidence_index, box in enumerate(evidence, 1):
            evidence_rows.append({
                "audit_id": audit_id,
                "evidence_id": f"E{evidence_index}",
                "evidence_class_id": box["cls"],
                "evidence_class": CLASS_LABELS[box["cls"]],
                "candidate_person_ids": "|".join(candidate_person_ids(box, persons)),
                "assigned_person_id": "",
                "assignment_confidence": "",
                "occluded_or_ambiguous": "",
                "notes": "",
            })

    for annotator in ("A", "B"):
        annotator_dir = args.out / f"annotator_{annotator}"
        write_csv(annotator_dir / "evidence_assignment.csv", evidence_rows)
        write_csv(annotator_dir / "worker_state.csv", worker_rows)
        (annotator_dir / "INSTRUCTIONS.md").write_text(
            (args.out / "ANNOTATION_INSTRUCTIONS.md").read_text(encoding="utf-8") if (args.out / "ANNOTATION_INSTRUCTIONS.md").exists() else "",
            encoding="utf-8",
        )
    write_csv(args.out / "audit_image_manifest.csv", image_rows)
    write_csv(args.out / "blinded_evidence_assignment_template.csv", evidence_rows)
    write_csv(args.out / "blinded_worker_state_template.csv", worker_rows)
    instructions = """# 随机人工 PPE-工人归属审计

这是对前一批“困难候选子集”审计的互补抽样。图像来自源级隔离协议的最终测试角色，按 11 个 filename group 分层；每组固定抽取 6 张，组内再按工人数和 PPE 框数量分层后使用固定随机种子抽样。抽样时没有读取检测器预测、RC-WSSI 结果、阈值或旧人工答案。

## 标注步骤

1. 只打开你自己的 `annotator_A` 或 `annotator_B` 页面/文件，不要打开另一个专家文件，也不要打开任何 sealed/reference 文件。
2. 看中间图片。黄色框是工人，编号为 `P1、P2...`；彩色框是 PPE，编号为 `E1、E2...`。右侧每一行对应一个 PPE 框。
3. 对每个 PPE 框，只判断“它现实中属于哪个工人”：点击对应的 `P1/P2...`。如果画面中确实没有对应工人，选 `NONE`；如果看不清、被遮挡或两人无法区分，选 `AMBIGUOUS`。
4. 置信度按你的视觉判断填写：`高`=归属清楚，`中`=大致能判断但有遮挡/距离问题，`低`=只能勉强判断。置信度不是模型分数，不要参考任何模型输出。
5. `是否遮挡/有歧义`：只要 PPE 或对应工人明显被遮挡、截断、重叠，选“是”；否则选“否”。
6. 备注只写必要理由，例如“手上拿着”“两人重叠”“只有半个身体”。不要为了凑结果而猜测；不确定就选 `AMBIGUOUS`。
7. 所有 PPE 行完成后，再冻结本人的标注。冻结后不能修改，这个“冻结”只是防止事后改动，不代表系统自动修改了专家答案。

## 类别提示

`H+`=有安全帽，`H-`=未戴安全帽，`V+`=有反光背心，`V-`=未穿反光背心。这里审计的核心是 PPE 框到工人的归属，不是评价检测器有没有检测到它。

## 重要禁止事项

不要看 `sealed_proposed_assignment_reference.csv`、预测缓存、阈值结果或另一位专家的 CSV。两位专家都冻结后，才进行第三位专家仲裁和方法对比。
"""
    (args.out / "ANNOTATION_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    for annotator in ("A", "B"):
        (args.out / f"annotator_{annotator}" / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest = {
        "protocol": "source_disjoint_four_way_v1",
        "selection": "source-stratified random sample from outer test, excluding the difficult audit",
        "selection_seed": args.seed,
        "count_per_source": args.count_per_source,
        "images": len(image_rows),
        "evidence_boxes": len(evidence_rows),
        "worker_boxes": len(worker_rows),
        "sources": sources,
        "excluded_difficult_rows": len(excluded),
        "annotation_status": "templates_created_not_yet_human_signed",
    }
    (args.out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
