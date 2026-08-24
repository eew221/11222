"""Create an open-set, three-pass version of a frozen random ownership audit.

The source audit supplies the already frozen image sample.  This script never
reads detector predictions or method assignments.  It only exposes every
visible person ID in each rendered image, so a human owner is not constrained
by the geometric candidate list used in the original audit template.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


FIELDS = (
    "audit_id", "evidence_id", "evidence_class_id", "evidence_class",
    "candidate_person_ids", "all_visible_person_ids", "assigned_person_id",
    "assignment_confidence", "occluded_or_ambiguous", "notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_audit.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")

    image_manifest = source / "audit_image_manifest.csv"
    evidence_template = source / "blinded_evidence_assignment_template.csv"
    images_dir = source / "annotated_images"
    if not image_manifest.is_file() or not evidence_template.is_file() or not images_dir.is_dir():
        raise FileNotFoundError("source audit is missing its manifest, template, or rendered images")

    images = {row["audit_id"]: row for row in read_csv(image_manifest)}
    original_rows = read_csv(evidence_template)
    if not images or not original_rows:
        raise ValueError("source audit has no images or evidence rows")

    rows: list[dict[str, str]] = []
    for row in original_rows:
        audit_id = row["audit_id"]
        if audit_id not in images:
            raise ValueError(f"template references unknown image {audit_id}")
        person_count = int(images[audit_id]["person_count"])
        rows.append({
            "audit_id": audit_id,
            "evidence_id": row["evidence_id"],
            "evidence_class_id": row["evidence_class_id"],
            "evidence_class": row["evidence_class"],
            "candidate_person_ids": row["candidate_person_ids"],
            "all_visible_person_ids": "|".join(f"P{i}" for i in range(1, person_count + 1)),
            "assigned_person_id": "",
            "assignment_confidence": "",
            "occluded_or_ambiguous": "",
            "notes": "",
        })

    out.mkdir(parents=True)
    shutil.copy2(image_manifest, out / "audit_image_manifest.csv")
    shutil.copytree(images_dir, out / "annotated_images")
    write_csv(out / "blinded_open_set_assignment_template.csv", rows)
    for annotator in ("A", "B", "C"):
        target = out / f"annotator_{annotator}"
        target.mkdir()
        write_csv(target / "evidence_assignment.csv", rows)

    source_manifest = json.loads((source / "audit_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "protocol": "source_stratified_random_open_set_owner_audit_v2",
        "selection": "reuses the frozen v1 image sample without resampling",
        "source_audit": str(source),
        "source_image_manifest_sha256": sha256(image_manifest),
        "source_evidence_template_sha256": sha256(evidence_template),
        "source_selection": source_manifest,
        "images": len(images),
        "evidence_boxes": len(rows),
        "annotators": ["A", "B", "C"],
        "blinding": (
            "A, B, and C annotate independently before any responses, detector predictions, "
            "thresholds, or method outputs are disclosed. C is a third independent pass, not "
            "an adjudicator who sees A/B selections."
        ),
        "open_set_rule": (
            "Annotators may select any visible person ID P1..Pn shown in the image, NONE, or "
            "AMBIGUOUS. The original geometric candidate list is retained only for post hoc "
            "candidate-recall reporting."
        ),
        "reference_rule": (
            "A final label is the three-pass majority when at least two blind annotators agree; "
            "three-way disagreements are retained as AMBIGUOUS rather than resolved after seeing "
            "other responses."
        ),
        "annotation_status": "templates_created_not_yet_human_signed",
        "created_utc": utc_now(),
    }
    (out / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    instructions = """# 随机 PPE-工人归属审计（开放候选，三位专家盲标）

## 本次审计要回答什么

这 66 张图像来自已经冻结的分层随机抽样：11 个 filename group 各 6 张，共 334 个 PPE 框。抽样时没有读取检测器预测、RC-WSSI、阈值或任何已有人工答案。本次只判断：**每个 PPE 框在画面中实际属于哪位已标号人员。**

## 标注规则

1. 图中黄色人员框标为 `P1`、`P2` 等；彩色 PPE 框标为 `E1`、`E2` 等。
2. 每个 `E` 框都可选择画面中 **任何** 显示的 `P1...Pn`，不限于系统标出的“几何候选”。
3. 若 PPE 明显不属于任何显示人员或是误框，选择 `NONE`。
4. 若因遮挡、重叠、模糊或画面截断而无法可靠判断，选择 `AMBIGUOUS`。不确定时优先选择它，不要猜测。
5. “几何候选”只以浅色标识供研究者在事后计算候选集覆盖率；它不是推荐答案，不能替代视觉判断。
6. 置信度填写的是人工视觉判断的把握程度，不是模型分数。高：清楚可见；中：大致可判断但有局部遮挡；低：只能勉强判断。

## 盲法与冻结

- A、B、C 三位专家必须分别完成自己的页面，不能查看、讨论或修改其他人的答案。
- 三人均不知道检测器输出、RC-WSSI 输出、阈值和其他专家的选择。
- 每一轮完成全部 PPE 行后点击“冻结本轮”。冻结后不可修改。
- 三轮均冻结后，程序仅用预先声明的多数规则：至少两人同意即为最终参考；三人完全不一致则记录为 `AMBIGUOUS`，不让仲裁者事后看到他人答案来改变结果。
"""
    (out / "ANNOTATION_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    for annotator in ("A", "B", "C"):
        (out / f"annotator_{annotator}" / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
