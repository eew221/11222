"""Audit source leakage in the image-level calibration/evaluation split.

The formal worker-state experiment uses a deterministic MD5 split of image
filenames. This CPU-only audit infers a tunnel/video source from the filename
prefix and reports whether a source appears on both sides of that split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def stable_half(path: Path) -> int:
    key = path.name.encode("utf-8", errors="ignore")
    return int(hashlib.md5(key).hexdigest()[:8], 16) % 2


def source_id(path: Path) -> str:
    """Infer a source name from names such as '11_tunnel (1028).jpg'."""
    stem = path.stem
    while Path(stem).suffix.lower() in IMAGE_SUFFIXES:
        stem = Path(stem).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return stem.strip() or path.stem


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=Path(r"D:\ppe_pilot"))
    parser.add_argument("--rates", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for rate in args.rates:
        for seed in args.seeds:
            for fold in args.folds:
                tag = f"r{rate}_s{seed}_f{fold}"
                image_dir = args.split_root / f"target_{rate}pct" / f"seed{seed}" / f"fold{fold}" / "images" / "val"
                images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
                if not images:
                    continue
                groups: dict[str, list[Path]] = defaultdict(list)
                for image in images:
                    groups[source_id(image)].append(image)
                source_rows = []
                for source, source_images in sorted(groups.items()):
                    cal_count = sum(stable_half(path) == 0 for path in source_images)
                    eval_count = len(source_images) - cal_count
                    source_rows.append((source, len(source_images), cal_count, eval_count))
                shared_sources = [item for item in source_rows if item[2] and item[3]]
                shared_images = sum(item[1] for item in shared_sources)
                rows.append(
                    {
                        "rate": rate,
                        "seed": seed,
                        "fold": fold,
                        "tag": tag,
                        "val_images": len(images),
                        "source_groups": len(source_rows),
                        "calibration_images": sum(item[2] for item in source_rows),
                        "evaluation_images": sum(item[3] for item in source_rows),
                        "shared_source_groups": len(shared_sources),
                        "shared_source_group_rate": len(shared_sources) / len(source_rows),
                        "images_from_shared_sources": shared_images,
                        "image_share_from_shared_sources": shared_images / len(images),
                        "source_grouping_rule": "filename_prefix_before_parenthesized_frame_index",
                    }
                )

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "image_hash_group_overlap.csv", rows)
    by_rate: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_rate[int(row["rate"])].append(row)
    report = [
        "# Image-Level Split Group-Overlap Audit",
        "",
        "This audit reads filenames only. It does not load images, labels, models, or GPU libraries.",
        "",
        "The source identifier is inferred from the filename prefix before the parenthesized frame index, for example `11_tunnel (1028).jpg -> 11_tunnel`.",
        "",
        "| Label rate | Split instances | Mean source groups | Mean shared-source rate | Mean image share from shared sources |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rate in sorted(by_rate):
        group = by_rate[rate]
        report.append(
            "| {rate}% | {n} | {sources:.2f} | {shared:.2%} | {images:.2%} |".format(
                rate=rate,
                n=len(group),
                sources=mean(float(row["source_groups"]) for row in group),
                shared=mean(float(row["shared_source_group_rate"]) for row in group),
                images=mean(float(row["image_share_from_shared_sources"]) for row in group),
            )
        )
    report.extend(
        [
            "",
            "## Consequence",
            "",
            "When the shared-source rate is nonzero, calibration and evaluation can contain frames from the same inferred tunnel/video source. This can make the image-level split optimistic and weakens an independence interpretation of the worker-level Clopper-Pearson bound.",
            "",
            "The next formal rerun should replace this filename-hash split with source/video/worker-track-disjoint calibration and evaluation partitions. The filename heuristic should be replaced with true video identifiers when they are available.",
        ]
    )
    (args.out / "image_hash_group_overlap_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[done] audited_splits={len(rows)} out={args.out}")


if __name__ == "__main__":
    main()
