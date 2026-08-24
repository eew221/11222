"""Build auditable real-image qualitative examples from RC-WSSI caches.

No detector inference is performed. The script reconstructs assignments and
worker states from saved predictions, applies deterministic case definitions,
and records every selected image plus its candidate count and ranking rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORKSPACE / ".ultralytics"))
(WORKSPACE / ".ultralytics").mkdir(parents=True, exist_ok=True)

import formal_worker_state_experiment as formal


CLASS_NAMES = {
    0: "helmet",
    1: "no helmet",
    2: "no vest",
    3: "person",
    4: "vest",
}
PERSON_CLASS = 3
SAFETY_CLASSES = {0, 1, 2, 4}
STATE_COLORS = {"safe": "#009E73", "unsafe": "#D55E00", "review": "#E69F00"}
PPE_COLORS = {
    0: "#56B4E9",
    1: "#CC79A7",
    2: "#D55E00",
    4: "#0072B2",
}
STATE_SHORT = {"safe": "S", "unsafe": "U", "review": "R"}
PRIVACY_REDACTIONS: dict[str, list[dict]] = {}


def path_key(path: Path | str) -> str:
    return str(Path(path).resolve()).lower()


def configure_annotation_semantics() -> None:
    formal.PERSON_CLASS = PERSON_CLASS
    formal.SAFETY_CLASSES = set(SAFETY_CLASSES)
    formal.HELMET_SAFE = 0
    formal.HELMET_UNSAFE = 1
    formal.VEST_UNSAFE = 2
    formal.VEST_SAFE = 4


configure_annotation_semantics()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for region in PRIVACY_REDACTIONS.get(path_key(path), []):
        height, width = image.shape[:2]
        x1 = max(0, min(width, int(round(float(region["x1"])))))
        y1 = max(0, min(height, int(round(float(region["y1"])))))
        x2 = max(0, min(width, int(round(float(region["x2"])))))
        y2 = max(0, min(height, int(round(float(region["y2"])))))
        if x2 <= x1 or y2 <= y1:
            continue
        source = image[y1:y2, x1:x2]
        # Pixelation is intentionally strong enough that the region cannot be
        # reconstructed from the released qualitative panel.
        small_width = max(1, min(12, source.shape[1] // 10 or 1))
        small_height = max(1, min(12, source.shape[0] // 10 or 1))
        mosaic = cv2.resize(source, (small_width, small_height), interpolation=cv2.INTER_AREA)
        image[y1:y2, x1:x2] = cv2.resize(
            mosaic, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    encoded = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", encoded)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buffer.tofile(str(path))


def load_privacy_redactions(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen"):
        raise ValueError("privacy redaction manifest must be frozen before rendering")
    images = payload.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("privacy redaction manifest has no image mapping")
    regions: dict[str, list[dict]] = {}
    for record in images.values():
        original = record.get("original_image_path")
        if not original or not record.get("reviewed"):
            continue
        rows = record.get("regions", [])
        if not isinstance(rows, list):
            raise ValueError("privacy redaction regions must be a list")
        regions[path_key(original)] = rows
    return {"payload": payload, "regions": regions}


def as_int(row: dict, key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def as_float(row: dict, key: str) -> float:
    return float(row.get(key, 0) or 0)


class CacheStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.payloads: dict[str, dict] = {}
        self.images: dict[str, dict[str, dict]] = {}

    def payload(self, tag: str) -> dict:
        if tag not in self.payloads:
            payload = json.loads((self.cache_dir / f"{tag}.json").read_text(encoding="utf-8"))
            semantic_names = {
                int(key): str(value)
                for key, value in payload["annotation_semantic_class_names"].items()
            }
            if semantic_names != {
                0: "helmet",
                1: "no_helmet",
                2: "no_reflective_vest",
                3: "person",
                4: "reflective_vest",
            }:
                raise ValueError(f"{tag}: unexpected annotation semantics {semantic_names}")
            self.payloads[tag] = payload
            self.images[tag] = {
                str(item["image_path"]).lower(): item for item in payload["images"]
            }
        return self.payloads[tag]

    def image(self, tag: str, path: str) -> dict:
        self.payload(tag)
        return self.images[tag][str(path).lower()]


def build_cases(rows: list[dict], rate: int) -> list[dict]:
    selected = [
        row
        for row in rows
        if as_int(row, "rate") == rate
        and row["method"] in {"state_selected_separate", "state_selected_no_assignment"}
    ]
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in selected:
        grouped[(row["tag"], row["image_path"])][row["method"]] = row
    cases = []
    for (tag, path), methods in grouped.items():
        if set(methods) != {"state_selected_separate", "state_selected_no_assignment"}:
            continue
        cases.append(
            {
                "tag": tag,
                "image_path": path,
                "worker": methods["state_selected_separate"],
                "global": methods["state_selected_no_assignment"],
            }
        )
    return cases


def transition_errors(row: dict) -> int:
    return sum(
        as_int(row, key)
        for key in ("unsafe_to_safe", "unsafe_to_review", "safe_to_unsafe", "safe_to_review")
    )


def person_area_fraction(case: dict, cache: CacheStore) -> float:
    item = cache.image(case["tag"], case["image_path"])
    persons = [box for box in item["ground_truth"] if int(box["cls"]) == PERSON_CLASS]
    image = read_image(Path(case["image_path"]))
    height, width = image.shape[:2]
    if not persons:
        return 0.0
    return max(
        (box["xyxy"][2] - box["xyxy"][0])
        * (box["xyxy"][3] - box["xyxy"][1])
        / (width * height)
        for box in persons
    )


def select_cases(cases: list[dict], cache: CacheStore) -> list[dict]:
    single = [
        case
        for case in cases
        if as_int(case["worker"], "gt_person_count") == 1
        and as_int(case["worker"], "matched") == 1
        and as_int(case["worker"], "unmatched_gt") == 0
        and as_int(case["worker"], "unmatched_pred") == 0
        and as_int(case["worker"], "pred_review") == 0
        and as_int(case["worker"], "tp_safe") + as_int(case["worker"], "tp_unsafe") == 1
    ]
    if not single:
        raise ValueError("no clean single-worker candidate")
    single.sort(
        key=lambda case: (
            -person_area_fraction(case, cache),
            case["image_path"].lower(),
            case["tag"],
        )
    )
    chosen_single = single[0]

    contamination = [
        case
        for case in cases
        if 2 <= as_int(case["worker"], "gt_person_count") <= 5
        and as_int(case["worker"], "unsafe_to_safe") == 0
        and as_int(case["global"], "unsafe_to_safe")
        > as_int(case["worker"], "unsafe_to_safe")
        and as_int(case["worker"], "tp_unsafe") > 0
        and as_int(case["worker"], "unmatched_gt") == 0
        and as_int(case["worker"], "unmatched_pred") == 0
        and transition_errors(case["worker"]) == 0
    ]
    if not contamination:
        raise ValueError("no clean global-pooling contamination candidate")
    contamination.sort(
        key=lambda case: (
            -(
                as_int(case["global"], "unsafe_to_safe")
                - as_int(case["worker"], "unsafe_to_safe")
            ),
            abs(as_int(case["worker"], "gt_person_count") - 3),
            case["image_path"].lower(),
            case["tag"],
        )
    )
    chosen_contamination = contamination[0]

    multi = [
        case
        for case in cases
        if 2 <= as_int(case["worker"], "gt_person_count") <= 5
        and as_int(case["worker"], "gt_review") == 0
        and as_int(case["worker"], "matched")
        == as_int(case["worker"], "gt_person_count")
        and as_int(case["worker"], "unmatched_gt") == 0
        and as_int(case["worker"], "unmatched_pred") == 0
        and transition_errors(case["worker"]) == 0
        and as_int(case["global"], "unsafe_to_safe") == 0
        and case["image_path"].lower() != chosen_contamination["image_path"].lower()
    ]
    if not multi:
        raise ValueError("no clean multi-worker candidate")
    multi.sort(
        key=lambda case: (
            0
            if as_int(case["worker"], "gt_safe") > 0
            and as_int(case["worker"], "gt_unsafe") > 0
            else 1,
            abs(as_int(case["worker"], "gt_person_count") - 3),
            abs(as_float(case["worker"], "brightness") - 0.40),
            case["image_path"].lower(),
            case["tag"],
        )
    )
    chosen_multi = multi[0]

    review = [
        case
        for case in cases
        if 1 <= as_int(case["worker"], "gt_person_count") <= 4
        and as_int(case["worker"], "unsafe_to_review")
        + as_int(case["worker"], "safe_to_review")
        > 0
        and as_int(case["worker"], "unmatched_gt") == 0
        and as_int(case["worker"], "unmatched_pred") == 0
        and case["image_path"].lower()
        not in {
            chosen_single["image_path"].lower(),
            chosen_multi["image_path"].lower(),
            chosen_contamination["image_path"].lower(),
        }
    ]
    if not review:
        raise ValueError("no review candidate")
    review.sort(
        key=lambda case: (
            abs(as_int(case["worker"], "gt_person_count") - 1),
            as_float(case["worker"], "brightness"),
            -person_area_fraction(case, cache),
            case["image_path"].lower(),
            case["tag"],
        )
    )
    chosen_review = review[0]

    selected = [
        {
            **chosen_single,
            "panel": "a",
            "case_type": "single_worker_correct",
            "title": "Single-worker decision",
            "candidate_count": len(single),
            "selection_rule": "largest annotated-person area among clean correct single-worker cases",
        },
        {
            **chosen_multi,
            "panel": "b",
            "case_type": "multi_worker_correct",
            "title": "Mixed multi-worker assignment",
            "candidate_count": len(multi),
            "selection_rule": "mixed safe/unsafe first, then three workers and brightness nearest 0.40",
        },
        {
            **chosen_contamination,
            "panel": "c",
            "case_type": "global_pooling_contamination",
            "title": "Global-pooling contamination",
            "candidate_count": len(contamination),
            "selection_rule": "largest added unsafe-to-safe count, then worker count nearest three",
        },
        {
            **chosen_review,
            "panel": "d",
            "case_type": "review_case",
            "title": "Review under weak evidence",
            "candidate_count": len(review),
            "selection_rule": "single-worker first, then lowest brightness and largest person area",
        },
    ]
    return selected


def crop_bounds(boxes: list[dict], width: int, height: int, target_aspect: float = 4 / 3):
    x1 = min(float(box["xyxy"][0]) for box in boxes)
    y1 = min(float(box["xyxy"][1]) for box in boxes)
    x2 = max(float(box["xyxy"][2]) for box in boxes)
    y2 = max(float(box["xyxy"][3]) for box in boxes)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = max(x2 - x1, width * 0.23) * 1.35
    crop_height = max(y2 - y1, height * 0.23) * 1.35
    if crop_width / crop_height < target_aspect:
        crop_width = crop_height * target_aspect
    else:
        crop_height = crop_width / target_aspect
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)
    left = max(0.0, min(width - crop_width, center_x - crop_width / 2))
    top = max(0.0, min(height - crop_height, center_y - crop_height / 2))
    return int(round(left)), int(round(top)), int(round(left + crop_width)), int(round(top + crop_height))


def top_evidence_by_class(assigned: list[dict]) -> list[dict]:
    best: dict[int, dict] = {}
    for prediction in assigned:
        cls_id = int(prediction["cls"])
        if cls_id not in best or float(prediction["conf"]) > float(best[cls_id]["conf"]):
            best[cls_id] = prediction
    return [best[key] for key in sorted(best)]


def draw_box(ax, box, *, color, linewidth, linestyle="-", zorder=3):
    x1, y1, x2, y2 = map(float, box["xyxy"])
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    )


def render_panel(ax, selected: dict, cache: CacheStore) -> None:
    item = cache.image(selected["tag"], selected["image_path"])
    image = read_image(Path(selected["image_path"]))
    predictions = item["predictions"]
    ground_truth = item["ground_truth"]
    threshold = as_float(selected["worker"], "threshold")
    persons = [box for box in predictions if int(box["cls"]) == PERSON_CLASS]
    evidence = [box for box in predictions if int(box["cls"]) in SAFETY_CLASSES]
    gt_persons = [box for box in ground_truth if int(box["cls"]) == PERSON_CLASS]
    worker_states = formal.infer_worker_state(persons, evidence, threshold, "worker")
    global_states = formal.infer_worker_state(persons, evidence, threshold, "global")
    gt_states = formal.infer_gt_worker_state(gt_persons, ground_truth)
    matches, unmatched_pred, unmatched_gt = formal.match_persons(persons, gt_persons)
    if unmatched_gt:
        raise ValueError(f"selected panel {selected['panel']} contains unmatched GT")
    match_by_pred = {pred_index: gt_index for pred_index, gt_index, _ in matches}
    assigned = formal.assign_safety_to_persons(persons, evidence)

    visible_boxes = list(gt_persons) + list(persons)
    for person_index in range(len(persons)):
        visible_boxes.extend(top_evidence_by_class(assigned.get(person_index, [])))
    height, width = image.shape[:2]
    left, top, right, bottom = crop_bounds(visible_boxes, width, height)

    ax.imshow(image)
    for person_index, state_record in enumerate(worker_states):
        state = state_record["state"]
        draw_box(ax, persons[person_index], color=STATE_COLORS[state], linewidth=2.2, zorder=5)
        x1, y1, x2, y2 = map(float, persons[person_index]["xyxy"])
        gt_state = (
            gt_states[match_by_pred[person_index]]["state"]
            if person_index in match_by_pred
            else "review"
        )
        global_state = global_states[person_index]["state"]
        if selected["case_type"] == "global_pooling_contamination" or global_state != state:
            label = (
                f"REF/RC/Pool: {STATE_SHORT[gt_state]}/"
                f"{STATE_SHORT[state]}/{STATE_SHORT[global_state]}"
            )
        else:
            label = f"REF/RC: {STATE_SHORT[gt_state]}/{STATE_SHORT[state]}"
        label_y = max(top + 4, y1 - 5)
        ax.text(
            x1,
            label_y,
            label,
            fontsize=6.6,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": STATE_COLORS[state], "edgecolor": "none", "alpha": 0.92, "pad": 1.3},
            zorder=8,
        )
        person_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        for ppe in top_evidence_by_class(assigned.get(person_index, [])):
            cls_id = int(ppe["cls"])
            draw_box(ax, ppe, color=PPE_COLORS[cls_id], linewidth=1.15, zorder=6)
            ex1, ey1, ex2, ey2 = map(float, ppe["xyxy"])
            evidence_center = ((ex1 + ex2) / 2, (ey1 + ey2) / 2)
            ax.plot(
                [evidence_center[0], person_center[0]],
                [evidence_center[1], person_center[1]],
                color=PPE_COLORS[cls_id],
                linewidth=0.65,
                alpha=0.65,
                zorder=4,
            )
            ax.text(
                ex1,
                min(bottom - 3, ey2 + 3),
                f"{CLASS_NAMES[cls_id]} {float(ppe['conf']):.2f}",
                fontsize=5.4,
                color="white",
                ha="left",
                va="top",
                bbox={"facecolor": "#202020", "edgecolor": PPE_COLORS[cls_id], "alpha": 0.82, "pad": 0.8},
                zorder=8,
            )

    for person_index in unmatched_pred:
        draw_box(ax, persons[person_index], color="#7F7F7F", linewidth=1.3, linestyle="--")

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")
        spine.set_linewidth(0.8)
    ax.set_title(
        f"({selected['panel']}) {selected['title']}",
        loc="left",
        fontsize=8.6,
        fontweight="bold",
        pad=3,
    )
    ax.text(
        0.99,
        0.015,
        f"{selected['tag']}  |  t={threshold:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color="white",
        bbox={"facecolor": "#202020", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        zorder=9,
    )


def manifest_row(selected: dict, copied_name: str) -> dict:
    worker = selected["worker"]
    global_row = selected["global"]
    return {
        "panel": selected["panel"],
        "case_type": selected["case_type"],
        "tag": selected["tag"],
        "image_name": worker["image_name"],
        "original_image_path": selected["image_path"],
        "copied_source_frame": copied_name,
        "source_group": worker["source_group"],
        "brightness": worker["brightness"],
        "selected_threshold": worker["threshold"],
        "gt_person_count": worker["gt_person_count"],
        "gt_safe": worker["gt_safe"],
        "gt_unsafe": worker["gt_unsafe"],
        "gt_review": worker["gt_review"],
        "rc_tp_safe": worker["tp_safe"],
        "rc_tp_unsafe": worker["tp_unsafe"],
        "rc_unsafe_to_safe": worker["unsafe_to_safe"],
        "rc_unsafe_to_review": worker["unsafe_to_review"],
        "global_unsafe_to_safe": global_row["unsafe_to_safe"],
        "candidate_count": selected["candidate_count"],
        "selection_rule": selected["selection_rule"],
    }


def main() -> None:
    global PRIVACY_REDACTIONS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            r"D:\ppe_pilot\person_conditioned_gate\wssi_mva_v2_5_10_20260805_0850"
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/rc_wssi_qualitative_20260805")
    )
    parser.add_argument("--rate", type=int, default=10)
    parser.add_argument(
        "--privacy-redaction-manifest",
        type=Path,
        help="Frozen JSON produced by serve_qualitative_redaction.py."
    )
    args = parser.parse_args()

    privacy_payload = None
    if args.privacy_redaction_manifest:
        privacy = load_privacy_redactions(args.privacy_redaction_manifest)
        privacy_payload = privacy["payload"]
        PRIVACY_REDACTIONS = privacy["regions"]

    args.out.mkdir(parents=True, exist_ok=False)
    source_dir = args.out / "selected_source_frames"
    source_dir.mkdir()
    rows = read_csv(args.run_root / "worker_state_image_rows.csv")
    cache = CacheStore(args.run_root / "prediction_cache")
    cases = build_cases(rows, args.rate)
    selected = select_cases(cases, cache)
    if privacy_payload:
        reviewed_paths = {
            path_key(record["original_image_path"])
            for record in privacy_payload.get("images", {}).values()
            if record.get("reviewed") and record.get("original_image_path")
        }
        missing = [item["image_path"] for item in selected if path_key(item["image_path"]) not in reviewed_paths]
        if missing:
            raise ValueError(
                "every selected qualitative panel must be reviewed in the frozen privacy manifest: "
                + ", ".join(missing)
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.55), constrained_layout=False)
    figure.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.085, wspace=0.04, hspace=0.16)
    for ax, item in zip(axes.flat, selected):
        render_panel(ax, item, cache)

    legend = [
        Line2D([0], [0], color=STATE_COLORS["safe"], lw=2.4, label="person: safe"),
        Line2D([0], [0], color=STATE_COLORS["unsafe"], lw=2.4, label="person: unsafe"),
        Line2D([0], [0], color=STATE_COLORS["review"], lw=2.4, label="person: review"),
        Line2D([0], [0], color=PPE_COLORS[0], lw=1.4, label="helmet"),
        Line2D([0], [0], color=PPE_COLORS[1], lw=1.4, label="no helmet"),
        Line2D([0], [0], color=PPE_COLORS[4], lw=1.4, label="vest"),
        Line2D([0], [0], color=PPE_COLORS[2], lw=1.4, label="no vest"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=6.5,
        handlelength=1.8,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.012),
    )
    figure.savefig(args.out / "fig5_real_qualitative.pdf", dpi=300, bbox_inches="tight")
    figure.savefig(args.out / "fig5_real_qualitative.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    manifest_rows = []
    for item in selected:
        source_name = f"panel_{item['panel']}_{Path(item['image_path']).name}"
        write_image(source_dir / source_name, read_image(Path(item["image_path"])))
        manifest_rows.append(manifest_row(item, source_name))
    write_csv(args.out / "qualitative_selection.csv", manifest_rows)
    manifest = {
        "run_root": str(args.run_root.resolve()),
        "rate": args.rate,
        "split": "source-disjoint evaluation only",
        "selection_timing": "case definitions fixed before rendering",
        "rendering": (
            "context crops from original frames; predicted person boxes are colored by "
            "RC-WSSI state; only the highest-confidence assigned PPE box per class is shown"
        ),
        "class_semantics": CLASS_NAMES,
        "gpu_inference": False,
        "panels": manifest_rows,
    }
    if privacy_payload:
        manifest["privacy_redaction"] = {
            "manifest": str(args.privacy_redaction_manifest.resolve()),
            "frozen": True,
            "reviewed_panels": len(selected),
            "redacted_regions": sum(
                len(PRIVACY_REDACTIONS.get(path_key(item["image_path"]), []))
                for item in selected
            ),
            "method": "manual-review coordinates with strong pixelation before rendering",
        }
    (args.out / "qualitative_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )
    print(json.dumps(manifest_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
