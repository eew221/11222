"""Validation helpers for the audited four-way source-disjoint protocol."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


EXPECTED_NAMES = {
    0: "helmet",
    1: "no_helmet",
    2: "no_reflective_vest",
    3: "person",
    4: "reflective_vest",
}
EXPECTED_ROLES = {"train", "detector_val", "calibration", "test"}


def normalize_names(names) -> dict[int, str]:
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(key): str(value) for key, value in (names or {}).items()}


def normalized_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").casefold()


def validate_protocol_cell(cell_root: Path, run_dir: Path, tag: str) -> dict:
    protocol_path = cell_root / "protocol.json"
    yaml_path = cell_root / "detector.yaml"
    marker_path = run_dir / "source_disjoint_training_complete.json"
    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"
    required = (protocol_path, yaml_path, marker_path, args_path, results_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete audited training output for {tag}: {missing}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != "source_disjoint_four_way_v1":
        raise RuntimeError(f"unexpected protocol for {tag}: {protocol.get('protocol')}")
    roles = protocol.get("roles", {})
    if set(roles) != EXPECTED_ROLES:
        raise RuntimeError(f"role mismatch for {tag}: {sorted(roles)}")
    for section in protocol.get("leakage_audit", {}).values():
        if any(int(value) != 0 for value in section.values()):
            raise RuntimeError(f"nonzero protocol leakage audit for {tag}")

    detector_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if detector_config.get("train") != "images/train":
        raise RuntimeError(f"detector train split mismatch for {tag}")
    if detector_config.get("val") != "images/detector_val":
        raise RuntimeError(f"checkpoint validation split mismatch for {tag}")
    if normalize_names(detector_config.get("names")) != EXPECTED_NAMES:
        raise RuntimeError(f"detector YAML class map mismatch for {tag}")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_best = run_dir / "weights" / "best.pt"
    checks = {
        "status": marker.get("status") == "complete",
        "tag": marker.get("tag") == tag,
        "protocol": marker.get("protocol") == "source_disjoint_four_way_v1",
        "checkpoint_selection_split": marker.get("checkpoint_selection_split") == "detector_val",
        "data_yaml": normalized_path(Path(marker.get("data_yaml", "")))
        == normalized_path(yaml_path),
        "best": normalized_path(Path(marker.get("best", "")))
        == normalized_path(expected_best),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid completion marker for {tag}: {failed}")

    training_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
    if normalized_path(Path(str(training_args.get("data", "")))) != normalized_path(yaml_path):
        raise RuntimeError(f"saved training data path mismatch for {tag}")
    if str(training_args.get("name", "")) != run_dir.name:
        raise RuntimeError(f"saved training run name mismatch for {tag}")
    if not expected_best.is_file():
        raise FileNotFoundError(f"missing selected best checkpoint for {tag}: {expected_best}")
    return protocol


def validate_existing_cache(cache_path: Path, tag: str, weights: Path) -> None:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    checks = {
        "tag": payload.get("tag") == tag,
        "protocol": payload.get("protocol") == "source_disjoint_four_way_v1",
        "weights": normalized_path(Path(payload.get("weights", "")))
        == normalized_path(weights),
        "roles": {item.get("role") for item in payload.get("images", [])}
        == {"calibration", "test"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"refusing to trust or overwrite incompatible cache {cache_path}: {failed}"
        )
