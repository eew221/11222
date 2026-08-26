"""Combine predeclared final-test prediction caches for a blind audit.

The output contains only final-test records.  It records every source cache
hash and rejects overlapping filename groups, so the human audit cannot be
quietly restricted to a favorable fold after prediction results are known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cache(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read cache {path}: {error}") from error
    if not isinstance(payload.get("images"), list):
        raise ValueError(f"{path}: missing image records")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, nargs="+", required=True, help="Frozen cache(s), one per fold.")
    parser.add_argument("--out", type=Path, required=True, help="New combined cache path.")
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit cache: {out}")

    sources, records, source_groups, image_paths = [], [], set(), set()
    for raw_path in args.cache:
        path = raw_path.resolve()
        payload = read_cache(path)
        test_records = [record for record in payload["images"] if record.get("role") == "test"]
        if not test_records:
            raise ValueError(f"{path}: no final-test records")
        groups = {str(record.get("source_group", "")) for record in test_records}
        if "" in groups:
            raise ValueError(f"{path}: test record has no source group")
        overlap = source_groups & groups
        if overlap:
            raise ValueError(f"filename groups occur in more than one cache: {sorted(overlap)}")
        duplicate_paths = image_paths & {str(record.get("image_path", "")) for record in test_records}
        if duplicate_paths:
            raise ValueError(f"test image occurs in more than one cache: {sorted(duplicate_paths)[:3]}")
        missing = [record["image_path"] for record in test_records if not Path(record["image_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{path}: {len(missing)} audit images are inaccessible; first is {missing[0]}")
        sources.append({"path": str(path), "sha256": sha256(path), "tag": payload.get("tag", "unknown"),
                        "test_records": len(test_records), "source_groups": sorted(groups)})
        records.extend(test_records)
        source_groups.update(groups)
        image_paths.update(str(record["image_path"]) for record in test_records)

    output = {
        "tag": "r10_s0_all_folds_pre_frozen_end_to_end_audit",
        "protocol": "combined_final_test_prediction_cache_for_blinded_detector_output_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_caches": sources,
        "declared_source_groups": sorted(source_groups),
        "images": records,
        "scope": "10 percent YOLOv8s, fixed seed 0, all three predeclared final-test folds",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "images": len(records), "source_groups": sorted(source_groups), "cache_hashes": sources}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
