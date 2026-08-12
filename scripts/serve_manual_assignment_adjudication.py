"""Serve a third-expert adjudication page for disagreements only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


FIELDS = (
    "audit_id",
    "evidence_id",
    "evidence_class",
    "candidate_person_ids",
    "annotator_a",
    "annotator_b",
    "adjudicated_assignment",
    "decision_confidence",
    "occluded_or_ambiguous",
    "rationale",
)
SPECIAL_ASSIGNMENTS = {"NONE", "AMBIGUOUS"}
VALID_ASSIGNMENT = re.compile(r"^(P[1-9][0-9]*|NONE|AMBIGUOUS)$")
CONFIDENCE = {"", "low", "medium", "high"}
OCCLUSION = {"", "no", "yes"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=root / "experiments" / "manual_worker_ppe_association_audit_20260810_v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdjudicationStore:
    def __init__(self, agreement_root: Path, audit_root: Path) -> None:
        self.agreement_root = agreement_root.resolve()
        self.audit_root = audit_root.resolve()
        self.out = self.agreement_root / "adjudication"
        self.out.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out / "adjudication.csv"
        self.lock_path = self.out / "ADJUDICATION_FINALIZED.json"
        self.page_path = self.agreement_root / "adjudication_app.html"
        self.images_dir = (self.audit_root / "annotated_images").resolve()
        self.manifest = self._read_manifest()
        self._ensure_template()

    def _read_manifest(self) -> dict[str, dict[str, str]]:
        rows = read_csv(self.audit_root / "audit_image_manifest.csv")
        return {row["audit_id"]: row for row in rows}

    def _ensure_template(self) -> None:
        if self.csv_path.exists():
            return
        pairwise = read_csv(self.agreement_root / "pairwise_assignment_rows.csv")
        source_rows = {
            (row["audit_id"], row["evidence_id"]): row
            for row in read_csv(self.audit_root / "annotator_A" / "evidence_assignment.csv")
        }
        rows = []
        for pair in pairwise:
            if pair["exact_agreement"] == "1":
                continue
            source = source_rows[(pair["audit_id"], pair["evidence_id"])]
            rows.append({
                "audit_id": pair["audit_id"],
                "evidence_id": pair["evidence_id"],
                "evidence_class": source["evidence_class"],
                "candidate_person_ids": source["candidate_person_ids"],
                "annotator_a": pair["annotator_a"],
                "annotator_b": pair["annotator_b"],
                "adjudicated_assignment": "",
                "decision_confidence": "",
                "occluded_or_ambiguous": "",
                "rationale": "",
            })
        if not rows:
            raise ValueError("no disagreements found for adjudication")
        atomic_write_csv(self.csv_path, rows)

    def rows(self) -> list[dict[str, str]]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("unexpected adjudication CSV header")
            return [{field: row.get(field, "") for field in FIELDS} for row in reader]

    def locked(self) -> bool:
        return self.lock_path.exists()

    def _image_name(self, audit_id: str) -> str:
        candidates = list(self.images_dir.glob(f"{audit_id}_*"))
        images = [path for path in candidates if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(images) != 1:
            raise FileNotFoundError(f"image for {audit_id}")
        return images[0].name

    def bootstrap(self) -> dict:
        rows = self.rows()
        audit_ids = sorted({row["audit_id"] for row in rows}, key=lambda value: int(value[1:]))
        return {
            "locked": self.locked(),
            "rows": rows,
            "images": [{"audit_id": audit_id, "image": self._image_name(audit_id), "source_group": self.manifest[audit_id]["source_group"], "fold": self.manifest[audit_id]["fold"]} for audit_id in audit_ids],
            "progress": {"complete": sum(bool(row["adjudicated_assignment"]) for row in rows), "total": len(rows)},
        }

    def update(self, updates: list[dict]) -> dict:
        if self.locked():
            raise PermissionError("adjudication is frozen")
        rows = self.rows()
        by_key = {(row["audit_id"], row["evidence_id"]): row for row in rows}
        for update in updates:
            key = (str(update.get("audit_id", "")).strip(), str(update.get("evidence_id", "")).strip())
            if key not in by_key:
                raise ValueError(f"unknown disagreement row {key}")
            row = by_key[key]
            assignment = str(update.get("adjudicated_assignment", "")).strip().upper()
            candidates = set(filter(None, row["candidate_person_ids"].split("|")))
            if assignment and (not VALID_ASSIGNMENT.fullmatch(assignment) or assignment not in candidates | SPECIAL_ASSIGNMENTS):
                raise ValueError(f"{key}: invalid adjudicated assignment")
            confidence = str(update.get("decision_confidence", "")).strip().lower()
            occlusion = str(update.get("occluded_or_ambiguous", "")).strip().lower()
            rationale = str(update.get("rationale", "")).strip()
            if confidence not in CONFIDENCE or occlusion not in OCCLUSION or len(rationale) > 2000:
                raise ValueError(f"{key}: invalid adjudication fields")
            row["adjudicated_assignment"] = assignment
            row["decision_confidence"] = confidence
            row["occluded_or_ambiguous"] = occlusion
            row["rationale"] = rationale
        atomic_write_csv(self.csv_path, list(by_key.values()))
        return {"complete": sum(bool(row["adjudicated_assignment"]) for row in by_key.values()), "total": len(rows)}

    def finalize(self) -> dict:
        if self.locked():
            raise PermissionError("adjudication is already frozen")
        rows = self.rows()
        incomplete = [f"{row['audit_id']}/{row['evidence_id']}" for row in rows if not row["adjudicated_assignment"]]
        if incomplete:
            raise ValueError(f"cannot freeze with {len(incomplete)} unresolved disagreements")
        record = {"annotation_status": "third_expert_adjudication_frozen", "evidence_boxes": len(rows), "csv_sha256": sha256(self.csv_path), "frozen_utc": datetime.now(timezone.utc).isoformat()}
        self.lock_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


def make_handler(store: AdjudicationStore):
    class Handler(SimpleHTTPRequestHandler):
        server_version = "PPEAssociationAdjudication/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

        def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path: Path) -> None:
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if mime.startswith("text/"):
                mime += "; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path in {"/", "/index.html"}:
                    self.send_file(store.page_path)
                elif path == "/api/bootstrap":
                    self.send_json(store.bootstrap())
                elif path == "/api/export":
                    self.send_file(store.csv_path)
                elif path.startswith("/images/"):
                    name = unquote(path.removeprefix("/images/"))
                    image = (store.images_dir / name).resolve()
                    if image.parent != store.images_dir or not image.is_file():
                        raise FileNotFoundError(name)
                    self.send_file(image)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            try:
                payload = self.read_body()
                if urlparse(self.path).path == "/api/save":
                    self.send_json({"ok": True, "progress": store.update(payload.get("updates", []))})
                elif urlparse(self.path).path == "/api/finalize":
                    self.send_json({"ok": True, "record": store.finalize()})
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except PermissionError as error:
                self.send_json({"error": str(error)}, HTTPStatus.FORBIDDEN)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    args = parse_args()
    store = AdjudicationStore(args.agreement_root, args.audit_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Adjudication: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
