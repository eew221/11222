"""Serve a blinded local PPE-to-worker association annotation pass.

The server exposes only the selected annotator's CSV template and the rendered
audit images. It deliberately has no route for sealed method assignments or the
other annotator's responses.
"""

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
    "evidence_class_id",
    "evidence_class",
    "candidate_person_ids",
    "assigned_person_id",
    "assignment_confidence",
    "occluded_or_ambiguous",
    "notes",
)
SPECIAL_ASSIGNMENTS = {"NONE", "AMBIGUOUS"}
CONFIDENCE = {"", "low", "medium", "high"}
OCCLUSION = {"", "no", "yes"}
PERSON_ID = re.compile(r"^P[1-9][0-9]*$")


def parse_args() -> argparse.Namespace:
    default_root = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "manual_worker_ppe_association_audit_20260810_v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=default_root)
    parser.add_argument("--annotator", choices=("A", "B"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page", type=Path, default=None,
                        help="optional annotation page, relative to --audit-root")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: unexpected CSV header")
        return [{field: row.get(field, "") for field in FIELDS} for row in reader]


def atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def csv_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuditStore:
    def __init__(self, root: Path, annotator: str) -> None:
        self.root = root.resolve()
        self.annotator = annotator
        self.annotator_dir = self.root / f"annotator_{annotator}"
        self.csv_path = self.annotator_dir / "evidence_assignment.csv"
        self.lock_path = self.annotator_dir / "ANNOTATION_FINALIZED.json"
        self.images_dir = (self.root / "annotated_images").resolve()
        self.page_path = self.root / "annotation_app.html"
        if not self.csv_path.is_file() or not self.images_dir.is_dir() or not self.page_path.is_file():
            raise FileNotFoundError("audit root is missing its template, images, or annotation_app.html")
        self.manifest = self._read_manifest()
        self.images = self._image_index()

    def _read_manifest(self) -> dict[str, dict[str, str]]:
        path = self.root / "audit_image_manifest.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return {row["audit_id"]: row for row in csv.DictReader(stream)}

    def _image_index(self) -> dict[str, str]:
        images: dict[str, str] = {}
        for path in self.images_dir.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            audit_id = path.name.split("_", maxsplit=1)[0]
            if audit_id in self.manifest:
                images[audit_id] = path.name
        missing = set(self.manifest) - set(images)
        if missing:
            raise ValueError(f"no rendered image for {sorted(missing)[:3]}")
        return images

    def locked(self) -> bool:
        return self.lock_path.exists()

    def rows(self) -> list[dict[str, str]]:
        return read_csv(self.csv_path)

    @staticmethod
    def complete(row: dict[str, str]) -> bool:
        return bool(row["assigned_person_id"].strip())

    def bootstrap(self) -> dict:
        rows = self.rows()
        audit_ids = sorted(self.manifest, key=lambda value: int(value[1:]))
        return {
            "annotator": self.annotator,
            "locked": self.locked(),
            "rows": rows,
            "images": [
                {
                    "audit_id": audit_id,
                    "image": self.images[audit_id],
                    "source_group": self.manifest[audit_id]["source_group"],
                    "fold": self.manifest[audit_id]["fold"],
                    "person_count": self.manifest[audit_id]["person_count"],
                    "evidence_count": self.manifest[audit_id]["evidence_count"],
                }
                for audit_id in audit_ids
            ],
            "progress": {"complete": sum(self.complete(row) for row in rows), "total": len(rows)},
        }

    def update(self, updates: list[dict]) -> dict:
        if self.locked():
            raise PermissionError("this annotation pass is frozen")
        rows = self.rows()
        by_key = {(row["audit_id"], row["evidence_id"]): row for row in rows}
        for update in updates:
            key = (str(update.get("audit_id", "")).strip(), str(update.get("evidence_id", "")).strip())
            if key not in by_key:
                raise ValueError(f"unknown evidence row {key}")
            row = by_key[key]
            assignment = str(update.get("assigned_person_id", "")).strip().upper()
            candidates = set(filter(None, row["candidate_person_ids"].split("|")))
            if assignment and assignment not in candidates | SPECIAL_ASSIGNMENTS:
                raise ValueError(f"{key}: assignment must be a listed candidate, NONE, or AMBIGUOUS")
            confidence = str(update.get("assignment_confidence", "")).strip().lower()
            occlusion = str(update.get("occluded_or_ambiguous", "")).strip().lower()
            notes = str(update.get("notes", "")).strip()
            if confidence not in CONFIDENCE or occlusion not in OCCLUSION:
                raise ValueError(f"{key}: invalid confidence or occlusion value")
            if len(notes) > 1000:
                raise ValueError(f"{key}: notes exceed 1000 characters")
            row["assigned_person_id"] = assignment
            row["assignment_confidence"] = confidence
            row["occluded_or_ambiguous"] = occlusion
            row["notes"] = notes
        atomic_write_csv(self.csv_path, rows)
        return {"complete": sum(self.complete(row) for row in rows), "total": len(rows)}

    def finalize(self) -> dict:
        if self.locked():
            raise PermissionError("this annotation pass is already frozen")
        rows = self.rows()
        incomplete = [f"{row['audit_id']}/{row['evidence_id']}" for row in rows if not self.complete(row)]
        if incomplete:
            raise ValueError(f"cannot freeze with {len(incomplete)} incomplete rows")
        record = {
            "annotation_status": "frozen_human_pass",
            "annotator": self.annotator,
            "completed_evidence_rows": len(rows),
            "csv_sha256": csv_sha256(self.csv_path),
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.lock_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


def make_handler(store: AuditStore):
    class Handler(SimpleHTTPRequestHandler):
        server_version = "PPEAssociationAudit/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

        def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path: Path, download: bool = False) -> None:
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if mime.startswith("text/"):
                mime = f"{mime}; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            self.wfile.write(data)

        def read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2_000_000:
                raise ValueError("request is too large")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path in {"/", "/index.html"}:
                    self.send_file(store.page_path)
                elif path == "/api/bootstrap":
                    self.send_json(store.bootstrap())
                elif path == "/api/export":
                    self.send_file(store.csv_path, download=True)
                elif path.startswith("/images/"):
                    image_name = unquote(path.removeprefix("/images/"))
                    image_path = (store.images_dir / image_name).resolve()
                    if image_path.parent != store.images_dir or not image_path.is_file():
                        raise FileNotFoundError(image_name)
                    self.send_file(image_path)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, ValueError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self.read_body()
                if path == "/api/save":
                    updates = payload.get("updates")
                    if not isinstance(updates, list) or not updates:
                        raise ValueError("updates must be a non-empty list")
                    self.send_json({"ok": True, "progress": store.update(updates), "locked": False})
                elif path == "/api/finalize":
                    self.send_json({"ok": True, "record": store.finalize(), "locked": True})
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except PermissionError as error:
                self.send_json({"error": str(error)}, HTTPStatus.FORBIDDEN)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    args = parse_args()
    store = AuditStore(args.audit_root, args.annotator)
    if args.page is not None:
        page = (store.root / args.page).resolve()
        if page.parent != store.root or not page.is_file():
            raise FileNotFoundError(f"annotation page is not under audit root: {page}")
        store.page_path = page
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Annotator {args.annotator}: http://{args.host}:{args.port}")
    print(f"Audit root: {store.root}")
    print("The server exposes no sealed reference or other-annotator response.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
