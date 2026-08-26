"""Serve one blinded open-set human PPE-owner annotation pass locally."""

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
    "audit_id", "evidence_id", "evidence_class_id", "evidence_class",
    "candidate_person_ids", "all_visible_person_ids", "assigned_person_id",
    "assignment_confidence", "occluded_or_ambiguous", "notes",
)
SPECIAL = {"NONE", "AMBIGUOUS", "OUTSIDE_DETECTED_PERSON_SET", "FALSE_DETECTION"}
PERSON = re.compile(r"^P[1-9][0-9]*$")
CONFIDENCE = {"", "low", "medium", "high"}
OCCLUSION = {"", "no", "yes"}

PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>开放候选 PPE-工人归属盲标</title><style>
body{margin:0;font:15px/1.45 system-ui,"Microsoft YaHei",sans-serif;color:#17211e;background:#f4f7f5}header{height:58px;padding:0 20px;display:flex;align-items:center;gap:16px;background:#164a41;color:#fff}header b{font-size:18px}.grow{flex:1}.status{font-size:13px}.layout{display:grid;grid-template-columns:235px minmax(440px,1fr) 380px;height:calc(100vh - 58px)}aside,main{min-height:0;background:#fff}.left{border-right:1px solid #d5dfda}.right{border-left:1px solid #d5dfda;overflow:auto}.head{padding:14px;border-bottom:1px solid #d5dfda}.images{overflow:auto;height:calc(100% - 66px);padding:8px}.imgbtn{display:block;width:100%;text-align:left;padding:9px;margin:0 0 6px;border:1px solid #d5dfda;background:#fff;cursor:pointer}.imgbtn.active{border-left:5px solid #147766;background:#e6f2ed}.imgbtn.done{color:#147766}.viewer{display:flex;flex-direction:column}.viewerhead{padding:10px 14px;border-bottom:1px solid #d5dfda;font-weight:700}.canvas{flex:1;overflow:auto;padding:16px;background:#1d2522;text-align:center}.canvas img{max-width:100%;height:auto}.card{margin:10px;padding:10px;border:1px solid #d5dfda;border-left:5px solid #d89b1a}.card.done{border-left-color:#147766}.top{display:flex;justify-content:space-between}.cls{color:#a94b26;font-weight:700}.hint{font-size:12px;color:#65736d;margin:7px 0}.buttons{display:flex;flex-wrap:wrap;gap:6px}.buttons button,header button{padding:6px 9px;border:1px solid #b9c9c1;background:#fff;cursor:pointer}.buttons button.selected{background:#147766;border-color:#147766;color:#fff}.buttons button.outside{border-style:dashed}.buttons button.special.selected{background:#a94b26;border-color:#a94b26}.fields{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}.fields label{font-size:12px;color:#50605a}.fields select,.fields input{box-sizing:border-box;width:100%;margin-top:2px;padding:5px;border:1px solid #c7d2cc}.note{grid-column:1/-1}.banner{padding:9px 14px;background:#fff4e9;color:#8a3d1c;font-size:13px}.toast{position:fixed;right:16px;bottom:16px;padding:9px 12px;background:#1b2c27;color:#fff;display:none}.toast.error{background:#a53730}@media(max-width:1100px){.layout{grid-template-columns:190px minmax(340px,1fr) 330px}}
</style></head><body><header><b>开放候选 PPE-工人归属盲标</b><span id="who"></span><span class="grow"></span><span id="progress" class="status"></span><button id="help">标注说明</button><button id="freeze">冻结本轮</button><button id="export">导出 CSV</button></header><div id="locked" class="banner" hidden>本轮已冻结，不能再修改。</div><section class="layout"><aside class="left"><div class="head"><b>图像</b><div id="imageCount" class="status"></div></div><div id="images" class="images"></div></aside><main class="viewer"><div id="title" class="viewerhead"></div><div class="canvas"><img id="image"></div></main><aside class="right"><div class="head"><b>PPE 归属</b><div class="status">选择任意画面中可见人员。浅色虚线为原几何候选，仅作事后统计，不是推荐答案。</div></div><div id="rows"></div></aside></section><div id="toast" class="toast"></div><script>
const S={rows:[],images:[],idx:0,locked:false,specialOptions:['NONE','AMBIGUOUS']};const $=id=>document.getElementById(id);const key=r=>r.audit_id+'|'+r.evidence_id;const byImage=()=>S.rows.reduce((m,r)=>((m[r.audit_id]??=[]).push(r),m),{});const done=r=>!!r.assigned_person_id;async function api(url,opt={}){const r=await fetch(url,opt),d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.error||'请求失败');return d}function note(t,bad=false){const n=$('toast');n.textContent=t;n.className='toast'+(bad?' error':'');n.style.display='block';clearTimeout(note.t);note.t=setTimeout(()=>n.style.display='none',2600)}function cur(){return S.images[S.idx]}function imageDone(id){const a=byImage()[id]||[];return a.length&&a.every(done)}function renderImages(){const box=$('images');box.innerHTML='';S.images.forEach((x,i)=>{let b=document.createElement('button');b.className='imgbtn'+(i===S.idx?' active':'')+(imageDone(x.audit_id)?' done':'');b.innerHTML='<b>'+x.audit_id+'</b><br><span class="status">'+x.source_group+' | fold '+x.fold+' | '+x.evidence_count+' PPE'+(imageDone(x.audit_id)?' | 已完成':'')+'</span>';b.onclick=()=>{S.idx=i;render()};box.append(b)});$('imageCount').textContent='共 '+S.images.length+' 张'}function set(r,v){r.assigned_person_id=v;save(r)}function specialLabel(v){return ({NONE:'NONE（无归属）',AMBIGUOUS:'AMBIGUOUS（无法判断）',OUTSIDE_DETECTED_PERSON_SET:'检测人员框外的真实 owner',FALSE_DETECTION:'FALSE DETECTION（误检 PPE）'})[v]||v}function button(r,v,kind=''){let b=document.createElement('button');b.textContent=S.specialOptions.includes(v)?specialLabel(v):v;b.className=kind+(r.assigned_person_id===v?' selected':'');b.disabled=S.locked;b.onclick=()=>set(r,v);return b}function field(r,label,prop,vals){let l=document.createElement('label');l.textContent=label;let e=prop==='notes'?document.createElement('input'):document.createElement('select');if(prop==='notes'){e.value=r[prop]||'';e.maxLength=1000}else vals.forEach(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;o.selected=r[prop]===v;e.append(o)});e.disabled=S.locked;e.onchange=()=>{r[prop]=e.value;save(r)};l.append(e);return l}function renderRows(){let box=$('rows');box.innerHTML='';let rows=byImage()[cur().audit_id]||[];rows.forEach(r=>{let c=document.createElement('article');c.className='card'+(done(r)?' done':'');c.innerHTML='<div class="top"><b>'+r.evidence_id+'</b><span class="cls">'+r.evidence_class+'</span></div>';let h=document.createElement('div');h.className='hint';h.textContent='原几何候选：'+(r.candidate_person_ids||'无')+'；可选全部可见人员：';c.append(h);let buttons=document.createElement('div');buttons.className='buttons';let candidates=new Set((r.candidate_person_ids||'').split('|').filter(Boolean));(r.all_visible_person_ids||'').split('|').filter(Boolean).forEach(p=>buttons.append(button(r,p,candidates.has(p)?'':'outside')));S.specialOptions.forEach(v=>buttons.append(button(r,v,'special')));c.append(buttons);let f=document.createElement('div');f.className='fields';f.append(field(r,'人工置信度','assignment_confidence',[['','未设置'],['high','高'],['medium','中'],['low','低']]));f.append(field(r,'遮挡/歧义','occluded_or_ambiguous',[['','未设置'],['no','否'],['yes','是']]));let n=field(r,'备注（可空）','notes',[]);n.className='note';f.append(n);c.append(f);box.append(c)})}function render(){let x=cur();$('title').textContent=x.audit_id+' | '+x.source_group+' | fold '+x.fold;$('image').src='/images/'+encodeURIComponent(x.image);$('locked').hidden=!S.locked;$('freeze').disabled=S.locked;renderImages();renderRows()}async function save(r){try{let d=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({updates:[r]})});$('progress').textContent=d.progress.complete+' / '+d.progress.total;render()}catch(e){note(e.message,true)}}async function boot(){try{let d=await api('/api/bootstrap');Object.assign(S,{rows:d.rows,images:d.images,locked:d.locked,specialOptions:d.special_options});$('who').textContent='盲标 '+d.annotator;$('progress').textContent=d.progress.complete+' / '+d.progress.total;render()}catch(e){note(e.message,true)}}$('export').onclick=()=>location.assign('/api/export');$('freeze').onclick=async()=>{if(!confirm('确认冻结？冻结后不能修改。'))return;try{let d=await api('/api/finalize',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});S.locked=true;$('progress').textContent=d.record.completed_evidence_rows+' / '+d.record.completed_evidence_rows;render();note('本轮已冻结')}catch(e){note(e.message,true)}};$('help').onclick=()=>alert('每个 E 框请选择实际佩戴该 PPE 的任意可见 P 人员。虚线按钮只是原几何候选，不代表推荐答案。若没有显示人员是 owner，选 NONE；看不清或无法可靠区分，选 AMBIGUOUS。检测输出审计还可选“检测人员框外的真实 owner”或“误检 PPE”。请独立完成本轮，不查看模型或其他专家答案。');boot();
</script></body></html>"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: unexpected CSV header")
        return [{field: row.get(field, "") for field in FIELDS} for row in reader]


def atomic_write(path: Path, rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


class Store:
    def __init__(self, root: Path, annotator: str) -> None:
        self.root, self.annotator = root.resolve(), annotator
        self.csv_path = self.root / f"annotator_{annotator}" / "evidence_assignment.csv"
        self.lock_path = self.csv_path.parent / "ANNOTATION_FINALIZED.json"
        self.images_dir = (self.root / "annotated_images").resolve()
        protocol_path = self.root / "audit_manifest.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
        self.special_options = (
            ["NONE", "AMBIGUOUS", "OUTSIDE_DETECTED_PERSON_SET", "FALSE_DETECTION"]
            if protocol.get("protocol") == "pre-frozen_end_to_end_detector_output_human_owner_audit_v1"
            else ["NONE", "AMBIGUOUS"]
        )
        with (self.root / "audit_image_manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            self.manifest = {row["audit_id"]: row for row in csv.DictReader(stream)}
        if not self.csv_path.is_file() or not self.images_dir.is_dir() or not self.manifest:
            raise FileNotFoundError("audit root is incomplete")
        self.images = {path.name.split("_", 1)[0]: path.name for path in self.images_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if set(self.images) != set(self.manifest):
            raise ValueError("rendered audit images do not match manifest")

    def rows(self) -> list[dict[str, str]]:
        return read_csv(self.csv_path)

    def locked(self) -> bool:
        return self.lock_path.exists()

    def bootstrap(self) -> dict:
        rows = self.rows()
        order = sorted(self.manifest, key=lambda value: int(value[1:]))
        evidence_counts = {}
        for row in rows:
            evidence_counts[row["audit_id"]] = evidence_counts.get(row["audit_id"], 0) + 1
        return {"annotator": self.annotator, "locked": self.locked(), "rows": rows, "special_options": self.special_options,
                "images": [{"audit_id": aid, "image": self.images[aid], "source_group": self.manifest[aid]["source_group"], "fold": self.manifest[aid]["fold"], "evidence_count": self.manifest[aid].get("evidence_count", evidence_counts[aid])} for aid in order],
                "progress": {"complete": sum(bool(row["assigned_person_id"]) for row in rows), "total": len(rows)}}

    def update(self, updates: list[dict]) -> dict:
        if self.locked():
            raise PermissionError("this annotation pass is frozen")
        rows = self.rows()
        indexed = {(row["audit_id"], row["evidence_id"]): row for row in rows}
        for update in updates:
            key = (str(update.get("audit_id", "")).strip(), str(update.get("evidence_id", "")).strip())
            if key not in indexed:
                raise ValueError(f"unknown evidence row {key}")
            row = indexed[key]
            assignment = str(update.get("assigned_person_id", "")).strip().upper()
            visible = set(filter(None, row["all_visible_person_ids"].split("|")))
            if assignment and (not PERSON.fullmatch(assignment) or assignment not in visible) and assignment not in SPECIAL:
                raise ValueError(f"{key}: assignment must be a visible Pn, NONE, or AMBIGUOUS")
            confidence = str(update.get("assignment_confidence", "")).strip().lower()
            occlusion = str(update.get("occluded_or_ambiguous", "")).strip().lower()
            notes = str(update.get("notes", "")).strip()
            if confidence not in CONFIDENCE or occlusion not in OCCLUSION or len(notes) > 1000:
                raise ValueError(f"{key}: invalid annotation details")
            row.update({"assigned_person_id": assignment, "assignment_confidence": confidence, "occluded_or_ambiguous": occlusion, "notes": notes})
        atomic_write(self.csv_path, rows)
        return {"complete": sum(bool(row["assigned_person_id"]) for row in rows), "total": len(rows)}

    def finalize(self) -> dict:
        if self.locked():
            raise PermissionError("this annotation pass is already frozen")
        rows = self.rows()
        if any(not row["assigned_person_id"] for row in rows):
            raise ValueError("cannot freeze with incomplete rows")
        record = {"annotation_status": "frozen_blind_open_set_human_pass", "annotator": self.annotator,
                  "completed_evidence_rows": len(rows), "csv_sha256": hashlib.sha256(self.csv_path.read_bytes()).hexdigest(),
                  "frozen_utc": datetime.now(timezone.utc).isoformat()}
        self.lock_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record


def handler(store: Store):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")

        def reply(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def file(self, path: Path, download: bool = False) -> None:
            data = path.read_bytes(); mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", mime); self.send_header("Content-Length", str(len(data)))
            if download: self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers(); self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path in {"/", "/index.html"}:
                    data = PAGE.encode("utf-8"); self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                elif path == "/api/bootstrap": self.reply(store.bootstrap())
                elif path == "/api/export": self.file(store.csv_path, True)
                elif path.startswith("/images/"):
                    image = (store.images_dir / unquote(path.removeprefix("/images/"))).resolve()
                    if image.parent != store.images_dir or not image.is_file(): raise FileNotFoundError(image.name)
                    self.file(image)
                else: self.send_error(HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, ValueError) as error: self.reply({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
                if urlparse(self.path).path == "/api/save": self.reply({"ok": True, "progress": store.update(body.get("updates", []))})
                elif urlparse(self.path).path == "/api/finalize": self.reply({"ok": True, "record": store.finalize()})
                else: self.send_error(HTTPStatus.NOT_FOUND)
            except PermissionError as error: self.reply({"error": str(error)}, HTTPStatus.FORBIDDEN)
            except (ValueError, json.JSONDecodeError) as error: self.reply({"error": str(error)}, HTTPStatus.BAD_REQUEST)
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--annotator", choices=("A", "B", "C"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    store = Store(args.audit_root, args.annotator)
    server = ThreadingHTTPServer((args.host, args.port), handler(store))
    print(f"Annotator {args.annotator}: http://{args.host}:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__":
    main()
