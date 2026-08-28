"""Serve the small C-expert adjudication pass for worker-state disagreements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

FIELDS = ("audit_id", "person_id", "source_group", "image_name", "annotator_A_helmet", "annotator_A_vest", "annotator_A_overall", "annotator_B_helmet", "annotator_B_vest", "annotator_B_overall", "A_confidence", "B_confidence", "final_helmet_state", "final_vest_state", "final_overall_state", "adjudicator_confidence", "adjudication_notes")
STATES = {"SAFE", "UNSAFE", "REVIEW"}
CONFIDENCE = {"", "high", "medium", "low"}

PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三专家工人状态仲裁</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7f5;color:#17211e;font:14px/1.45 "Microsoft YaHei",Arial,sans-serif}button,select,input{font:inherit}button{padding:7px 11px;border:1px solid #c5d2cb;border-radius:4px;background:#fff;cursor:pointer}button:disabled,select:disabled,input:disabled{background:#edf1ef;color:#89948e;cursor:not-allowed}.top{height:62px;padding:0 18px;display:flex;align-items:center;gap:16px;background:#164a41;color:#fff}.top b{font-size:17px}.grow{flex:1}.status{font-size:12px}.layout{height:calc(100vh - 62px);display:grid;grid-template-columns:310px minmax(460px,1fr) 500px}.left,.viewer,.right{min-height:0;background:#fff}.left{border-right:1px solid #d4dfd9}.right{border-left:1px solid #d4dfd9;overflow:auto}.head{padding:14px;border-bottom:1px solid #d4dfd9}.muted{font-size:12px;color:#63716a}.items{padding:8px;overflow:auto;height:calc(100% - 68px)}.item{display:block;width:100%;text-align:left;margin-bottom:6px}.item.active{border-left:5px solid #147766;background:#e4f1ed}.viewer{display:flex;flex-direction:column}.viewerhead{padding:12px 14px;border-bottom:1px solid #d4dfd9;font-weight:700}.canvas{flex:1;overflow:auto;padding:18px;text-align:center;background:#202724}.canvas img{max-width:none;box-shadow:0 2px 14px #0008}.card{margin:12px;padding:12px;border:1px solid #d4dfd9;border-left:5px solid #d49a18}.card.done{border-left-color:#147766}..rowtitle{display:flex;justify-content:space-between;font-weight:700}..compare{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.compare div{padding:8px;background:#f1f5f3}.fields{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px}.fields label{font-size:12px;color:#53625b}.fields select,.fields input{width:100%;margin-top:3px;padding:6px;border:1px solid #c5d2cb;border-radius:3px}.note{grid-column:1/-1}.notice{position:fixed;right:16px;bottom:16px;padding:9px 12px;background:#1b2c27;color:#fff;display:none}.notice.bad{background:#a53730}@media(max-width:1200px){.layout{grid-template-columns:240px minmax(360px,1fr) 400px}}
</style></head><body><header class="top"><b>第三专家工人状态仲裁</b><span class="grow"></span><span id="progress" class="status"></span><button id="freeze">冻结仲裁</button><button id="export">导出 CSV</button></header><div id="notice" class="notice"></div><section class="layout"><aside class="left"><div class="head"><b>待仲裁分歧</b><div class="muted">只显示 A/B 不一致的行，共 3 行。</div></div><div id="items" class="items"></div></aside><main class="viewer"><div id="title" class="viewerhead"></div><div class="canvas"><img id="image" alt="带人员编号的审计图像"></div></main><aside class="right"><div class="head"><b>选择最终人工状态</b><div class="muted">先独立观察图像，再参考 A/B 的选择。只需仲裁分歧行。</div></div><div id="cards"></div></aside></section><script>
const S={rows:[],images:[],idx:0,locked:false};const $=id=>document.getElementById(id);const done=r=>r.final_helmet_state&&r.final_vest_state&&r.final_overall_state&&r.adjudicator_confidence;async function api(u,o={}){const r=await fetch(u,o),d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.error||'请求失败');return d}function msg(t,b=false){let n=$('notice');n.textContent=t;n.className='notice'+(b?' bad':'');n.style.display='block';clearTimeout(msg.t);msg.t=setTimeout(()=>n.style.display='none',2600)}function cur(){return S.rows[S.idx]}function opt(v,t){let o=document.createElement('option');o.value=v;o.textContent=t;return o}function sel(r,label,key,values){let l=document.createElement('label');l.textContent=label;let s=document.createElement('select');values.forEach(x=>s.append(opt(x[0],x[1])));s.value=r[key]||'';s.disabled=S.locked;s.onchange=()=>{r[key]=s.value;save(r)};l.append(s);return l}function renderItems(){let box=$('items');box.innerHTML='';S.rows.forEach((r,i)=>{let b=document.createElement('button');b.className='item'+(i===S.idx?' active':'')+(done(r)?' done':'');b.innerHTML='<b>'+r.audit_id+'/'+r.person_id+'</b><br><span class="status">'+r.source_group+' | '+(done(r)?'已完成':'待仲裁')+'</span>';b.onclick=()=>{S.idx=i;render()};box.append(b)})}function renderCard(){let r=cur();let c=document.createElement('article');c.className='card'+(done(r)?' done':'');c.innerHTML='<div class="rowtitle"><span>'+r.audit_id+' / '+r.person_id+'</span><span>'+(done(r)?'已填写':'待填写')+'</span></div><div class="compare"><div><b>专家 A</b><br>安全帽：'+r.annotator_A_helmet+'<br>背心：'+r.annotator_A_vest+'<br>总体：'+r.annotator_A_overall+'</div><div><b>专家 B_retry</b><br>安全帽：'+r.annotator_B_helmet+'<br>背心：'+r.annotator_B_vest+'<br>总体：'+r.annotator_B_overall+'</div></div>';let f=document.createElement('div');f.className='fields';f.append(sel(r,'最终安全帽','final_helmet_state',[['','请选择'],['SAFE','SAFE 合格'],['UNSAFE','UNSAFE 不合格'],['REVIEW','REVIEW 无法判断']]));f.append(sel(r,'最终反光背心','final_vest_state',[['','请选择'],['SAFE','SAFE 合格'],['UNSAFE','UNSAFE 不合格'],['REVIEW','REVIEW 无法判断']]));f.append(sel(r,'最终总体状态','final_overall_state',[['','请选择'],['SAFE','SAFE 两项合格'],['UNSAFE','UNSAFE 有一项不合格'],['REVIEW','REVIEW 无法判断']]));f.append(sel(r,'仲裁置信度','adjudicator_confidence',[['','请选择'],['high','高'],['medium','中'],['low','低']]));let l=document.createElement('label');l.className='note';l.textContent='仲裁备注（可空）';let i=document.createElement('input');i.value=r.adjudication_notes||'';i.maxLength=1000;i.disabled=S.locked;i.onchange=()=>{r.adjudication_notes=i.value;save(r)};l.append(i);f.append(l);c.append(f);$('cards').replaceChildren(c)}function render(){let r=cur();$('title').textContent=r.audit_id+' | '+r.source_group+' | '+r.image_name+' | '+r.person_id;$('image').src='/images/'+encodeURIComponent(r.rendered_image);$('progress').textContent=S.rows.filter(done).length+' / '+S.rows.length;$('freeze').disabled=S.locked;renderItems();renderCard()}async function save(r){try{let d=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({row:r})});Object.assign(r,d.row);render();msg('已保存')}catch(e){msg(e.message,true)}}async function boot(){try{let d=await api('/api/bootstrap');Object.assign(S,d);render()}catch(e){msg(e.message,true)}}$('export').onclick=()=>location.assign('/api/export');$('freeze').onclick=async()=>{if(!confirm('确认 3 行仲裁都已完成？冻结后不能修改。'))return;try{let d=await api('/api/finalize',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});S.locked=true;render();msg('仲裁已冻结')}catch(e){msg(e.message,true)}};boot();
</script></body></html>'''


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError(f"{path}: unexpected adjudication header")
    return [{key: row.get(key, "") for key in FIELDS} for row in rows]


def atomic_write(path: Path, rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows); temp = Path(f.name)
    temp.replace(path)


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve(); self.csv = self.root / "results_ab_retry" / "adjudication_template.csv"; self.lock = self.root / "annotator_C" / "ANNOTATION_FINALIZED.json"; self.images_dir = self.root / "annotated_images"
        if not self.csv.is_file() or not self.images_dir.is_dir(): raise FileNotFoundError("仲裁文件或图像目录不存在，请先生成 A/B 分歧表")
        with (self.root / "audit_image_manifest.csv").open("r", encoding="utf-8-sig", newline="") as f: self.manifest={r["audit_id"]:r for r in csv.DictReader(f)}
    def rows(self): return read_rows(self.csv)
    def locked(self): return self.lock.exists()
    def bootstrap(self):
        rows=self.rows(); images=[]
        for r in rows:
            name=next((p.name for p in self.images_dir.glob(r["audit_id"]+"_*")),None)
            if not name: raise ValueError("找不到仲裁图像 "+r["audit_id"])
            r["rendered_image"]=name; images.append(r)
        return {"rows":rows,"images":images,"idx":0,"locked":self.locked()}
    def update(self,row):
        if self.locked(): raise PermissionError("仲裁已冻结")
        rows=self.rows(); index={(r["audit_id"],r["person_id"]):r for r in rows}; key=(row.get("audit_id",""),row.get("person_id",""))
        if key not in index: raise ValueError("未知仲裁行")
        allowed={"","SAFE","UNSAFE","REVIEW"}
        for field in ("final_helmet_state","final_vest_state","final_overall_state"):
            if str(row.get(field,"")) not in allowed: raise ValueError("状态值不合法")
        if str(row.get("adjudicator_confidence","")) not in CONFIDENCE or len(str(row.get("adjudication_notes","")))>1000: raise ValueError("置信度或备注不合法")
        for field in ("final_helmet_state","final_vest_state","final_overall_state","adjudicator_confidence","adjudication_notes"): index[key][field]=str(row.get(field,""))
        atomic_write(self.csv,rows); return index[key]
    def finalize(self):
        if self.locked(): raise PermissionError("仲裁已冻结")
        rows=self.rows()
        if any(not all(r[f] for f in ("final_helmet_state","final_vest_state","final_overall_state","adjudicator_confidence")) for r in rows): raise ValueError("仍有未完成仲裁行")
        self.lock.parent.mkdir(exist_ok=True)
        record={"annotation_status":"frozen_worker_state_adjudication","annotator":"C","rows":len(rows),"csv_sha256":hashlib.sha256(self.csv.read_bytes()).hexdigest(),"frozen_utc":datetime.now(timezone.utc).isoformat()}; self.lock.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8"); return record


def handler(store):
    class H(SimpleHTTPRequestHandler):
        def reply(self,payload,status=HTTPStatus.OK):
            data=json.dumps(payload,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
        def file(self,path,download=False):
            data=path.read_bytes(); self.send_response(HTTPStatus.OK); self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream"); self.send_header("Content-Length",str(len(data)));
            if download:self.send_header("Content-Disposition",f'attachment; filename="{path.name}"')
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            try:
                p=urlparse(self.path).path
                if p in {"/","/index.html"}: data=PAGE.encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
                elif p=="/api/bootstrap":self.reply(store.bootstrap())
                elif p=="/api/export":self.file(store.csv,True)
                elif p.startswith("/images/"):
                    q=(store.images_dir/unquote(p.removeprefix("/images/"))).resolve()
                    if q.parent!=store.images_dir or not q.is_file():raise FileNotFoundError(q.name)
                    self.file(q)
                else:self.send_error(404)
            except (FileNotFoundError,ValueError) as e:self.reply({"error":str(e)},HTTPStatus.BAD_REQUEST)
        def do_POST(self):
            try:
                body=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))).decode()); p=urlparse(self.path).path
                if p=="/api/save":self.reply({"ok":True,"row":store.update(body.get("row",{}))})
                elif p=="/api/finalize":self.reply({"ok":True,"record":store.finalize()})
                else:self.send_error(404)
            except PermissionError as e:self.reply({"error":str(e)},HTTPStatus.FORBIDDEN)
            except (ValueError,json.JSONDecodeError) as e:self.reply({"error":str(e)},HTTPStatus.BAD_REQUEST)
    return H


def main():
    p=argparse.ArgumentParser(); p.add_argument("--audit-root",type=Path,required=True); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8798); a=p.parse_args(); store=Store(a.audit_root); server=ThreadingHTTPServer((a.host,a.port),handler(store)); print(f"Adjudicator C: http://{a.host}:{a.port}");
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
