"""Serve one blinded worker-state annotation pass."""

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

FIELDS = ("audit_id", "person_id", "helmet_state", "vest_state", "overall_state", "annotator_confidence", "visibility_issue", "notes")
STATES = {"", "SAFE", "UNSAFE", "REVIEW"}
CONFIDENCE = {"", "high", "medium", "low"}
VISIBILITY = {"", "none", "unlisted_person", "person_unclear"}

PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>独立 worker-state 人工审计</title><style>
*{box-sizing:border-box}body{margin:0;color:#17211e;background:#f3f6f4;font:14px/1.45 "Microsoft YaHei",Arial,sans-serif}button,select,input{font:inherit}button{cursor:pointer;border:1px solid #c5d2cb;background:#fff;border-radius:4px;padding:7px 10px}button:disabled,select:disabled,input:disabled{cursor:not-allowed;background:#edf1ef;color:#89948e}.top{height:62px;padding:0 18px;display:flex;gap:16px;align-items:center;background:#164a41;color:#fff}.top b{font-size:17px}.grow{flex:1}.status{font-size:12px}.layout{height:calc(100vh - 62px);display:grid;grid-template-columns:255px minmax(460px,1fr) 430px}.left,.viewer,.right{min-height:0;background:#fff}.left{border-right:1px solid #d4dfd9}.right{border-left:1px solid #d4dfd9;overflow:auto}.head{padding:14px;border-bottom:1px solid #d4dfd9}.muted{color:#63716a;font-size:12px}.images{height:calc(100% - 70px);overflow:auto;padding:8px}.imgbtn{display:block;width:100%;text-align:left;margin-bottom:5px}.imgbtn.active{border-left:5px solid #147766;background:#e4f1ed}.imgbtn.done{color:#147766}.viewer{display:flex;flex-direction:column}.viewerhead{padding:12px 14px;border-bottom:1px solid #d4dfd9;font-weight:700}.canvas{flex:1;overflow:auto;padding:18px;text-align:center;background:#202724}.canvas img{max-width:none;box-shadow:0 2px 14px #0008}.righthead{position:sticky;top:0;background:#fff;z-index:1}.card{margin:10px;padding:11px;border:1px solid #d4dfd9;border-left:5px solid #d49a18}.card.done{border-left-color:#147766}.card b{font-size:16px}.rowtitle{display:flex;justify-content:space-between}.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:9px}.grid label,.detail label{font-size:12px;color:#53625b}.grid select,.detail select,.detail input{width:100%;margin-top:3px;padding:6px;border:1px solid #c5d2cb;border-radius:3px;background:#fff}.detail{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}.note{grid-column:1/-1}.notice{position:fixed;right:16px;bottom:16px;background:#1b2c27;color:#fff;padding:9px 12px;display:none}.notice.bad{background:#a53730}.locked{padding:9px 14px;background:#fff2e8;color:#863c20}.help{position:fixed;inset:0;display:none;place-items:center;background:#15201b99;z-index:4;padding:20px}.help.open{display:grid}.modal{max-width:760px;max-height:calc(100vh - 40px);overflow:auto;background:#fff;padding:24px;border-radius:6px}.modal li{margin:7px 0}.modal h2{margin-top:0}@media(max-width:1100px){.layout{grid-template-columns:205px minmax(360px,1fr) 360px}}
</style></head><body><header class="top"><b>独立 worker-state 人工审计</b><span id="who"></span><span class="grow"></span><span id="progress" class="status"></span><button id="help">标注说明</button><button id="freeze">冻结本轮</button><button id="export">导出 CSV</button></header><div id="locked" class="locked" hidden>本轮已冻结，不能再修改。</div><section class="layout"><aside class="left"><div class="head"><b>图像列表</b><div id="count" class="muted"></div></div><div id="images" class="images"></div></aside><main class="viewer"><div id="title" class="viewerhead"></div><div class="canvas"><img id="image" alt="带人员编号的审计图像"></div></main><aside class="right"><div class="head righthead"><b>逐个判断工人状态</b><div class="muted">只根据原图判断，不看模型输出。每个 P 行的三个状态都要填写。</div></div><div id="rows"></div></aside></section><div id="notice" class="notice"></div><div id="helpbox" class="help"><div class="modal"><h2>傻瓜式标注说明</h2><p>图中黄色框和 P1、P2 是工人编号。右侧每张卡对应一个工人。</p><ol><li>看这个 P 工人实际有没有戴安全帽，选择安全帽状态。</li><li>看这个 P 工人实际有没有穿反光背心，选择反光背心状态。</li><li>总体状态：两项都明确合格才选 SAFE；任一项明确不合格选 UNSAFE；看不清就选 REVIEW。</li><li>被遮挡、太小、光照差、只露出局部时不要猜，选 REVIEW，并可在备注写原因。</li><li>置信度是你的把握程度，不是模型分数。</li><li>如果图中有明显可见但没有黄色 P 框的人员，在“可见性问题”选“有未框出人员”，并在备注说明。</li><li>完成全部 P 行后导出备份，再冻结。两位专家必须独立标注。</li></ol><button id="close">我明白了，开始标注</button></div></div><script>
const S={rows:[],images:[],idx:0,locked:false};const $=id=>document.getElementById(id);const byImage=()=>S.rows.reduce((m,r)=>(m[r.audit_id]??=[]).push(r)&&m,{});const done=r=>r.helmet_state&&r.vest_state&&r.overall_state&&r.annotator_confidence&&r.visibility_issue;const imageDone=id=>(byImage()[id]||[]).every(done);async function api(u,o={}){const r=await fetch(u,o),d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.error||'请求失败');return d}function msg(t,b=false){let n=$('notice');n.textContent=t;n.className='notice'+(b?' bad':'');n.style.display='block';clearTimeout(msg.t);msg.t=setTimeout(()=>n.style.display='none',2600)}function cur(){return S.images[S.idx]}function option(v,t){let o=document.createElement('option');o.value=v;o.textContent=t;return o}function select(r,label,key,vals){let l=document.createElement('label');l.textContent=label;let s=document.createElement('select');vals.forEach(x=>s.append(option(x[0],x[1])));s.value=r[key]||'';s.disabled=S.locked;s.onchange=()=>{r[key]=s.value;save(r)};l.append(s);return l}function renderImages(){let b=$('images');b.innerHTML='';S.images.forEach((x,i)=>{let q=document.createElement('button');q.className='imgbtn'+(i===S.idx?' active':'')+(imageDone(x.audit_id)?' done':'');q.innerHTML='<b>'+x.audit_id+'</b><br><span class="status">'+x.source_group+' | '+x.person_count+' 人'+(imageDone(x.audit_id)?' | 已完成':'')+'</span>';q.onclick=()=>{S.idx=i;render()};b.append(q)});$('count').textContent='共 '+S.images.length+' 张'}function renderRows(){let box=$('rows');box.innerHTML='';(byImage()[cur().audit_id]||[]).forEach(r=>{let c=document.createElement('article');c.className='card'+(done(r)?' done':'');let t=document.createElement('div');t.className='rowtitle';t.innerHTML='<b>'+r.person_id+'</b><span class="muted">'+(done(r)?'已填写':'待填写')+'</span>';c.append(t);let g=document.createElement('div');g.className='grid';g.append(select(r,'安全帽','helmet_state',[['','请选择'],['SAFE','SAFE 合格'],['UNSAFE','UNSAFE 不合格'],['REVIEW','REVIEW 无法判断']]));g.append(select(r,'反光背心','vest_state',[['','请选择'],['SAFE','SAFE 合格'],['UNSAFE','UNSAFE 不合格'],['REVIEW','REVIEW 无法判断']]));g.append(select(r,'总体状态','overall_state',[['','请选择'],['SAFE','SAFE 两项合格'],['UNSAFE','UNSAFE 有一项不合格'],['REVIEW','REVIEW 无法判断']]));let d=document.createElement('div');d.className='detail';d.append(select(r,'置信度','annotator_confidence',[['','请选择'],['high','高'],['medium','中'],['low','低']]));d.append(select(r,'可见性问题','visibility_issue',[['','请选择'],['none','无'],['unlisted_person','有未框出人员'],['person_unclear','该人员框不清楚']]));let n=document.createElement('label');n.className='note';n.textContent='备注（可空）';let inp=document.createElement('input');inp.value=r.notes||'';inp.maxLength=1000;inp.disabled=S.locked;inp.onchange=()=>{r.notes=inp.value;save(r)};n.append(inp);d.append(n);c.append(g,d);box.append(c)})}function render(){let x=cur();$('title').textContent=x.audit_id+' | '+x.source_group+' | '+x.person_count+' 人 | fold '+x.fold;$('image').src='/images/'+encodeURIComponent(x.image);$('image').style.width='100%';$('locked').hidden=!S.locked;$('freeze').disabled=S.locked;$('progress').textContent=S.rows.filter(done).length+' / '+S.rows.length;renderImages();renderRows()}async function save(r){try{let d=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({updates:[r]})});$('progress').textContent=d.progress.complete+' / '+d.progress.total;render()}catch(e){msg(e.message,true)}}async function boot(){try{let d=await api('/api/bootstrap');Object.assign(S,{rows:d.rows,images:d.images,locked:d.locked});$('who').textContent='盲标 '+d.annotator;render()}catch(e){msg(e.message,true)}}$('help').onclick=()=>$('helpbox').classList.add('open');$('close').onclick=()=>$('helpbox').classList.remove('open');$('export').onclick=()=>location.assign('/api/export');$('freeze').onclick=async()=>{if(!confirm('确认所有工人状态均已填写，并冻结本轮吗？'))return;try{let d=await api('/api/finalize',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});S.locked=true;render();msg('本轮已冻结')}catch(e){msg(e.message,true)}};boot();
</script></body></html>'''


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"unexpected CSV header in {path}")
        return [{k: row.get(k, "") for k in FIELDS} for row in reader]


def atomic_write(path, rows):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows); tmp = Path(f.name)
    tmp.replace(path)


class Store:
    def __init__(self, root: Path, annotator: str):
        self.root = root.resolve(); self.annotator = annotator; self.dir = self.root / f"annotator_{annotator}"
        self.csv = self.dir / "worker_state.csv"; self.lock = self.dir / "ANNOTATION_FINALIZED.json"; self.images_dir = self.root / "annotated_images"
        if not self.csv.is_file() or not self.images_dir.is_dir(): raise FileNotFoundError("审计包不完整")
        with (self.root / "audit_image_manifest.csv").open("r", encoding="utf-8-sig", newline="") as f: self.manifest = {r["audit_id"]: r for r in csv.DictReader(f)}
        self.images = {p.name.split("_", 1)[0]: p.name for p in self.images_dir.glob("*.jpg")}
        if set(self.images) != set(self.manifest): raise ValueError("图像与清单不一致")
    def rows(self): return read_rows(self.csv)
    def locked(self): return self.lock.exists()
    def complete(self, r): return all(r[k] for k in ("helmet_state","vest_state","overall_state","annotator_confidence","visibility_issue"))
    def bootstrap(self):
        rows=self.rows(); order=sorted(self.manifest,key=lambda x:int(x[1:])); return {"annotator":self.annotator,"locked":self.locked(),"rows":rows,"images":[{"audit_id":i,"image":self.images[i],"source_group":self.manifest[i]["source_group"],"fold":self.manifest[i]["fold"],"person_count":self.manifest[i]["person_count"]} for i in order],"progress":{"complete":sum(self.complete(r) for r in rows),"total":len(rows)}}
    def update(self, updates):
        if self.locked(): raise PermissionError("本轮已经冻结")
        rows=self.rows(); index={(r["audit_id"],r["person_id"]):r for r in rows}
        for u in updates:
            key=(str(u.get("audit_id","")).strip(),str(u.get("person_id","")).strip());
            if key not in index: raise ValueError("未知工人行")
            r=index[key]; vals={k:str(u.get(k,"")).strip() for k in FIELDS if k not in ("audit_id","person_id")}
            if vals["helmet_state"] not in STATES or vals["vest_state"] not in STATES or vals["overall_state"] not in STATES or vals["annotator_confidence"] not in CONFIDENCE or vals["visibility_issue"] not in VISIBILITY or len(vals["notes"])>1000: raise ValueError("状态、置信度或备注不合法")
            r.update(vals)
        atomic_write(self.csv,rows); return {"complete":sum(self.complete(r) for r in rows),"total":len(rows)}
    def finalize(self):
        if self.locked(): raise PermissionError("本轮已经冻结")
        rows=self.rows(); missing=[r["audit_id"]+"/"+r["person_id"] for r in rows if not self.complete(r)]
        if missing: raise ValueError(f"还有 {len(missing)} 行未完成")
        record={"annotation_status":"frozen_independent_worker_state_pass","annotator":self.annotator,"completed_worker_rows":len(rows),"csv_sha256":hashlib.sha256(self.csv.read_bytes()).hexdigest(),"frozen_utc":datetime.now(timezone.utc).isoformat()}
        self.lock.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8"); return record


def handler(store):
    class H(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args): print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")
        def reply(self, payload, status=HTTPStatus.OK):
            data=json.dumps(payload,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
        def file(self,path,download=False):
            data=path.read_bytes(); self.send_response(HTTPStatus.OK); self.send_header("Content-Type",mimetypes.guess_type(path.name)[0] or "application/octet-stream"); self.send_header("Content-Length",str(len(data)));
            if download:self.send_header("Content-Disposition",f'attachment; filename="{path.name}"')
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            try:
                p=urlparse(self.path).path
                if p in {"/","/index.html"}: data=PAGE.encode(); self.send_response(HTTPStatus.OK); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
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
                n=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(n).decode())
                p=urlparse(self.path).path
                if p=="/api/save":self.reply({"ok":True,"progress":store.update(body.get("updates",[]))})
                elif p=="/api/finalize":self.reply({"ok":True,"record":store.finalize()})
                else:self.send_error(404)
            except PermissionError as e:self.reply({"error":str(e)},HTTPStatus.FORBIDDEN)
            except (ValueError,json.JSONDecodeError) as e:self.reply({"error":str(e)},HTTPStatus.BAD_REQUEST)
    return H


def main():
    p=argparse.ArgumentParser(); p.add_argument("--audit-root",type=Path,required=True); p.add_argument("--annotator",choices=("A","B","B_retry"),required=True); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8775); a=p.parse_args(); store=Store(a.audit_root,a.annotator); server=ThreadingHTTPServer((a.host,a.port),handler(store)); print(f"Annotator {a.annotator}: http://{a.host}:{a.port}"); print(f"Audit root: {store.root}");
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
