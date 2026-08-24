"""Serve a local, manual privacy-redaction review for qualitative panels.

The server never uploads source images. It writes only rectangle coordinates and
review status into privacy_redaction_annotations.json beside the local audit.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure 5 隐私打码复核</title>
<style>
body{margin:0;background:#f5f6f4;color:#1f2825;font:15px system-ui,"Microsoft YaHei",sans-serif}
header{padding:14px 22px;background:#193f38;color:#fff;display:flex;justify-content:space-between;gap:12px;align-items:center}
main{max-width:1360px;margin:18px auto;padding:0 18px;display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:18px}
.panel{background:#fff;border:1px solid #cbd5d1;padding:16px}.note{font-size:13px;line-height:1.55;color:#4c5b56;margin:0 0 13px}
canvas{max-width:100%;height:auto;display:block;border:1px solid #77867f;cursor:crosshair;background:#111}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
button,select{font:inherit;padding:7px 10px;border:1px solid #75857f;background:#fff;color:#17221e;cursor:pointer}button.primary{background:#176b5c;color:#fff;border-color:#176b5c}button.warn{background:#8e3f2c;color:#fff;border-color:#8e3f2c}button:disabled{opacity:.45;cursor:not-allowed}
.image-btn{display:block;width:100%;text-align:left;margin:6px 0}.image-btn.active{border-left:5px solid #176b5c;background:#e1f0ea}.status{font-size:13px;color:#53645d}.region{display:flex;justify-content:space-between;gap:6px;border-top:1px solid #dde5e1;padding:8px 0;font-size:13px}.ok{color:#176b5c;font-weight:700}.bad{color:#a33d2a;font-weight:700}label{display:block;margin:12px 0}#message{min-height:22px;font-weight:600}@media(max-width:850px){main{grid-template-columns:1fr}}
</style>
<header><b>Figure 5 隐私打码复核</b><span id="lock"></span></header>
<main><section class="panel"><p class="note">逐张检查人脸、工牌/姓名、车牌、公司标识和其他可识别信息。鼠标拖拽框选需要遮挡区域，再选择原因。此工具只保存坐标，不上传图像。确认本图无遗漏后勾选“已人工复核”。冻结后不可修改。</p><div class="controls"><select id="reason"><option value="face">人脸</option><option value="badge_or_name">工牌或姓名</option><option value="vehicle_plate">车牌</option><option value="company_mark">公司标识</option><option value="other_identifier">其他可识别信息</option></select><button id="undo">删除最后一框</button><button id="save" class="primary">保存本图</button></div><canvas id="canvas"></canvas><label><input type="checkbox" id="reviewed"> 本图已经人工检查；所有可识别信息已框选，或确认不存在。</label><div id="message"></div></section><aside class="panel"><b>图片列表</b><div id="images"></div><hr><b>当前图片的遮挡框</b><div id="regions"></div><hr><button id="freeze" class="warn">冻结全部复核结果</button><p class="status">冻结条件：每张图均已勾选人工复核。冻结后生成的 JSON 才能用于论文 Figure 5 导出。</p></aside></main>
<script>
let state, current=0, image=new Image(), drawing=null;
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d');
function currentItem(){return state.items[current]}
function rec(){return state.data.images[currentItem().panel]}
async function api(path,method='GET',body){let r=await fetch(path,{method,headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});let d=await r.json();if(!r.ok)throw Error(d.error||'操作失败');return d}
function message(s,ok=false){$('message').textContent=s;$('message').className=ok?'ok':'bad'}
function draw(){if(!image.complete)return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;ctx.drawImage(image,0,0);for(const [i,r] of rec().regions.entries()){ctx.strokeStyle='#ff3b30';ctx.lineWidth=Math.max(3,canvas.width/500);ctx.strokeRect(r.x1,r.y1,r.x2-r.x1,r.y2-r.y1);ctx.fillStyle='rgba(255,59,48,.85)';ctx.font=`${Math.max(16,canvas.width/50)}px sans-serif`;ctx.fillText(`${i+1}: ${r.reason}`,r.x1+4,Math.max(18,r.y1-4))}if(drawing){ctx.setLineDash([8,5]);ctx.strokeStyle='#ffdc00';ctx.strokeRect(drawing.x1,drawing.y1,drawing.x2-drawing.x1,drawing.y2-drawing.y1);ctx.setLineDash([])}}
function imageList(){let box=$('images');box.innerHTML='';state.items.forEach((x,i)=>{let b=document.createElement('button');b.className='image-btn '+(i===current?'active':'');let r=state.data.images[x.panel];b.innerHTML=`(${x.panel}) ${x.image_name}<br><span class="${r.reviewed?'ok':'bad'}">${r.reviewed?'已复核':'未复核'}，${r.regions.length} 个遮挡框</span>`;b.disabled=state.data.frozen;b.onclick=()=>{current=i;load()};box.appendChild(b)})}
function regionList(){let box=$('regions');box.innerHTML='';if(!rec().regions.length){box.textContent='尚未框选区域';return}rec().regions.forEach((r,i)=>{let d=document.createElement('div');d.className='region';d.innerHTML=`<span>#${i+1} ${r.reason}<br>${Math.round(r.x1)},${Math.round(r.y1)} - ${Math.round(r.x2)},${Math.round(r.y2)}</span>`;let b=document.createElement('button');b.textContent='删除';b.disabled=state.data.frozen;b.onclick=()=>{rec().regions.splice(i,1);render()};d.appendChild(b);box.appendChild(d)})}
function render(){$('reviewed').checked=rec().reviewed;$('reviewed').disabled=state.data.frozen;$('save').disabled=state.data.frozen;$('undo').disabled=state.data.frozen;$('freeze').disabled=state.data.frozen; $('lock').textContent=state.data.frozen?'已冻结':'未冻结';draw();imageList();regionList()}
function load(){let x=currentItem();image=new Image();image.onload=render;image.src='/image/'+encodeURIComponent(x.panel)+'?v='+Date.now();message('')}
function point(e){let r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height}}
canvas.onpointerdown=e=>{if(state.data.frozen)return;let p=point(e);drawing={x1:p.x,y1:p.y,x2:p.x,y2:p.y};canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!drawing)return;let p=point(e);drawing.x2=p.x;drawing.y2=p.y;draw()};
canvas.onpointerup=e=>{if(!drawing)return;let p=point(e);drawing.x2=p.x;drawing.y2=p.y;let r=drawing;drawing=null;let x1=Math.min(r.x1,r.x2),y1=Math.min(r.y1,r.y2),x2=Math.max(r.x1,r.x2),y2=Math.max(r.y1,r.y2);if(x2-x1>5&&y2-y1>5){rec().regions.push({x1,y1,x2,y2,reason:$('reason').value});render()}else draw()};
$('undo').onclick=()=>{rec().regions.pop();render()};
$('save').onclick=async()=>{try{rec().reviewed=$('reviewed').checked;state.data=await api('/api/save','POST',{panel:currentItem().panel,record:rec()});render();message('已保存本图。',true)}catch(e){message(e.message)}};
$('freeze').onclick=async()=>{if(!confirm('冻结后不能修改，确认全部图片已人工复核？'))return;try{state.data=await api('/api/freeze','POST');render();message('已冻结。可用此 JSON 重新生成脱敏 Figure 5。',true)}catch(e){message(e.message)}};
(async()=>{try{let b=await api('/api/bootstrap');state={items:b.items,data:b.data};load()}catch(e){message(e.message)}})();
</script></html>"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_items(root: Path) -> list[dict]:
    manifest = json.loads((root / "qualitative_manifest.json").read_text(encoding="utf-8"))
    return [
        {
            "panel": str(row["panel"]),
            "image_name": str(row["image_name"]),
            "original_image_path": str(row["original_image_path"]),
            "copied_source_frame": str(row["copied_source_frame"]),
        }
        for row in manifest["panels"]
    ]


def default_data(items: list[dict]) -> dict:
    return {
        "version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "frozen": False,
        "images": {
            item["panel"]: {
                "panel": item["panel"],
                "original_image_path": item["original_image_path"],
                "reviewed": False,
                "regions": [],
            }
            for item in items
        },
    }


def normalize_record(record: dict, item: dict) -> dict:
    regions = []
    for row in record.get("regions", []):
        try:
            x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("遮挡框坐标无效") from error
        if x2 <= x1 or y2 <= y1:
            raise ValueError("遮挡框必须有正面积")
        regions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "reason": str(row.get("reason", "other_identifier"))})
    return {
        "panel": item["panel"],
        "original_image_path": item["original_image_path"],
        "reviewed": bool(record.get("reviewed")),
        "regions": regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualitative-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    root = args.qualitative_root.resolve()
    items = load_items(root)
    by_panel = {item["panel"]: item for item in items}
    output = root / "privacy_redaction_annotations.json"
    data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else default_data(items)

    def save() -> None:
        data["updated_at"] = utc_now()
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def reply(self, payload: dict, status: int = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                encoded = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if path == "/api/bootstrap":
                self.reply({"items": items, "data": data, "output": str(output)})
                return
            if path.startswith("/image/"):
                panel = unquote(path.removeprefix("/image/"))
                item = by_panel.get(panel)
                source = root / "selected_source_frames" / item["copied_source_frame"] if item else None
                if not source or not source.is_file():
                    self.reply({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                    return
                content = source.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(str(source))[0] or "image/jpeg")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self.reply({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if data.get("frozen"):
                    raise ValueError("本次复核已冻结，不能修改")
                if self.path == "/api/save":
                    panel = str(payload.get("panel", ""))
                    if panel not in by_panel:
                        raise ValueError("未知图片")
                    data["images"][panel] = normalize_record(payload.get("record", {}), by_panel[panel])
                    save()
                    self.reply(data)
                    return
                if self.path == "/api/freeze":
                    incomplete = [panel for panel, record in data["images"].items() if not record.get("reviewed")]
                    if incomplete:
                        raise ValueError("尚未人工复核的图片: " + ", ".join(incomplete))
                    data["frozen"] = True
                    data["frozen_at"] = utc_now()
                    save()
                    self.reply(data)
                    return
                self.reply({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as error:
                self.reply({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Privacy redaction review: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
