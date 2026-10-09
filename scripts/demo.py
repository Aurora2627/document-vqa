# -*- coding: utf-8 -*-
"""Loopback-only document QA demo using the same pinned PyTorch baseline."""
import argparse,json,sys,threading,time,uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import torch
from modeling import load_pretrained
from batching import prompt_messages,move_batch
from views import prepare_views
from demo_input import parse_request
from run_record import snapshot_run,sha
MODEL='llava-hf/llava-onevision-qwen2-0.5b-ov-hf'
REVISION='74dd0bf867a4cda7950c17663794267c60cf4b40'
PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLaVA 文档问答</title>
<style>body{font:16px system-ui;color:#172838;background:#f3f6fa;max-width:1000px;margin:40px auto;padding:0 20px}main{display:grid;grid-template-columns:1fr 1fr;gap:24px}section{background:white;border:1px solid #d9e1e9;border-radius:16px;padding:24px}h1{font-size:28px}p{line-height:1.6}label{display:block;margin:16px 0 8px}textarea,select{box-sizing:border-box;width:100%;font:inherit;padding:12px;border:1px solid #bbc8d5;border-radius:8px}button{padding:12px 22px;background:#205cca;border:0;color:white;border-radius:8px;font:inherit;margin-top:18px}button:disabled{opacity:.5}img{max-width:100%;max-height:480px;object-fit:contain}#answer{white-space:pre-wrap;font-size:22px;overflow-wrap:anywhere}small{color:#506277}@media(max-width:700px){main{grid-template-columns:1fr}}</style>
<h1>LLaVA 文档视觉问答</h1><p>上传单页文档，询问日期、金额、人名或其他页面内容。当前模型主要验证了英文文档。</p><main><section><form id="form"><label for="image">文档图片</label><input id="image" type="file" accept="image/png,image/jpeg,image/webp" required><label for="question">你的问题</label><textarea id="question" rows="3" maxlength="2000" placeholder="What is the date on this document?" required></textarea><label for="edge">输入长边上限</label><select id="edge"><option value="512">512</option><option value="768">768</option></select><button id="submit">读取文档并回答</button></form><p><small>回答可能出错，请对照原文核实。当前页面不提供证据定位。图片和运行记录保存在本机。</small></p><p id="status" role="status"></p></section><section><img id="preview" alt="文档预览" hidden><h2>模型回答</h2><div id="answer">等待你的问题</div><p id="details"></p></section></main>
<script>const file=document.getElementById('image'),preview=document.getElementById('preview');let previewURL;
file.onchange=()=>{if(previewURL)URL.revokeObjectURL(previewURL);if(file.files[0]){previewURL=URL.createObjectURL(file.files[0]);preview.src=previewURL;preview.hidden=false}};
document.getElementById('form').onsubmit=async e=>{e.preventDefault();const button=document.getElementById('submit'),status=document.getElementById('status'),answer=document.getElementById('answer');button.disabled=true;status.textContent='正在阅读文档…';try{const image=file.files[0];if(image.size>5*1024*1024)throw Error('图片文件需小于 5 MB');const encoded=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(image)});const response=await fetch('/api/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:encoded,question:document.getElementById('question').value,max_image_edge:Number(document.getElementById('edge').value)})});const result=await response.json();if(!response.ok)throw Error(result.error);answer.textContent=result.answer;document.getElementById('details').textContent=`生成用时 ${result.generation_seconds.toFixed(2)} 秒 · 输入 ${result.input_tokens} tokens`;status.textContent='已完成';}catch(error){status.textContent=error.message;}finally{button.disabled=false}};</script></html>'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--device',choices=['mps','cuda','cpu'],default='mps');args=p.parse_args()
    dtype=torch.float32 if args.device=='cpu' else torch.float16
    model,processor=load_pretrained(MODEL,REVISION,args.device,dtype);model.eval();lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,payload,content_type='application/json; charset=utf-8'):
            body=payload.encode() if isinstance(payload,str) else json.dumps(payload,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def allowed(self):
            expected='127.0.0.1:'+str(args.port)
            return self.headers.get('Host')==expected and self.headers.get('Origin') in [None,'http://'+expected]
        def do_GET(self):
            if not self.allowed():return self.reply(403,{'error':'仅支持本机访问'})
            if self.path=='/':self.reply(200,PAGE,'text/html; charset=utf-8')
            elif self.path=='/health':self.reply(200,{'ready':True,'framework':'PyTorch','device':args.device,'model':MODEL})
            else:self.reply(404,{'error':'页面不存在'})
        def do_POST(self):
            if not self.allowed():return self.reply(403,{'error':'仅支持本机访问'})
            if self.path!='/api/answer':return self.reply(404,{'error':'接口不存在'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<8*1024*1024 or self.headers.get('Content-Type')!='application/json': raise ValueError('请求格式或大小无效')
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict):raise ValueError('请求需为 JSON 对象')
                image,question,edge=parse_request(payload)
            except (ValueError,TypeError) as error:return self.reply(400,{'error':str(error)})
            if not lock.acquire(blocking=False):return self.reply(429,{'error':'模型正在处理另一个问题，请稍后重试'})
            out=None
            try:
                name=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
                out=snapshot_run(ROOT/'runs'/'demo-requests'/name,{'question':question,'max_image_edge':edge,'model':MODEL,'revision':REVISION,'device':args.device,'adapter':None,'launch':'VS Code local demo'})
                image.save(out/'input.png');(out/'image-sha256.json').write_text(json.dumps({'sha256':sha(out/'input.png')}))
                view=prepare_views(image,'whole',edge)[0]
                prompt=processor.apply_chat_template(prompt_messages(question),tokenize=False,add_generation_prompt=True)
                batch=move_batch(processor(text=prompt,images=view,return_tensors='pt'),args.device,dtype)
                if args.device=='mps':torch.mps.synchronize()
                if args.device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                with torch.inference_mode():tokens=model.generate(**batch,max_new_tokens=64,do_sample=False)
                if args.device=='mps':torch.mps.synchronize()
                if args.device=='cuda':torch.cuda.synchronize()
                result={'answer':processor.tokenizer.decode(tokens[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip(),
                    'generation_seconds':time.perf_counter()-start,'input_tokens':batch['input_ids'].shape[1],'run_id':name}
                (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));self.reply(200,result)
            except Exception as error:
                if out:(out/'error.json').write_text(json.dumps({'error_type':type(error).__name__}))
                self.reply(500,{'error':'推理失败，请查看本机运行记录'})
            finally:lock.release()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print('DEMO_READY http://127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
