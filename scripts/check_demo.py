"""Start the real MPS demo, verify HTTP contracts and compare one saved baseline answer."""
import argparse,base64,json,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from data import load_manifest
from run_record import snapshot_run

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    out=snapshot_run(args.output,vars(args),[args.manifest,args.predictions])
    row=load_manifest(args.manifest)[0];expected={r['question_id']:r['prediction'] for r in map(json.loads,Path(args.predictions).read_text().splitlines())}[row['question_id']]
    base='http://127.0.0.1:8766'
    with (out/'server.log').open('x') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/demo.py'),'--port','8766'],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+120
            while True:
                if process.poll() is not None:raise RuntimeError('Demo server exited before ready')
                try:
                    with urlopen(base+'/health',timeout=2) as response:health=json.load(response)
                    break
                except (URLError,TimeoutError):
                    if time.monotonic()>deadline:raise RuntimeError('Demo readiness timed out')
                    time.sleep(1)
            with urlopen(base+'/',timeout=5) as response:page=response.read().decode()
            if 'LLaVA 文档视觉问答' not in page:raise ValueError('Demo page missing')
            bad=Request(base+'/api/answer',data=json.dumps({'question':''}).encode(),headers={'Content-Type':'application/json'})
            try:urlopen(bad,timeout=5);raise AssertionError('Invalid request accepted')
            except HTTPError as error:
                if error.code!=400:raise
            foreign=Request(base+'/health',headers={'Origin':'https://example.com'})
            try:urlopen(foreign,timeout=5);raise AssertionError('Foreign origin accepted')
            except HTTPError as error:
                if error.code!=403:raise
            payload={'question':row['question'],'image_base64':base64.b64encode(Path(row['image_path']).read_bytes()).decode(),'max_image_edge':512}
            request=Request(base+'/api/answer',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=120) as response:result=json.load(response)
            (out/'response.json').write_text(json.dumps(result,indent=2))
            if result['answer']!=expected:raise ValueError('Demo output differs from saved baseline')
            report={'framework':'PyTorch','launch':'VS Code','health':health,'page_served':True,
                'invalid_request_status':400,'foreign_origin_status':403,'matches_saved_baseline':True,
                'question_id':row['question_id'],'generation_seconds':result['generation_seconds'],'input_tokens':result['input_tokens'],
                'scope':'one real document end-to-end smoke check, not an accuracy benchmark'}
            (out/'summary.json').write_text(json.dumps(report,indent=2))
            with Path(args.report).open('x') as output:json.dump(report,output,indent=2)
            print('DEMO_CHECK_PASSED',flush=True)
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
