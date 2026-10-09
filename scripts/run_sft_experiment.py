"""VS Code entry: immutable real-data baseline/SFT/paired-evaluation protocol."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data import load_manifest,check_document_splits
from run_record import snapshot_run,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--protocol',default='configs/real-sft-v0.4.0.json');args=p.parse_args()
    data=Path(args.data).resolve();cfg=json.loads(Path(args.protocol).read_text())
    manifests=[data/(role+'.jsonl') for role in ['train','val','test']]
    check_document_splits(*(load_manifest(path) for path in manifests))
    out=snapshot_run(args.output,{**vars(args),'protocol':cfg,'protocol_sha256':sha(args.protocol)},manifests).resolve()
    commands=[];started=time.time()
    def run(name,script,arguments):
        command=[sys.executable,str(ROOT/'scripts'/script)]+list(map(str,arguments))
        record={'name':name,'command':command,'started':time.time()};commands.append(record)
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        print('STAGE_STARTED',name,flush=True)
        with (out/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        record.update(exit_code=code,finished=time.time());(out/'commands.json').write_text(json.dumps(commands,indent=2))
        if code:raise RuntimeError('Stage failed: '+name+'; inspect saved log')
    common=['--model',cfg['model'],'--revision',cfg['revision'],'--device',cfg['device'],'--max-image-edge',cfg['max_image_edge']]
    run('checks','run_checks.py',[])
    for role in ['val','test']:
        run('baseline-'+role,'infer.py',['--manifest',data/(role+'.jsonl'),'--split',role,'--output',out/('baseline-'+role)]+common)
    train_args=['--train',data/'train.jsonl','--val',data/'val.jsonl','--output',out/'training']+common
    for key in ['rank','alpha','lr','epochs','accumulation','seed']:train_args+=['--'+key,cfg[key]]
    if cfg['train_projector']:train_args+=['--train-projector']
    if cfg.get('gradient_checkpointing'):train_args+=['--gradient-checkpointing']
    run('training','train_lora.py',train_args)
    for role in ['val','test']:
        run('adapter-'+role,'infer.py',['--manifest',data/(role+'.jsonl'),'--split',role,'--adapter',out/'training/adapter.pt','--output',out/('adapter-'+role)]+common)
        run('score-baseline-'+role,'evaluate.py',['--predictions',out/('baseline-'+role)/'predictions.jsonl','--output',out/('baseline-'+role)/'answer-metrics.json'])
        run('score-adapter-'+role,'evaluate.py',['--predictions',out/('adapter-'+role)/'predictions.jsonl','--compare-to',out/('baseline-'+role)/'predictions.jsonl','--output',out/('adapter-'+role)/'answer-metrics.json'])
    report={'protocol':cfg,'dataset':json.loads((data/'dataset-summary.json').read_text()),
        'training':json.loads((out/'training/metrics.json').read_text()),
        'adapter_sha256':sha(out/'training/adapter.pt'),'elapsed_seconds':time.time()-started,
        'launch':'VS Code integrated terminal; saved Python orchestrator',
        'results':{role:{kind:json.loads((out/(kind+'-'+role)/'answer-metrics.json').read_text()) for kind in ['baseline','adapter']} for role in ['val','test']},
        'scope':'small real-data SFT experiment; reserved test role comes from labeled official validation, not official blind test; no full benchmark claim'}
    (out/'experiment-summary.json').write_text(json.dumps(report,indent=2));print('REAL_SFT_EXPERIMENT_FINISHED',flush=True)
if __name__=='__main__':main()
