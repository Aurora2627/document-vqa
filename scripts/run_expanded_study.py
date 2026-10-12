"""Expanded SFT, prior-adapter comparator and trained-model region development ablation."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run,sha
from evaluation import summarize,compare
from paired_statistics import paired_bootstrap

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--previous-adapter',required=True);p.add_argument('--development',required=True)
    p.add_argument('--report');p.add_argument('--region-report');p.add_argument('--device',choices=['cuda','mps'],default='cuda');args=p.parse_args()
    for target in [args.report,args.region_report]:
        if target and Path(target).exists():raise ValueError('Report already exists; choose a new name')
    data=Path(args.data).resolve();protocol=ROOT/'configs/expanded-sft-cuda.json';cfg=json.loads(protocol.read_text());cfg['device']=args.device
    files=[data/(role+'.jsonl') for role in ['train','val','test']]+[args.previous_adapter,args.development,protocol]
    out=snapshot_run(args.output,{**vars(args),'previous_adapter_sha256':sha(args.previous_adapter),
        'protocol':cfg,'primary_comparison':'expanded LoRA versus frozen base at identical 512 resolution',
        'secondary_comparison':'expanded LoRA versus fixed prior LoRA; validation selection differs; not pure data-scale causality'},files).resolve()
    commands=[]
    def run(name,script,arguments):
        command=[sys.executable,str(ROOT/'scripts'/script)]+list(map(str,arguments));record={'name':name,'command':command};commands.append(record)
        print('EXPANDED_STAGE',name,flush=True)
        with (out/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            record['exit_code']=process.wait()
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        if record['exit_code']:raise RuntimeError('Failed stage '+name)
    run('sft','run_sft_experiment.py',['--data',data,'--output',out/'sft','--protocol',protocol,'--device',args.device])
    run('prior-adapter-test','infer.py',['--manifest',data/'test.jsonl','--split','test','--adapter',Path(args.previous_adapter).resolve(),
        '--model',cfg['model'],'--revision',cfg['revision'],'--device',cfg['device'],'--max-image-edge',cfg['max_image_edge'],'--output',out/'prior-adapter-test'])
    locations={'base':out/'sft/baseline-test/predictions.jsonl','prior':out/'prior-adapter-test/predictions.jsonl','expanded':out/'sft/adapter-test/predictions.jsonl'}
    rows={name:[json.loads(line) for line in path.read_text().splitlines()] for name,path in locations.items()}
    sft=json.loads((out/'sft/experiment-summary.json').read_text())
    summary={'version':'0.7.0','protocol':cfg,'dataset':sft['dataset'],'training':sft['training'],
        'adapter_sha256':sft['adapter_sha256'],'results':{name:summarize(group) for name,group in rows.items()},
        'comparisons':{name+'_to_expanded':{'paired':compare(rows[name],rows['expanded']),
            'uncertainty':paired_bootstrap(rows[name],rows['expanded'])} for name in ['base','prior']},
        'prediction_sha256':{name:sha(path) for name,path in locations.items()},
        'limitations':['64 reserved labeled-validation documents; finite candidate pool, not official blind test',
            'ANLS-style metrics; official evaluator parity not verified',
            'Prior and expanded adapters use different validation sets/checkpoint selection, so this is not a pure data-volume ablation',
            'Exploratory paired intervals not adjusted for multiple comparisons; dataset inspected after evaluation']}
    (out/'training-study-summary.json').write_text(json.dumps(summary,indent=2))
    with (Path(args.report) if args.report else out/'study-summary.json').open('x') as public:json.dump(summary,public,indent=2)
    run('trained-region-dev','run_region_experiment.py',['--manifest',Path(args.development).resolve(),
        '--device',args.device,'--adapter',out/'sft/training/adapter.pt','--output',out/'region-dev',
        '--report',Path(args.region_report).resolve() if args.region_report else out/'region-study-summary.json'])
    print('EXPANDED_STUDY_FINISHED',flush=True)
if __name__=='__main__':main()
