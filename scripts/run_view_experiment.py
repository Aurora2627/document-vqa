"""Fixed development experiment: whole page, repeated pages, and fixed halves."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run
from evaluation import compare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    protocol={'model':'llava-hf/llava-onevision-qwen2-0.5b-ov-hf','revision':'74dd0bf867a4cda7950c17663794267c60cf4b40',
        'max_image_edge':512,'device':'mps','adapter':None,'modes':['whole','repeat3','whole-split'],
        'selection':'fixed page halves, no question/answer-dependent crop selection',
        'control':'repeat3 and whole-split use exactly the same three-image prompt and processor path',
        'scope':'previously inspected 20-question development slice; no held-out performance claim'}
    out=snapshot_run(args.output,{**vars(args),**protocol},[args.manifest]).resolve();commands=[]
    def run(name,script,arguments):
        command=[sys.executable,str(ROOT/'scripts'/script)]+list(map(str,arguments));record={'name':name,'command':command};commands.append(record)
        print('VIEW_STAGE',name,flush=True)
        with (out/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            record['exit_code']=process.wait()
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        if record['exit_code']:raise RuntimeError('Failed stage '+name)
    run('checks','run_checks.py',[]);summaries={};predictions={}
    for mode in protocol['modes']:
        target=out/mode
        run(mode,'infer.py',['--manifest',Path(args.manifest).resolve(),'--model',protocol['model'],'--revision',protocol['revision'],'--device','mps','--max-image-edge',512,'--view',mode,'--output',target])
        run('score-'+mode,'evaluate.py',['--predictions',target/'predictions.jsonl','--output',target/'answer-metrics.json'])
        summaries[mode]=json.loads((target/'answer-metrics.json').read_text())
        predictions[mode]=[json.loads(line) for line in (target/'predictions.jsonl').read_text().splitlines()]
        summaries[mode]['input_tokens']=[r['input_tokens'] for r in predictions[mode]]
    if summaries['repeat3']['input_tokens']!=summaries['whole-split']['input_tokens']:raise ValueError('Three-image control token budgets differ')
    summary={'protocol':protocol,'results':summaries,
        'crop_vs_repeated_control':compare(predictions['repeat3'],predictions['whole-split']),
        'crop_vs_single_page':compare(predictions['whole'],predictions['whole-split'])}
    (out/'experiment-summary.json').write_text(json.dumps(summary,indent=2));print('VIEW_EXPERIMENT_FINISHED',flush=True)
