"""Locked resolution/LoRA factorial evaluation and auditable local failure inventory."""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from evaluation import summarize, compare
from paired_statistics import paired_bootstrap
from run_record import snapshot_run, sha
from data import load_manifest
from metrics import score_answer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--report',required=True); p.add_argument('--adapter'); args=p.parse_args()
    protocol={'model':'llava-hf/llava-onevision-qwen2-0.5b-ov-hf', 'revision':'74dd0bf867a4cda7950c17663794267c60cf4b40',
        'edges':[512,768], 'view':'whole', 'adapter_sha256':sha(args.adapter) if args.adapter else None,
        'adapter_selection':'existing v0.4 epoch-2 checkpoint, selected using previous validation NLL; no new training',
        'device':'mps', 'generation':'greedy max_new_tokens=64',
        'scope':'new document-disjoint labeled validation subset; no official benchmark claim',
        'primary_comparison':'base-768 versus base-512; other paired contrasts exploratory'}
    inputs=[args.manifest]+([args.adapter] if args.adapter else [])
    # Checkpoint remains in ignored run inputs, never in a public report.
    out=snapshot_run(args.output,{**vars(args),**protocol},inputs).resolve(); commands=[]
    def run(name,script,arguments):
        command=[sys.executable,str(ROOT/'scripts'/script)]+list(map(str,arguments)); record={'name':name,'command':command}; commands.append(record)
        print('INDEPENDENT_STAGE',name,flush=True)
        with (out/(name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout: log.write(line); log.flush(); print(line,end='',flush=True)
            record['exit_code']=process.wait()
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        if record['exit_code']: raise RuntimeError('Failed stage '+name)
    run('checks','run_checks.py',[]); predictions={}; results={}
    for kind in ['base']+(['adapter'] if args.adapter else []):
        for edge in protocol['edges']:
            name=kind+'-'+str(edge); target=out/name
            arguments=['--manifest',Path(args.manifest).resolve(),'--split','test','--model',protocol['model'],
                '--revision',protocol['revision'],'--device','mps','--max-image-edge',edge,'--output',target]
            if kind=='adapter': arguments+=['--adapter',Path(args.adapter).resolve()]
            run(name,'infer.py',arguments)
            predictions[name]=[json.loads(line) for line in (target/'predictions.jsonl').read_text().splitlines()]
            results[name]=summarize(predictions[name])
            (target/'answer-metrics.json').write_text(json.dumps(results[name],indent=2))
    manifest={r['question_id']:r for r in load_manifest(args.manifest)}
    if any({r['question_id'] for r in group} != manifest.keys() for group in predictions.values()): raise ValueError('Incomplete prediction coverage')
    contrasts=[('base-512','base-768')]
    if args.adapter: contrasts += [('base-512','adapter-512'),('base-768','adapter-768'),('adapter-512','adapter-768')]
    comparisons={a+'_to_'+b:{'paired':compare(predictions[a],predictions[b]),
        'uncertainty':paired_bootstrap(predictions[a],predictions[b])} for a,b in contrasts}
    lookup={name:{r['question_id']:r for r in group} for name,group in predictions.items()}
    failures=[]
    for key,row in manifest.items():
        scores={name:score_answer(group[key]['prediction'],row['answers']) for name,group in lookup.items()}
        if any(not score['exact_match'] for score in scores.values()):
            failures.append({'question_id':key,'question':row['question'],'image':str(row['image_path']),
                'references':row['answers'],'predictions':{name:group[key]['prediction'] for name,group in lookup.items()},
                'scores':scores,'question_types':row.get('question_types',[]),
                'review_status':'unreviewed; no automatic visual-error attribution'})
    (out/'failure-review.json').write_text(json.dumps(failures,indent=2))
    summary={'version':'0.6.0','protocol':protocol,'manifest_sha256':sha(args.manifest),'results':results,
        'comparisons':comparisons,'failure_review_count':len(failures),
        'prediction_sha256':{name:sha(out/name/'predictions.jsonl') for name in predictions},
        'limitations':['100 questions, one per source document, from a finite 800-row candidate pool',
            'Public official-validation mirror, not official blind test; viewer revision not pinned',
            'ANLS-style evaluator; official parity not verified',
            'Once evaluated, this slice is no longer pristine for future tuning',
            'Bootstrap intervals describe individual paired contrasts; exploratory comparisons not multiplicity-adjusted',
            'Generation timing excludes load and preprocessing; no peak MPS memory measurement']}
    (out/'experiment-summary.json').write_text(json.dumps(summary,indent=2))
    with Path(args.report).open('x') as report: json.dump(summary,report,indent=2)
    print('INDEPENDENT_EVAL_FINISHED',json.dumps({e:{k:v for k,v in r.items() if k in ['count','anls','exact_match']} for e,r in results.items()}),flush=True)
if __name__=='__main__': main()
