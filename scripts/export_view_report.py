"""Export matched document-view and resolution summaries from saved runs."""
import argparse,json,sys
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run,sha
from evaluation import compare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--views',required=True);p.add_argument('--higher-resolution',required=True)
    p.add_argument('--output',required=True);p.add_argument('--audit-output',required=True);args=p.parse_args()
    views=Path(args.views);high=Path(args.higher_resolution)
    files=[views/mode/'predictions.jsonl' for mode in ['whole','repeat3','whole-split']]+[high/'predictions.jsonl']
    audit=snapshot_run(args.audit_output,vars(args),files)
    report=json.loads((views/'experiment-summary.json').read_text())
    rows=[[json.loads(line) for line in path.read_text().splitlines()] for path in files]
    report['results']['whole-768']=json.loads((high/'answer-metrics.json').read_text())
    report['resolution_768_vs_512']=compare(rows[0],rows[3])
    for mode,group,path in zip(['whole','repeat3','whole-split','whole-768'],rows,files):
        tokens=[r['input_tokens'] for r in group]
        report['results'][mode]['input_token_summary']={'min':min(tokens),'median':median(tokens),'max':max(tokens)}
        report['results'][mode]['predictions_sha256']=sha(path)
    report.update(version='0.5.0',date='2026-10-09',framework='PyTorch',launch='VS Code integrated terminal',
        limitations=['All results are on the previously inspected 20-question development set',
                     'Single-image anyres and nested multi-image processing use different feature layouts',
                     'Repeat3 and fixed-halves share the exact prompt and token counts',
                     'No OCR, learned crop selection or evidence localization claim',
                     'Generation time excludes preprocessing/model loading; no peak memory measurement'])
    with Path(args.output).open('x') as f:json.dump(report,f,indent=2)
    (audit/'report-sha256.json').write_text(json.dumps({'sha256':sha(args.output)},indent=2))
    print('VIEW_REPORT_EXPORTED',flush=True)
