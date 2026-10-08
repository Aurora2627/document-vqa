"""Evaluate saved model predictions without loading a language model."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import score_answer
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    rows=[json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('No predictions')
    scores=[score_answer(r['prediction'],r['answers']) for r in rows]
    out=Path(args.output)
    with out.open('x') as f:json.dump({'count':len(rows),'anls':sum(s['anls'] for s in scores)/len(scores),'exact_match':sum(s['exact_match'] for s in scores)/len(scores)},f,indent=2)
