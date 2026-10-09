"""Evaluate saved model predictions without loading a language model."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation import summarize,compare
from run_record import sha,snapshot_run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--compare-to');p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    snapshot_run(out.parent/(out.stem+'-evaluation'),vars(args),[args.predictions]+([args.compare_to] if args.compare_to else []))
    rows=[json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    report=summarize(rows);report['predictions_sha256']=sha(args.predictions)
    if args.compare_to:
        left=[json.loads(line) for line in Path(args.compare_to).read_text().splitlines() if line.strip()]
        report['paired_comparison']=compare(left,rows);report['comparison_predictions_sha256']=sha(args.compare_to)
    with out.open('x') as f:json.dump(report,f,indent=2)
