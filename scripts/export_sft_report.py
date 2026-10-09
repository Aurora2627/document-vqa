"""Copy the completed experiment's non-sensitive summary to a public report."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    experiment=Path(args.experiment)
    report=json.loads((experiment/'experiment-summary.json').read_text())
    commands=json.loads((experiment/'commands.json').read_text())
    if not commands or any(r.get('exit_code')!=0 for r in commands):raise ValueError('Experiment stages did not all succeed')
    expected=report['dataset']['splits']['test']['questions']
    if report['results']['test']['baseline']['count']!=expected or report['results']['test']['adapter']['count']!=expected:
        raise ValueError('Reserved evaluation incomplete')
    report.update(version='0.4.0',date='2026-10-09',local_experiment=experiment.name,
        limitations=['Small deterministic candidate pool, not a representative full benchmark',
                     'Observed mirror revisions and recorded asset hashes; viewer responses not revision-pinned',
                     'ANLS-style metric, official evaluator parity not verified',
                     'This evaluation subset must become development data after its results are inspected'])
    with Path(args.output).open('x') as f:json.dump(report,f,indent=2)
    print('SFT_REPORT_EXPORTED')
