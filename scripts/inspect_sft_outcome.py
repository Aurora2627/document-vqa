"""Audit saved PyTorch adapter updates and exact generated-answer changes."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run,sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    experiment=Path(args.experiment)
    checkpoint=experiment/'training/adapter.pt'
    manifests=[experiment/(kind+'-'+role)/'predictions.jsonl' for role in ['val','test'] for kind in ['baseline','adapter']]
    out=snapshot_run(args.output,{**vars(args),'adapter_sha256':sha(checkpoint)},manifests)
    adapter=torch.load(checkpoint,map_location='cpu',weights_only=True);state=adapter['state_dict']
    if not state or any(not name.endswith(('.a','.b')) for name in state):raise ValueError('Expected LoRA-only tensors')
    if any(not torch.isfinite(value).all() for value in state.values()):raise ValueError('Nonfinite adapter')
    b=[value for name,value in state.items() if name.endswith('.b')]
    if not b or any(torch.count_nonzero(value)==0 for value in b):raise ValueError('Unchanged zero-initialized B matrix')
    changes={}
    for role in ['val','test']:
        rows={kind:{str(r['question_id']):r for r in map(json.loads,(experiment/(kind+'-'+role)/'predictions.jsonl').read_text().splitlines())} for kind in ['baseline','adapter']}
        if rows['baseline'].keys()!=rows['adapter'].keys():raise ValueError('Question ID mismatch')
        changed=[qid for qid,row in rows['baseline'].items() if row['prediction']!=rows['adapter'][qid]['prediction']]
        changes[role]={'questions':len(rows['baseline']),'raw_predictions_changed':len(changed),'changed_question_ids':changed}
    report={'adapter_sha256':sha(checkpoint),'trainable_parameters':sum(v.numel() for v in state.values()),
        'b_matrices':len(b),'nonzero_b_matrices':sum(torch.count_nonzero(v)>0 for v in b),
        'b_frobenius_norm':float(torch.sqrt(sum(v.float().square().sum() for v in b))),
        'selected_epoch':adapter['best_epoch'],'prediction_changes':changes,
        'interpretation':'Nonzero B proves optimizer updates; generation quality must still be evaluated separately'}
    report['nonzero_b_matrices']=int(report['nonzero_b_matrices'])
    (out/'audit.json').write_text(json.dumps(report,indent=2));print('SFT_AUDIT_FINISHED',report,flush=True)
