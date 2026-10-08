"""Reload the saved adapter into a fresh real pretrained model process."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from modeling import load_pretrained
from lora import configure_lora,restore_adapter
from batching import build_example,move_batch,answer_loss
from data import load_manifest
from run_record import snapshot_run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args();root=Path(__file__).resolve().parents[1]
    run=Path(args.run);bundle=torch.load(run/'adapter.pt',map_location='cpu',weights_only=True);previous=json.loads((run/'metrics.json').read_text())
    out=snapshot_run(args.output,{'scope':'fresh process pretrained adapter reload; synthetic fixture only','source_run':str(run),'model_id':bundle['model_id'],'revision':bundle['revision'],'device':'mps'},[root/'data/sample/fixtures.jsonl'])
    model,processor=load_pretrained(bundle['model_id'],bundle['revision'],'mps',torch.float16)
    configure_lora(model,bundle['rank'],bundle['alpha'],bundle['dropout'],bundle['train_projector']);restore_adapter(model,bundle['state_dict']);model.eval()
    row=[r for r in load_manifest(root/'data/sample/fixtures.jsonl') if r['split']=='train'][0]
    batch=move_batch(build_example(processor,row,max_image_edge=384),'mps',torch.float16)
    with torch.inference_mode():loss=float(answer_loss(model,batch).cpu())
    if abs(loss-previous['loss_after_step'])>1e-6:raise RuntimeError('Fresh-model reload output differs')
    report={'passed':True,'device':'mps','scope':'fresh model checkpoint reload, synthetic fixture only','loss':loss,'previous_loss':previous['loss_after_step'],'absolute_difference':abs(loss-previous['loss_after_step'])}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print('FRESH_PRETRAINED_RELOAD_PASSED')
