"""Actual HF tiny LLaVA forward/backward/checkpoint; no pretrained quality claim."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from modeling import tiny_llava,tiny_batch
from lora import configure_lora,adapter_state,restore_adapter
from run_record import snapshot_run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps');p.add_argument('--output',required=True);args=p.parse_args()
    config={**vars(args),'scope':'random tiny HF LLaVA architecture only; synthetic tensor batch; no document QA accuracy','seed':42,'framework':'PyTorch','launch':'VS Code'}
    out=snapshot_run(args.output,config);torch.manual_seed(42)
    model=tiny_llava().to(args.device);base={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    targets=configure_lora(model,rank=4,alpha=8,dropout=0,train_projector=True)
    batch=tiny_batch(args.device);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    log=[];model.train()
    for step in range(4):
        optimizer.zero_grad(set_to_none=True);result=model(**batch,use_cache=False);result.loss.backward()
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        entry={'step':step+1,'loss':float(result.loss.detach().cpu()),'gradient_l1':sum(float(p.grad.abs().sum().detach().cpu()) for _,p in trainable if p.grad is not None),'device':str(next(model.parameters()).device)}
        if not torch.isfinite(result.loss) or entry['gradient_l1']<=0:raise RuntimeError('Invalid loss/gradient')
        optimizer.step();log.append(entry)
    frozen_vision=model.model.vision_tower.state_dict()
    for name,value in frozen_vision.items():torch.testing.assert_close(value.cpu(),base['model.vision_tower.'+name],rtol=0,atol=0)
    bundle={'state_dict':adapter_state(model),'rank':4,'alpha':8,'dropout':0,'train_projector':True,'targets':targets,'scope':config['scope']}
    torch.save(bundle,out/'adapter.pt');torch.save(base,out/'random-base.pt');model.eval()
    restored=tiny_llava().to(args.device);restored.load_state_dict(base);configure_lora(restored,4,8,0,True)
    restore_adapter(restored,torch.load(out/'adapter.pt',weights_only=True)['state_dict']);restored.eval()
    with torch.inference_mode():
        before=model(**batch,use_cache=False).logits;after=restored(**batch,use_cache=False).logits
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        prediction=model.generate(input_ids=batch['input_ids'][:,:8],attention_mask=batch['attention_mask'][:,:8],pixel_values=batch['pixel_values'],max_new_tokens=3,do_sample=False)
    (out/'training-log.jsonl').write_text('\n'.join(json.dumps(r) for r in log)+'\n')
    (out/'predictions.json').write_text(json.dumps({'generated_token_ids':prediction.cpu().tolist(),'scope':config['scope']}))
    report={'passed':True,'scope':config['scope'],'device':args.device,'steps':4,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'targets':targets,'frozen_vision_exact':True,'reload_exact':True,'losses':[r['loss'] for r in log]}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print('ARCHITECTURE_SMOKE_PASSED',args.device)
