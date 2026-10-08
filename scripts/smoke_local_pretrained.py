"""One real small pretrained LLaVA LoRA step on a synthetic software fixture."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from modeling import load_pretrained
from batching import build_example,move_batch,answer_loss
from lora import configure_lora,adapter_state,restore_adapter
from data import load_manifest
from run_record import snapshot_run,sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();root=Path(__file__).resolve().parents[1]
    info=json.loads((root/'reports/local-model.json').read_text());manifest=root/'data/sample/fixtures.jsonl'
    row=[r for r in load_manifest(manifest) if r['split']=='train'][0]
    config={'scope':'single LoRA step on synthetic fixture; hardware/training compatibility only; no QA quality claim','model_id':info['model_id'],'revision':info['revision'],'device':'mps','dtype':'float16 base/float32 LoRA','rank':4,'alpha':8,'train_projector':False,'max_image_edge':384,'seed':42,'image_sha256':sha(row['image_path'])}
    out=snapshot_run(args.output,config,[manifest]);torch.manual_seed(42)
    model,processor=load_pretrained(info['model_id'],info['revision'],'mps',torch.float16)
    targets=configure_lora(model,4,8,0,False);model.config.use_cache=False;model.train()
    batch=move_batch(build_example(processor,row,max_image_edge=384),'mps',torch.float16)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    started=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss=answer_loss(model,batch)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward();gradient=sum(float(p.grad.abs().sum().detach().cpu()) for p in model.parameters() if p.requires_grad and p.grad is not None)
    if gradient<=0:raise RuntimeError('No adapter gradient')
    optimizer.step();torch.mps.synchronize()
    state=adapter_state(model)
    bundle={**config,'state_dict':state,'dropout':0,'targets':targets,'best_epoch':1};torch.save(bundle,out/'adapter.pt')
    model.eval()
    with torch.inference_mode():reference=answer_loss(model,batch).detach().cpu()
    restored=torch.load(out/'adapter.pt',map_location='cpu',weights_only=True);restore_adapter(model,restored['state_dict'])
    with torch.inference_mode():torch.testing.assert_close(answer_loss(model,batch).cpu(),reference,rtol=0,atol=0)
    report={'passed':True,'scope':config['scope'],'device':'mps','input_tokens':int(batch['input_ids'].shape[1]),'answer_tokens':int((batch['labels']!=-100).sum()),'loss_before_step':float(loss.detach().cpu()),'loss_after_step':float(reference),'adapter_gradient_l1':gradient,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'elapsed_seconds':time.perf_counter()-started,'mps_current_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory(),'reload_same_model_exact':True}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));(out/'training-log.jsonl').write_text(json.dumps(report)+'\n');print('PRETRAINED_LOCAL_LORA_SMOKE_PASSED')
