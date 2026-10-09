"""PyTorch LLaVA supervised fine-tuning: answer masking, frozen vision, LoRA+projector."""
import argparse,json,sys,random
from contextlib import nullcontext
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from modeling import load_pretrained
from batching import build_example,move_batch,answer_loss
from data import load_manifest,check_document_splits
from lora import configure_lora,adapter_state
from run_record import snapshot_run,sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cuda','mps'],default='cuda');p.add_argument('--max-image-edge',type=int,default=0);p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--model',default='llava-hf/llava-v1.6-mistral-7b-hf');p.add_argument('--revision',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=2);p.add_argument('--rank',type=int,default=8);p.add_argument('--alpha',type=int,default=16);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--max-length',type=int,default=4096);p.add_argument('--seed',type=int,default=42);p.add_argument('--train-projector',action='store_true');p.add_argument('--gradient-checkpointing',action='store_true');args=p.parse_args()
    if args.epochs<1 or args.accumulation<1:raise ValueError('Positive epochs and accumulation required')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    train=load_manifest(args.train);val=load_manifest(args.val);check_document_splits(train,val)
    if any(r['split']!='train' for r in train) or any(r['split']!='val' for r in val):raise ValueError('Use separate train/val manifests; never train on test')
    config={**vars(args),'device':args.device,'framework':'PyTorch','launch':'VS Code terminal or Remote SSH','train_projector':args.train_projector,'dropout':.05,'selection':'lowest validation answer-token NLL; ANLS evaluated separately','image_sha256':{r['question_id']:sha(r['image_path']) for r in train+val}}
    out=snapshot_run(args.output,config,[args.train,args.val]);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    dtype=torch.bfloat16 if args.device=='cuda' and torch.cuda.is_bf16_supported() else torch.float16
    if args.train_projector and dtype==torch.float16:raise ValueError('Joint projector training requires bf16 CUDA in this entry; otherwise use LoRA-only')
    model,processor=load_pretrained(args.model,args.revision,args.device,dtype);model.config.use_cache=False
    targets=configure_lora(model,args.rank,args.alpha,.05,args.train_projector)
    if args.gradient_checkpointing:
        # Non-reentrant checkpointing supports frozen embeddings with trainable LoRA.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    scaler=torch.amp.GradScaler('cuda',enabled=args.device=='cuda' and dtype==torch.float16)
    trainable=[p for p in model.parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for p in trainable)
    print('TRAINABLE_PARAMETERS',trainable_count,flush=True)
    def validation_nll():
        model.eval();total=0.;count=0
        with torch.inference_mode():
            for row in val:
                batch=move_batch(build_example(processor,row,args.max_length,args.max_image_edge),args.device,dtype)
                with (torch.autocast('cuda',dtype=dtype) if args.device=='cuda' else nullcontext()):loss=answer_loss(model,batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite validation loss')
                tokens=int((batch['labels'][:,1:]!=-100).sum());total+=float(loss.cpu())*tokens;count+=tokens
        return total/count
    best=float('inf');rng=random.Random(args.seed);optimizer_steps=0;best_epoch=None
    checkpoints=out/'checkpoints';checkpoints.mkdir()
    with (out/'training-log.jsonl').open('w') as log:
        initial_nll=validation_nll()
        log.write(json.dumps({'epoch':0,'validation_answer_token_nll':initial_nll})+'\n');log.flush()
        print('INITIAL_VALIDATION_NLL',initial_nll,flush=True)
        for epoch in range(args.epochs):
            order=list(range(len(train)));rng.shuffle(order);model.train();optimizer.zero_grad(set_to_none=True)
            for step,index in enumerate(order):
                batch=move_batch(build_example(processor,train[index],args.max_length,args.max_image_edge),args.device,dtype)
                window_start=(step//args.accumulation)*args.accumulation;window_size=min(args.accumulation,len(order)-window_start)
                with (torch.autocast('cuda',dtype=dtype) if args.device=='cuda' else nullcontext()):loss=answer_loss(model,batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                scaler.scale(loss/window_size).backward()
                if (step+1)%args.accumulation==0 or step+1==len(order):
                    scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    optimizer_steps+=1
                log.write(json.dumps({'epoch':epoch+1,'example':step+1,'answer_loss':float(loss.detach().cpu()),'answer_tokens':int((batch['labels'][:,1:]!=-100).sum()),'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None})+'\n');log.flush()
                if (step+1)%args.accumulation==0 or step+1==len(order):
                    log.write(json.dumps({'epoch':epoch+1,'optimizer_step':optimizer_steps,'gradient_norm_before_clip':float(norm.cpu())})+'\n');log.flush()
                    print('TRAIN_PROGRESS',epoch+1,step+1,'/',len(order),flush=True)
            score=validation_nll()
            log.write(json.dumps({'epoch':epoch+1,'validation_answer_token_nll':score})+'\n');log.flush()
            checkpoint={'state_dict':adapter_state(model),'model_id':args.model,'revision':args.revision,'rank':args.rank,'alpha':args.alpha,'dropout':.05,'train_projector':args.train_projector,'targets':targets,'best_epoch':epoch+1,'val_nll':score}
            torch.save(checkpoint,checkpoints/('epoch-'+str(epoch+1)+'.pt'))
            print('EPOCH_VALIDATION_NLL',epoch+1,score,flush=True)
            if score<best:
                best=score;best_epoch=epoch+1;torch.save(checkpoint,out/'adapter.pt')
    (out/'metrics.json').write_text(json.dumps({'initial_validation_nll':initial_nll,'best_validation_nll':best,'best_epoch':best_epoch,'optimizer_steps':optimizer_steps,'trainable_parameters':trainable_count,'train_size':len(train),'val_size':len(val),'selection':'lowest trained-epoch validation NLL; compare to frozen baseline separately'},indent=2));print('LORA_TRAINING_FINISHED')
