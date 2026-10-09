"""Single-page LLaVA zero-shot/adapter inference, with optional fixed crops."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from PIL import Image
from modeling import load_pretrained
from batching import prompt_messages,move_batch
from lora import configure_lora,restore_adapter
from data import load_manifest,check_document_splits
from run_record import snapshot_run,sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--model',default='llava-hf/llava-v1.6-mistral-7b-hf');p.add_argument('--revision',required=True);p.add_argument('--device',choices=['cuda','cpu','mps'],default='cuda');p.add_argument('--split',choices=['train','val','test'],default='val');p.add_argument('--adapter');p.add_argument('--view',choices=['whole','top','bottom'],default='whole');p.add_argument('--max-image-edge',type=int,default=0);p.add_argument('--output',required=True);args=p.parse_args()
    rows=load_manifest(args.manifest);check_document_splits(rows);rows=[r for r in rows if r['split']==args.split]
    if not rows:raise ValueError('No selected examples')
    config={**vars(args),'framework':'PyTorch','launch':'VS Code','decoding':'greedy, max_new_tokens=64','evidence':'no localization claim; fixed crops ignore ground-truth answers/regions','image_sha256':{r['question_id']:sha(r['image_path']) for r in rows}}
    if args.adapter:config['adapter_sha256']=sha(args.adapter)
    out=snapshot_run(args.output,config,[args.manifest]);dtype=torch.bfloat16 if args.device=='cuda' and torch.cuda.is_bf16_supported() else (torch.float16 if args.device!='cpu' else torch.float32)
    model,processor=load_pretrained(args.model,args.revision,args.device,dtype)
    if args.adapter:
        adapter=torch.load(args.adapter,map_location='cpu',weights_only=True)
        if adapter['model_id']!=args.model or adapter['revision']!=args.revision:raise ValueError('Adapter/base mismatch')
        configure_lora(model,adapter['rank'],adapter['alpha'],adapter['dropout'],adapter['train_projector']);restore_adapter(model,adapter['state_dict'])
    model.eval()
    with (out/'predictions.jsonl').open('w') as predictions:
        for row in rows:
            with Image.open(row['image_path']) as image:
                image=image.convert('RGB');w,h=image.size
                if args.view=='top':image=image.crop((0,0,w,h//2))
                if args.view=='bottom':image=image.crop((0,h//2,w,h))
                if args.max_image_edge>0:image.thumbnail((args.max_image_edge,args.max_image_edge))
                prompt=processor.apply_chat_template(prompt_messages(row['question']),tokenize=False,add_generation_prompt=True)
                batch=move_batch(processor(text=prompt,images=image,return_tensors='pt'),args.device,dtype)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            if args.device=='mps':torch.mps.synchronize()
            started=time.perf_counter()
            with torch.inference_mode():tokens=model.generate(**batch,max_new_tokens=64,do_sample=False)
            if args.device=='cuda':torch.cuda.synchronize()
            if args.device=='mps':torch.mps.synchronize()
            answer=processor.tokenizer.decode(tokens[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            predictions.write(json.dumps({'question_id':row['question_id'],'document_id':row['document_id'],'answers':row['answers'],'question_types':row.get('question_types',[]),'prediction':answer,'seconds':time.perf_counter()-started,'input_tokens':batch['input_ids'].shape[1],'peak_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None})+'\n');predictions.flush()
            print('PREDICTED',row['question_id'],flush=True)
    print('INFERENCE_FINISHED')
