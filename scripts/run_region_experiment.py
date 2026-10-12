"""Question-guided reading with fixed/shuffled-question controls; development only."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from PIL import Image
from modeling import load_pretrained
from region_reading import locate_regions,read_views,prepare_region_views
from lora import configure_lora,restore_adapter
from data import load_manifest
from run_record import snapshot_run,sha
from evaluation import summarize,compare
from paired_statistics import paired_bootstrap

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True)
    p.add_argument('--adapter');p.add_argument('--device',default='cuda',choices=['mps','cuda','cpu']);args=p.parse_args()
    rows=load_manifest(args.manifest)
    if len(rows)<2 or len({r['document_id'] for r in rows})!=len(rows):raise ValueError('Use at least two documents, one question each')
    if any(row['question'].strip()==rows[(n+1)%len(rows)]['question'].strip() for n,row in enumerate(rows)):
        raise ValueError('Shuffled control must change question text')
    model_id='llava-hf/llava-onevision-qwen2-0.5b-ov-hf';revision='74dd0bf867a4cda7950c17663794267c60cf4b40'
    modes=['whole','selected-crop','fixed-fusion','shuffled-fusion','selected-fusion']
    protocol={'model':model_id,'revision':revision,'locator_edge':512,'reader_edge':768,'crop_fraction':0.6,'modes':modes,
        'selection':'top two next-token digit logits for a 3x3 grid; restricted nine-digit distribution',
        'fixed_regions':[4,6],'shuffle':'next question in manifest order, wrap-around; same document image',
        'adapter_sha256':sha(args.adapter) if args.adapter else None,
        'scope':'previously inspected development documents; no new held-out gain or verified evidence claim'}
    image_hashes={r['question_id']:sha(r['image_path']) for r in rows}
    if any(r.get('image_sha256',image_hashes[r['question_id']])!=image_hashes[r['question_id']] for r in rows):raise ValueError('Image differs from prepared manifest')
    out=snapshot_run(args.output,{**vars(args),**protocol,'image_sha256':image_hashes},[args.manifest]+([args.adapter] if args.adapter else []))
    dtype=torch.float32 if args.device=='cpu' else torch.float16;model,processor=load_pretrained(model_id,revision,args.device,dtype)
    if model.config.model_type!='llava_onevision':raise ValueError('Requires OneVision multi-image processor')
    if args.adapter:
        checkpoint=torch.load(args.adapter,map_location='cpu',weights_only=True)
        if checkpoint['model_id']!=model_id or checkpoint['revision']!=revision:raise ValueError('Adapter mismatch')
        configure_lora(model,checkpoint['rank'],checkpoint['alpha'],checkpoint['dropout'],checkpoint['train_projector']);restore_adapter(model,checkpoint['state_dict'])
    model.eval();predictions={mode:[] for mode in modes};handles={mode:(out/(mode+'.jsonl')).open('x') for mode in modes}
    try:
        with (out/'regions.jsonl').open('x') as regions:
            for index,row in enumerate(rows):
                with Image.open(row['image_path']) as raw:image=raw.convert('RGB')
                guided=locate_regions(model,processor,image,row['question'],args.device,dtype)
                shuffled=locate_regions(model,processor,image,rows[(index+1)%len(rows)]['question'],args.device,dtype)
                sample=out/'candidate-regions'/row['question_id'];sample.mkdir(parents=True)
                _,boxes=prepare_region_views(image,guided['ranking'][:2]);guided['pixel_boxes']=boxes
                for n,box in enumerate(boxes):image.crop(box).save(sample/('region-'+str(n+1)+'.png'))
                regions.write(json.dumps({'question_id':row['question_id'],'image_size':list(image.size),'guided':guided,'shuffled':shuffled,
                    'shuffled_question_id':rows[(index+1)%len(rows)]['question_id'],'evidence_status':'unverified candidate; no ground-truth region used'})+'\n');regions.flush()
                specifications={'whole':([],True,None),'selected-crop':(guided['ranking'][:1],False,guided),
                    'fixed-fusion':([4,6],True,None),'shuffled-fusion':(shuffled['ranking'][:2],True,shuffled),'selected-fusion':(guided['ranking'][:2],True,guided)}
                for mode,(ids,whole,locator) in specifications.items():
                    views,used_boxes=prepare_region_views(image,ids,protocol['reader_edge'],whole)
                    prediction=read_views(model,processor,views,row['question'],args.device,dtype)
                    prediction.update(question_id=row['question_id'],document_id=row['document_id'],answers=row['answers'],question_types=row.get('question_types',[]),
                        selected_regions=ids,pixel_boxes=used_boxes,locator_seconds=locator['seconds'] if locator else 0.,
                        locator_input_tokens=locator['input_tokens'] if locator else 0)
                    prediction['pipeline_wall_seconds']=prediction['reader_wall_seconds']+prediction['locator_seconds']
                    # Locator work is shared by modes here; report each method's standalone cost, not amortized total runtime.
                    predictions[mode].append(prediction);handles[mode].write(json.dumps(prediction)+'\n');handles[mode].flush()
                if len({predictions[mode][-1]['input_tokens'] for mode in ['fixed-fusion','shuffled-fusion','selected-fusion']})!=1:
                    raise ValueError('Fusion reader token budgets differ')
                print('REGION_PROGRESS',index+1,'/',len(rows),flush=True)
    finally:
        for handle in handles.values():handle.close()
    from statistics import median
    results={mode:{**summarize(group),'pipeline_wall_seconds_median':median(r['pipeline_wall_seconds'] for r in group),
        'total_input_tokens_median':median(r['input_tokens']+r['locator_input_tokens'] for r in group)} for mode,group in predictions.items()}
    contrasts=[('whole','selected-crop'),('whole','selected-fusion'),('fixed-fusion','selected-fusion'),('shuffled-fusion','selected-fusion')]
    summary={'version':'0.7.0','protocol':protocol,'results':results,
        'comparisons':{a+'_to_'+b:{'paired':compare(predictions[a],predictions[b]),'uncertainty':paired_bootstrap(predictions[a],predictions[b])} for a,b in contrasts},
        'same_top1_under_shuffled_question':sum(r1==r2 for r1,r2 in [(json.loads(line)['guided']['ranking'][0],json.loads(line)['shuffled']['ranking'][0]) for line in (out/'regions.jsonl').read_text().splitlines()]),
        'prediction_sha256':{mode:sha(out/(mode+'.jsonl')) for mode in modes},
        'limitations':['20 previously inspected development documents; exploratory comparisons, no multiplicity correction',
            'Single-image anyres differs from native multi-image processing; fixed/shuffled fusion share the reader path',
            'Digit logits are a zero-shot locator, not trained region supervision or calibrated confidence',
            'No ground-truth region annotations, no IoU or verified grounding claim',
            'Pipeline cost includes locator and reader preprocessing/forward/generation; excludes model load and crop saving']}
    (out/'experiment-summary.json').write_text(json.dumps(summary,indent=2))
    with Path(args.report).open('x') as output:json.dump(summary,output,indent=2)
    print('REGION_EXPERIMENT_FINISHED',{mode:result['exact_match'] for mode,result in results.items()},flush=True)
if __name__=='__main__':main()
