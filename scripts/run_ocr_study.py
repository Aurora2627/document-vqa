"""OCR-localized crops and text/image fusion, with no-answer and shuffled-question controls."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from PIL import Image
from modeling import load_pretrained
from lora import configure_lora,restore_adapter
from data import load_manifest
from run_record import snapshot_run,sha
from region_reading import read_views
from ocr_regions import retrieve_regions,context_question
from evaluation import summarize,compare
from paired_statistics import paired_bootstrap

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--ocr',required=True);p.add_argument('--output',required=True)
    p.add_argument('--report',required=True);p.add_argument('--adapter');p.add_argument('--wait-for-study');p.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda');args=p.parse_args()
    if args.wait_for_study:
        deadline=time.monotonic()+7200;path=Path(args.wait_for_study)/'commands.json';print('WAITING_FOR_PRIOR_STUDY',flush=True)
        while True:
            commands=json.loads(path.read_text()) if path.exists() else []
            if any(r.get('exit_code',0)!=0 for r in commands):raise RuntimeError('Prior study failed; do not continue silently')
            if any(r['name']=='trained-region-dev' and r.get('exit_code')==0 for r in commands):break
            if time.monotonic()>deadline:raise RuntimeError('Prior study wait timed out')
            time.sleep(5)
    rows=load_manifest(args.manifest);ocr=Path(args.ocr)
    if len(rows)<2 or len({r['document_id'] for r in rows})!=len(rows):raise ValueError('One question per source document required')
    if any(r['question'].strip()==rows[(n+1)%len(rows)]['question'].strip() for n,r in enumerate(rows)):raise ValueError('Control question must differ')
    records={r['question_id']:ocr/'records'/(r['question_id']+'.json') for r in rows}
    for row in rows:
        if json.loads(records[row['question_id']].read_text())['image_sha256']!=sha(row['image_path']):raise ValueError('OCR/document mismatch')
    modes=['whole','question-only','ocr-only','ocr-crop','whole-ocr','shuffled-ocr']
    protocol={'model':'llava-hf/llava-onevision-qwen2-0.5b-ov-hf','revision':'74dd0bf867a4cda7950c17663794267c60cf4b40',
        'device':args.device,'reader_edge':768,'ocr':json.loads((ocr/'config.json').read_text()),
        'retrieval':'PyTorch BM25 over line plus neighbours, full-width bands +/-18% page height, top 2; nearest 8 lines per band, coordinates in percent',
        'context_limit':'128 tokens per retrieved band, at most two bands; explicit truncation recorded',
        'adapter_sha256':sha(args.adapter) if args.adapter else None,'modes':modes,
        'scope':'previously inspected 20-document development slice; exploratory ablation only',
        'evidence_status':'unverified OCR candidates; no reference answer or true box used in selection'}
    inputs=[args.manifest,ocr/'ocr-index.jsonl',ocr/'models.json']+list(records.values())+([args.adapter] if args.adapter else [])
    out=snapshot_run(args.output,{**vars(args),**protocol},inputs)
    model,processor=load_pretrained(protocol['model'],protocol['revision'],args.device,torch.float16)
    if args.adapter:
        checkpoint=torch.load(args.adapter,map_location='cpu',weights_only=True)
        if checkpoint['model_id']!=protocol['model'] or checkpoint['revision']!=protocol['revision']:raise ValueError('Adapter mismatch')
        configure_lora(model,checkpoint['rank'],checkpoint['alpha'],checkpoint['dropout'],checkpoint['train_projector']);restore_adapter(model,checkpoint['state_dict'])
    model.eval();predictions={mode:[] for mode in modes};handles={mode:(out/(mode+'.jsonl')).open('x') for mode in modes}
    def context(selection):
        parts=[];original=used=0
        for region in selection['regions']:
            ids=processor.tokenizer.encode(region['context'],add_special_tokens=False);original+=len(ids);used+=min(128,len(ids))
            parts.append(processor.tokenizer.decode(ids[:128],skip_special_tokens=True))
        return '\n'.join(parts),{'original_tokens':original,'used_tokens':used,'truncated':original>used}
    try:
        with (out/'retrieved-regions.jsonl').open('x') as audit:
            for n,row in enumerate(rows):
                with Image.open(row['image_path']) as source:image=source.convert('RGB')
                record=json.loads(records[row['question_id']].read_text());start=time.perf_counter()
                guided=retrieve_regions(record['items'],row['question'],image.size)
                guided_context,guided_budget=context(guided);retrieval_seconds=time.perf_counter()-start
                shuffled_start=time.perf_counter();shuffled=retrieve_regions(record['items'],rows[(n+1)%len(rows)]['question'],image.size)
                shuffled_context,shuffled_budget=context(shuffled)
                shuffled_seconds=time.perf_counter()-shuffled_start
                whole=image.copy();whole.thumbnail((768,768));crop=image.crop(guided['regions'][0]['box']) if guided['regions'] else image.copy();crop.thumbnail((768,768))
                directory=out/'candidate-regions'/row['question_id'];directory.mkdir(parents=True);crop.save(directory/'candidate.png')
                audit.write(json.dumps({'question_id':row['question_id'],'guided':guided,'shuffled':shuffled,'guided_budget':guided_budget,'shuffled_budget':shuffled_budget})+'\n');audit.flush()
                specifications={'whole':([whole],row['question'],False),'question-only':([],row['question'],False),
                    'ocr-only':([],context_question(row['question'],guided_context),True),'ocr-crop':([crop],row['question'],True),
                    'whole-ocr':([whole],context_question(row['question'],guided_context),True),
                    'shuffled-ocr':([whole],context_question(row['question'],shuffled_context),True)}
                for mode,(views,question,uses_ocr) in specifications.items():
                    prediction=read_views(model,processor,views,question,args.device,torch.float16)
                    prediction.update(question_id=row['question_id'],document_id=row['document_id'],answers=row['answers'],question_types=row.get('question_types',[]),
                        ocr_seconds=record['seconds'] if uses_ocr else 0.,
                        retrieval_seconds=(shuffled_seconds if mode=='shuffled-ocr' else retrieval_seconds) if uses_ocr else 0.)
                    prediction['pipeline_wall_seconds']=prediction['reader_wall_seconds']+prediction['ocr_seconds']+prediction['retrieval_seconds']
                    predictions[mode].append(prediction);handles[mode].write(json.dumps(prediction)+'\n');handles[mode].flush()
                print('OCR_STUDY_PROGRESS',n+1,'/',len(rows),flush=True)
    finally:
        for handle in handles.values():handle.close()
    from statistics import median
    results={mode:{**summarize(group),'pipeline_wall_seconds_median':median(r['pipeline_wall_seconds'] for r in group),
        'reader_input_tokens_median':median(r['input_tokens'] for r in group)} for mode,group in predictions.items()}
    pairs=[('whole','ocr-crop'),('whole','whole-ocr'),('ocr-only','whole-ocr'),('shuffled-ocr','whole-ocr'),('question-only','ocr-only')]
    summary={'version':'0.7.0','protocol':protocol,'results':results,
        'comparisons':{a+'_to_'+b:{'paired':compare(predictions[a],predictions[b]),'uncertainty':paired_bootstrap(predictions[a],predictions[b])} for a,b in pairs},
        'prediction_sha256':{mode:sha(out/(mode+'.jsonl')) for mode in modes},
        'limitations':['Development results only; no independent OCR/region gain claim',
            'BM25 is lexical retrieval, not learned grounding; OCR bands are unverified candidates',
            'OCR-only and question-only use the same language model without image tokens',
            'Context length varies, token budgets recorded; no perfectly matched total-input-cost claim',
            'OCR computation is cached in this experiment but included in per-request pipeline cost; excludes weight/model loading',
            'No IoU because no true evidence boxes; exploratory bootstrap intervals not adjusted for multiple comparisons']}
    (out/'experiment-summary.json').write_text(json.dumps(summary,indent=2))
    with Path(args.report).open('x') as public:json.dump(summary,public,indent=2)
    print('OCR_STUDY_FINISHED',{mode:result['exact_match'] for mode,result in results.items()},flush=True)
if __name__=='__main__':main()
