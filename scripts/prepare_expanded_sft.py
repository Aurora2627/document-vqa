"""Expand nested real training documents and freeze new document-disjoint val/test."""
import argparse,concurrent.futures,json,shutil,sys
from pathlib import Path
from urllib.parse import urlencode
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_docvqa import fetch
from prepare_sft_data import select_documents,DATASET,VALIDATION_DATASET
from data import load_manifest,check_document_splits
from run_record import snapshot_run,sha
TRAIN_OFFSETS=[0,1000,2000,3000,4000,5000]
VAL_OFFSETS=[200,400,600,800,1200,1400,1600,1800,2200,2400,2600,2800,3200,3400]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--previous-train',required=True)
    p.add_argument('--exclude',nargs='+',required=True);args=p.parse_args()
    previous=load_manifest(args.previous_train);excluded=[r for path in args.exclude for r in load_manifest(path)]
    excluded_docs={r['document_id'] for r in excluded};old_docs={r['document_id'] for r in previous}
    if excluded_docs&old_docs:raise ValueError('Previous train overlaps excluded evaluation')
    config={**vars(args),'train_documents':64,'max_new_questions_per_document':2,'val_documents':24,'test_documents':64,
        'train_offsets':TRAIN_OFFSETS,'validation_offsets':VAL_OFFSETS,'selection_seed':42,
        'selection':'retain every old training question; add SHA256-sorted source documents; new val/test locked before training',
        'scope':'labeled validation mirror, not official blind test'}
    out=snapshot_run(args.output,config,[args.previous_train]+args.exclude);raw=out/'raw';raw.mkdir();images=out/'images';images.mkdir()
    endpoints={'train':'https://huggingface.co/api/datasets/'+DATASET,'validation':'https://huggingface.co/api/datasets/'+VALIDATION_DATASET}
    candidates={};revisions={}
    for split,offsets in [('train',TRAIN_OFFSETS),('validation',VAL_OFFSETS)]:
        fetch(endpoints[split],raw/(split+'-before.json'));revisions[split]=json.loads((raw/(split+'-before.json')).read_text())['sha'];candidates[split]=[]
        for offset in offsets:
            query=urlencode({'dataset':DATASET if split=='train' else VALIDATION_DATASET,'config':'default' if split=='train' else 'DocVQA','split':split,'offset':offset,'length':100})
            path=raw/(split+'-'+str(offset)+'.json');fetch('https://datasets-server.huggingface.co/rows?'+query,path)
            payload=json.loads(path.read_text())
            if [i['row_idx'] for i in payload['rows']]!=list(range(offset,offset+100)):raise ValueError('Incomplete source page')
            for item in payload['rows']:
                if split=='validation':
                    row=item['row'];item['row']={**row,'question_id':row['questionId'],'other_metadata':{'ucsf_document_id':row['ucsf_document_id'],'doc_id':row['docId']}}
            candidates[split]+=payload['rows'];print('EXPANDED_CANDIDATES',split,offset,flush=True)
    additional=select_documents(candidates['train'],64-len(old_docs),excluded_docs|old_docs,2)
    train_docs=old_docs|{str(i['row']['other_metadata']['ucsf_document_id']) for i in additional}
    evaluation=select_documents(candidates['validation'],88,excluded_docs|train_docs,1)
    selected=[('train',i) for i in additional]+[('val' if n<24 else 'test',i) for n,i in enumerate(evaluation)]
    blocked_hashes={sha(r['image_path']) for r in excluded};rows=[]
    for row in previous:
        path=images/(row['question_id']+'.png');shutil.copy2(row['image_path'],path)
        rows.append({k:v for k,v in {**row,'image':'images/'+path.name,'image_sha256':sha(path)}.items() if k!='image_path'})
    def download(pair):
        role,item=pair;row=item['row'];qid=str(row['question_id']);meta=row['other_metadata'];path=images/(qid+'.png')
        fetch(row['image']['src'],path)
        with Image.open(path) as image:image.load();size=list(image.size)
        if size!=[row['image']['width'],row['image']['height']]:raise ValueError('Image size mismatch')
        if row['data_split']!=('train' if role=='train' else 'val'):raise ValueError('Wrong source split')
        return {'question_id':qid,'document_id':str(meta['ucsf_document_id']),'page_id':str(meta['doc_id']),
            'image':'images/'+path.name,'question':row['question'],'answers':row['answers'],'split':role,
            'source_split':'train' if role=='train' else 'validation','source_row':item['row_idx'],
            'question_types':row.get('question_types',[]),'image_sha256':sha(path),'image_size':size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows+=list(pool.map(download,selected))
    hashes={};ids=set();excluded_ids={r['question_id'] for r in excluded}
    for row in rows:
        if row['question_id'] in ids or row['question_id'] in excluded_ids:raise ValueError('Duplicate/excluded question')
        ids.add(row['question_id']);digest=row['image_sha256']
        if digest in blocked_hashes:raise ValueError('Image overlaps previous evaluation')
        if hashes.setdefault(digest,row['split'])!=row['split']:raise ValueError('Image leakage across roles')
    groups={role:[r for r in rows if r['split']==role] for role in ['train','val','test']}
    for role,group in groups.items():(out/(role+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in group))
    check_document_splits(*(load_manifest(out/(role+'.jsonl')) for role in groups))
    for split,endpoint in endpoints.items():
        fetch(endpoint,raw/(split+'-after.json'))
        if json.loads((raw/(split+'-after.json')).read_text())['sha']!=revisions[split]:raise ValueError('Repository changed')
    report={'dataset':DATASET,'validation_dataset':VALIDATION_DATASET,'repository_revisions_observed':revisions,'viewer_revision_pinned':False,
        'retained_training_questions':len(previous),'excluded_documents':len(excluded_docs),
        'raw_response_sha256':{p.name:sha(p) for p in raw.glob('*.json')},
        'splits':{role:{'questions':len(group),'source_documents':len({r['document_id'] for r in group}),'manifest_sha256':sha(out/(role+'.jsonl'))} for role,group in groups.items()},
        'scope':'expanded nested training set; new validation and reserved labeled-validation evaluation, not official blind test'}
    (out/'dataset-summary.json').write_text(json.dumps(report,indent=2));print('EXPANDED_DATA_PREPARED',report['splits'],flush=True)
if __name__=='__main__':main()
