"""Build small real DocVQA SFT splits without touching the old development set."""
import argparse, concurrent.futures, hashlib, json, sys, shutil
from pathlib import Path
from urllib.parse import urlencode
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_docvqa import fetch
from data import load_manifest,check_document_splits
from run_record import snapshot_run,sha
DATASET='pixparse/docvqa-single-page-questions'
VALIDATION_DATASET='lmms-lab-encoder/DocVQA'
OFFSETS={'train':[0,1000,2000], 'validation':[1000,2000,3000]}

def select_documents(items,count,excluded,questions_per_document,seed=42):
    groups={}
    for item in items:
        if set(item.get('truncated_cells',[]))-{'ocr_results'}:
            raise ValueError('Truncated required fields')
        row=item['row'];doc=str(row['other_metadata']['ucsf_document_id'])
        if doc not in excluded:groups.setdefault(doc,{})[str(row['question_id'])]=item
    ordered=sorted(groups,key=lambda doc:hashlib.sha256((str(seed)+':'+doc).encode()).hexdigest())
    if len(ordered)<count:raise ValueError('Insufficient distinct source documents')
    return [item for doc in ordered[:count] for item in sorted(groups[doc].values(),key=lambda r:r['row_idx'])[:questions_per_document]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--exclude-manifest',required=True);p.add_argument('--reuse-raw');args=p.parse_args()
    old=load_manifest(args.exclude_manifest);excluded={r['document_id'] for r in old}
    out=snapshot_run(args.output,{**vars(args),'dataset':DATASET,'validation_dataset':VALIDATION_DATASET,'offsets':OFFSETS,'page_length':100,'seed':42,
        'train_documents':32,'max_train_questions_per_document':2,'validation_documents':12,'evaluation_documents':20,
        'selection':'SHA256(seed:source_document) ordering within fixed candidate pages; no answer-based selection',
        'evaluation_source':'labeled official validation mirror; reserved from this SFT selection, not official blind test'},[args.exclude_manifest])
    raw=out/'raw';raw.mkdir();images=out/'images';images.mkdir()
    endpoint='https://huggingface.co/api/datasets/'+DATASET
    fetch(endpoint,raw/'repository-before.json');candidates={};reused={}
    validation_endpoint='https://huggingface.co/api/datasets/'+VALIDATION_DATASET
    fetch(validation_endpoint,raw/'validation-repository-before.json')
    revision=json.loads((raw/'repository-before.json').read_text())['sha']
    if args.reuse_raw and json.loads((Path(args.reuse_raw)/'repository-before.json').read_text())['sha']!=revision:
        raise ValueError('Cannot reuse responses from a different observed repository revision')
    for split,offsets in OFFSETS.items():
        candidates[split]=[]
        for offset in offsets:
            path=raw/(split+'-'+str(offset)+'.json')
            query=urlencode({'dataset':DATASET if split=='train' else VALIDATION_DATASET,
                'config':'default' if split=='train' else 'DocVQA','split':split,'offset':offset,'length':100})
            previous=Path(args.reuse_raw)/path.name if args.reuse_raw else None
            usable=False
            if split=='train' and previous and previous.is_file():
                try:
                    cached=json.loads(previous.read_text())
                    usable=len(cached['rows'])==100 and [r['row_idx'] for r in cached['rows']]==list(range(offset,offset+100))
                except (ValueError,KeyError):pass
            if usable:shutil.copy2(previous,path);reused[path.name]=sha(previous)
            else:fetch('https://datasets-server.huggingface.co/rows?'+query,path)
            payload=json.loads(path.read_text())
            if split=='validation':
                for item in payload['rows']:
                    row=item['row'];item['row']={**row,'question_id':row['questionId'],
                        'other_metadata':{'ucsf_document_id':row['ucsf_document_id'],'doc_id':row['docId']}}
            candidates[split].extend(payload['rows'])
            print('CANDIDATE_PAGE',split,offset,'reused' if usable else 'downloaded',flush=True)
    train=select_documents(candidates['train'],32,excluded,2)
    excluded|={str(i['row']['other_metadata']['ucsf_document_id']) for i in train}
    evaluation_pool=select_documents(candidates['validation'],32,excluded,1)
    selected=[('train',item) for item in train]+[('val' if n<12 else 'test',item) for n,item in enumerate(evaluation_pool)]
    def download(pair):
        role,item=pair;row=item['row'];qid=str(row['question_id']);meta=row['other_metadata']
        expected='train' if role=='train' else 'val'
        if row['data_split']!=expected:raise ValueError('Unexpected source split')
        path=images/(qid+'.png');fetch(row['image']['src'],path)
        with Image.open(path) as image:image.load();size=list(image.size)
        if size!=[row['image']['width'],row['image']['height']]:raise ValueError('Wrong image size')
        return {'question_id':qid,'document_id':str(meta['ucsf_document_id']),'page_id':str(meta['doc_id']),
            'image':'images/'+path.name,'question':row['question'],'answers':row['answers'],'split':role,
            'source_split':'train' if role=='train' else 'validation','source_row':item['row_idx'],
            'question_types':row.get('question_types',[]),
            'image_sha256':sha(path),'image_size':size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(download,selected))
    fetch(endpoint,raw/'repository-after.json')
    fetch(validation_endpoint,raw/'validation-repository-after.json')
    before=json.loads((raw/'repository-before.json').read_text())['sha']
    if before!=json.loads((raw/'repository-after.json').read_text())['sha']:raise ValueError('Repository changed')
    validation_revision=json.loads((raw/'validation-repository-before.json').read_text())['sha']
    if validation_revision!=json.loads((raw/'validation-repository-after.json').read_text())['sha']:raise ValueError('Validation repository changed')
    groups={role:[r for r in rows if r['split']==role] for role in ['train','val','test']}
    hashes={};ids=set();old_hashes={sha(r['image_path']) for r in old}
    for row in rows:
        if row['question_id'] in ids:raise ValueError('Duplicate question')
        ids.add(row['question_id'])
        digest=row['image_sha256']
        if digest in old_hashes:raise ValueError('Image overlaps old development set')
        owner=hashes.setdefault(digest,row['split'])
        if owner!=row['split']:raise ValueError('Image hash leaks between roles')
    for role,group in groups.items():
        path=out/(role+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in group))
    check_document_splits(*(load_manifest(out/(role+'.jsonl')) for role in groups))
    report={'dataset':DATASET,'repository_revision_observed':before,'viewer_revision_pinned':False,
        'validation_dataset':VALIDATION_DATASET,'validation_repository_revision_observed':validation_revision,
        'offsets':OFFSETS,'seed':42,'reused_raw_sha256':reused,'exclude_manifest_sha256':sha(args.exclude_manifest),
        'raw_response_sha256':{p.name:sha(p) for p in raw.glob('*.json')},
        'splits':{role:{'questions':len(group),'source_documents':len({r['document_id'] for r in group}),
            'manifest_sha256':sha(out/(role+'.jsonl'))} for role,group in groups.items()},
        'items':[{k:r[k] for k in ['question_id','document_id','page_id','split','source_split','source_row','image_sha256','image_size']} for r in rows],
        'evaluation_scope':'small reserved evaluation from official validation; once inspected, no longer pristine for future tuning'}
    (out/'dataset-summary.json').write_text(json.dumps(report,indent=2));print('SFT_DATA_PREPARED',report['splits'],flush=True)
if __name__=='__main__':main()
