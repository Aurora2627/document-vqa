"""Real PyTorch EasyOCR text/coordinates, independent of questions and reference answers."""
import argparse,hashlib,json,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'data/cache/ocr-python'));sys.path.insert(0,str(ROOT/'src'))
import torch
from device_runtime import require_device,synchronize
import easyocr
from easyocr.config import detection_models,recognition_models
from ocr_weights import ensure_weight
from data import load_manifest
from run_record import snapshot_run,sha
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=p.parse_args();require_device(args.device)
    rows=load_manifest(args.manifest);torch.set_num_threads(2)
    out=snapshot_run(args.output,{**vars(args),'ocr':'EasyOCR 1.7.2','framework':'PyTorch','device':args.device,'languages':['en'],
        'canvas_size':1600,'batch_size':1,'workers':0,'paragraph':False,'cpu_threads':2,'scope':'no question/answer used during OCR extraction'},[args.manifest])
    weights=ROOT/'data/cache/easyocr-models';weights.mkdir(parents=True,exist_ok=True);sources=[]
    for config in [detection_models['craft'],recognition_models['gen2']['english_g2']]:
        sources.append(ensure_weight(config,weights,out))
    (out/'models.json').write_text(json.dumps(sources,indent=2))
    reader=easyocr.Reader(['en'],gpu=args.device,model_storage_directory=str(weights),user_network_directory=str(weights/'user-network'),download_enabled=False,verbose=False,quantize=False)
    records=out/'records';records.mkdir()
    with (out/'ocr-index.jsonl').open('x') as index:
        for n,row in enumerate(rows):
            synchronize(args.device);started=time.perf_counter();items=reader.readtext(row['image_path'],detail=1,paragraph=False,canvas_size=1600,batch_size=1,workers=0)
            synchronize(args.device)
            record={'question_id':row['question_id'],'image_sha256':sha(row['image_path']),'seconds':time.perf_counter()-started,
                'items':[{'box':[[float(x),float(y)] for x,y in box],'text':text,'confidence':float(confidence)} for box,text,confidence in items]}
            path=records/(row['question_id']+'.json');path.write_text(json.dumps(record,indent=2))
            index.write(json.dumps({'question_id':row['question_id'],'record':str(path.relative_to(out)),'sha256':sha(path),'items':len(items),'seconds':record['seconds']})+'\n');index.flush()
            print('OCR_PROGRESS',n+1,'/',len(rows),'lines',len(items),flush=True)
    print('OCR_PREPARED',flush=True)
