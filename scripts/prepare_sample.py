"""Synthetic document fixtures, never used as real DocVQA benchmark results."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    out=ROOT/'data/sample';out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,(split,total) in enumerate([('train','128.50'),('val','73.20'),('test','246.00')]):
        image=Image.new('RGB',(700,900),'white');draw=ImageDraw.Draw(image)
        draw.text((60,60),'SYNTHETIC INVOICE - SOFTWARE FIXTURE',fill='black',font_size=24)
        draw.text((60,150),'Invoice total: '+total,fill='black',font_size=28)
        name='synthetic-'+split+'.png';image.save(out/name)
        rows.append({'question_id':'fixture-q'+str(i),'document_id':'fixture-doc'+str(i),'image':name,'question':'What is the invoice total?','answers':[total],'split':split,'source':'synthetic software fixture, not real benchmark','evidence_bbox':[50,140,500,200]})
    (out/'fixtures.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    for split in ['train','val','test']:
        (out/(split+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows if r['split']==split)+'\n')
    print('SYNTHETIC_FIXTURES_SAVED')
