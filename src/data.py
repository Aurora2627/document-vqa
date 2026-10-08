"""Validated single-page question manifests with document-disjoint splits."""
import json
from pathlib import Path

def load_manifest(path):
    path=Path(path);rows=[];seen=set()
    for line in path.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        for field in ['question_id','document_id','image','question','answers','split']:
            if field not in row:raise ValueError('Missing '+field)
        if row['question_id'] in seen:raise ValueError('Duplicate question ID')
        seen.add(row['question_id'])
        if row['split'] not in ['train','val','test']:raise ValueError('Unknown split')
        if not isinstance(row['answers'],list) or not row['answers'] or not all(isinstance(a,str) and a.strip() for a in row['answers']):raise ValueError('Nonempty answer strings required')
        image=(path.parent/row['image']).resolve()
        if not image.is_file():raise FileNotFoundError(image)
        row={**row,'image_path':str(image)};rows.append(row)
    if not rows:raise ValueError('Empty manifest')
    return rows

def check_document_splits(*groups):
    owners={}
    for group in groups:
        for row in group:
            old=owners.setdefault(row['document_id'],row['split'])
            if old!=row['split']:raise ValueError('Document leaks between splits: '+row['document_id'])
