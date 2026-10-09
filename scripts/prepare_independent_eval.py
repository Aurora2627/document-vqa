"""Freeze a document-disjoint evaluation slice before inference, without answer selection."""
import argparse, concurrent.futures, hashlib, json, sys
from pathlib import Path
from urllib.parse import urlencode
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_docvqa import fetch, DATASET
from data import load_manifest, check_document_splits
from run_record import snapshot_run, sha
OFFSETS = [3500, 3700, 3900, 4100, 4300, 4500, 4700, 4900]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--exclude', nargs='+', required=True)
    p.add_argument('--count', type=int, default=100)
    args = p.parse_args()
    previous = [row for path in args.exclude for row in load_manifest(path)]
    excluded_docs = {r['document_id'] for r in previous}
    excluded_ids = {r['question_id'] for r in previous}
    excluded_hashes = {sha(r['image_path']) for r in previous}
    out = snapshot_run(args.output, {**vars(args), 'dataset': DATASET, 'offsets': OFFSETS,
        'seed': 20261009, 'selection': 'SHA256(seed:document_id), then first source row per document',
        'protocol': 'whole page 512 versus 768; frozen base; greedy 64 tokens; no tuning on this slice',
        'scope': 'public labeled validation mirror; not official blind test; finite candidate pool'}, args.exclude)
    raw = out / 'raw'; raw.mkdir(); images = out / 'images'; images.mkdir()
    endpoint = 'https://huggingface.co/api/datasets/' + DATASET
    fetch(endpoint, raw / 'repository-before.json')
    groups = {}
    for offset in OFFSETS:
        query = urlencode({'dataset': DATASET, 'config': 'DocVQA', 'split': 'validation', 'offset': offset, 'length': 100})
        path = raw / ('rows-' + str(offset) + '.json')
        fetch('https://datasets-server.huggingface.co/rows?' + query, path)
        payload = json.loads(path.read_text())
        if [r['row_idx'] for r in payload['rows']] != list(range(offset, offset + 100)):
            raise ValueError('Incomplete candidate page')
        for item in payload['rows']:
            if set(item.get('truncated_cells', [])) - {'ocr_results'}:
                raise ValueError('Truncated required fields')
            row = item['row']; doc = str(row['ucsf_document_id']); qid = str(row['questionId'])
            if doc in excluded_docs or qid in excluded_ids: continue
            if row['data_split'] != 'val' or not row['answers']: raise ValueError('Invalid labeled validation row')
            groups.setdefault(doc, []).append(item)
        print('EVAL_CANDIDATES', offset, len(groups), flush=True)
    ordered = sorted(groups, key=lambda d: hashlib.sha256(('20261009:' + d).encode()).hexdigest())
    if len(ordered) < args.count: raise ValueError('Insufficient independent documents')
    selected = [min(groups[d], key=lambda r: r['row_idx']) for d in ordered[:args.count]]
    def download(item):
        row = item['row']; qid = str(row['questionId']); path = images / (qid + '.png')
        fetch(row['image']['src'], path)
        with Image.open(path) as image: image.load(); size = list(image.size)
        if size != [row['image']['width'], row['image']['height']]: raise ValueError('Image size mismatch')
        digest = sha(path)
        if digest in excluded_hashes: raise ValueError('Image overlaps previous experiments')
        return {'question_id': qid, 'document_id': str(row['ucsf_document_id']), 'page_id': str(row['docId']),
            'image': 'images/' + path.name, 'question': row['question'], 'answers': row['answers'],
            'split': 'test', 'source_split': 'validation', 'source_row': item['row_idx'],
            'question_types': row['question_types'], 'image_sha256': digest, 'image_size': size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(download, selected))
    if len({r['image_sha256'] for r in rows}) != len(rows): raise ValueError('Duplicate image across source documents')
    fetch(endpoint, raw / 'repository-after.json')
    before = json.loads((raw / 'repository-before.json').read_text())['sha']
    if before != json.loads((raw / 'repository-after.json').read_text())['sha']: raise ValueError('Repository changed during fetch')
    manifest = out / 'test.jsonl'; manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    check_document_splits(load_manifest(manifest))
    summary = {'dataset': DATASET, 'repository_revision_observed': before, 'viewer_revision_pinned': False,
        'count': len(rows), 'source_documents': len(groups), 'offsets': OFFSETS, 'manifest_sha256': sha(manifest),
        'excluded_documents': len(excluded_docs), 'excluded_image_hashes': len(excluded_hashes),
        'raw_response_sha256': {f.name: sha(f) for f in raw.glob('*.json')},
        'usage': 'new isolated evaluation of a preselected resolution candidate; inspected after this run',
        'items': [{k: r[k] for k in ['question_id','document_id','source_row','image_sha256']} for r in rows]}
    (out / 'dataset-summary.json').write_text(json.dumps(summary, indent=2))
    print('INDEPENDENT_EVAL_PREPARED', len(rows), flush=True)
if __name__ == '__main__': main()
