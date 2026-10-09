"""Fetch a fixed, auditable development slice from the public DocVQA mirror.

The viewer is not revision-pinned: record repository revisions before/after,
raw response hashes and image hashes. This is a development slice, not an
official benchmark score. Signed asset URLs remain in ignored raw data only.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode, urlparse
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_record import snapshot_run, sha
from data import load_manifest, check_document_splits

DATASET = 'lmms-lab-encoder/DocVQA'

def fetch(url, destination):
    if urlparse(url).scheme != 'https':
        raise ValueError('HTTPS source required')
    result = subprocess.run(['curl', '-fsSL', '--retry', '3', '--connect-timeout', '20',
                    '--max-time', '180', url, '-o', str(destination)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        # Do not include signed asset URLs in console errors.
        raise RuntimeError('Download failed for ' + destination.name + ': curl exit ' + str(result.returncode))

def select_rows(payload, count):
    selected, documents = [], set()
    for item in payload['rows']:
        if item.get('truncated_cells'):
            raise ValueError('Truncated annotation')
        row = item['row']
        # One question per source document, not merely one per page.
        document = str(row['ucsf_document_id'])
        if document in documents:
            continue
        if row['data_split'] != 'val' or not row['answers']:
            raise ValueError('Expected labeled validation rows')
        selected.append(item)
        documents.add(document)
        if len(selected) == count:
            return selected
    raise ValueError('Not enough distinct documents; increase candidate rows')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--candidate-rows', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.count <= args.candidate_rows <= 100:
        raise ValueError('Use 1 <= count <= candidate rows <= 100')
    out = snapshot_run(args.output, {**vars(args), 'dataset': DATASET,
                       'selection': 'first question for each distinct source document in first viewer page',
                       'purpose': 'development only; not representative of the complete validation split'})
    raw = out / 'raw'; raw.mkdir()
    images = out / 'images'; images.mkdir()
    endpoint = 'https://huggingface.co/api/datasets/' + DATASET
    fetch(endpoint, raw / 'repository-before.json')
    query = urlencode({'dataset': DATASET, 'config': 'DocVQA', 'split': 'validation',
                       'offset': 0, 'length': args.candidate_rows})
    fetch('https://datasets-server.huggingface.co/rows?' + query, raw / 'rows.json')
    payload = json.loads((raw / 'rows.json').read_text())
    selected = select_rows(payload, args.count)
    def download(item):
        row = item['row']; question = str(row['questionId'])
        path = images / (question + '.png')
        fetch(row['image']['src'], path)
        with Image.open(path) as image:
            image.load(); size = list(image.size)
            if size != [row['image']['width'], row['image']['height']]:
                raise ValueError('Image size differs from viewer metadata')
        return {'question_id': question, 'document_id': str(row['ucsf_document_id']),
                'page_id': str(row['docId']), 'image': 'images/' + path.name,
                'question': row['question'], 'answers': row['answers'], 'split': 'val',
                'question_types': row['question_types'], 'source_row': item['row_idx'],
                'image_sha256': sha(path), 'image_size': size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(download, selected))
    fetch(endpoint, raw / 'repository-after.json')
    before = json.loads((raw / 'repository-before.json').read_text())['sha']
    after = json.loads((raw / 'repository-after.json').read_text())['sha']
    if before != after:
        raise ValueError('Repository changed during download; preserve this failed run')
    manifest = out / 'dev.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    check_document_splits(load_manifest(manifest))
    report = {'dataset': DATASET, 'repository_revision_observed': before,
              'viewer_revision_pinned': False, 'raw_response_sha256': sha(raw / 'rows.json'),
              'manifest_sha256': sha(manifest), 'selection': 'first question per distinct source document',
              'candidate_rows': args.candidate_rows, 'questions': len(rows),
              'source_documents': len({r['document_id'] for r in rows}),
              'usage': 'development only; no official benchmark or held-out test claim',
              'items': [{k: row[k] for k in ['question_id', 'document_id', 'page_id',
                         'source_row', 'image_sha256', 'image_size', 'question_types']} for row in rows]}
    (out / 'dataset-summary.json').write_text(json.dumps(report, indent=2))
    print('DOCVQA_PREPARED', len(rows), 'distinct documents', flush=True)

if __name__ == '__main__':
    main()
