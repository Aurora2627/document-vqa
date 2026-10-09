"""Strict paired evaluation for saved, document-disjoint development runs."""
from statistics import median
from metrics import score_answer

METRIC = 'ANLS-style: lowercase, strip/collapse whitespace, normalized edit distance < 0.5; best reference'

def index_predictions(rows):
    if not rows:
        raise ValueError('No predictions')
    indexed = {}
    for row in rows:
        key = str(row['question_id'])
        if key in indexed:
            raise ValueError('Duplicate prediction ID: ' + key)
        if not isinstance(row['prediction'], str):
            raise ValueError('Prediction must be text')
        indexed[key] = row
    return indexed

def summarize(rows):
    indexed = index_predictions(rows)
    scores = {key: score_answer(row['prediction'], row['answers']) for key, row in indexed.items()}
    categories = {}
    for key, row in indexed.items():
        for category in set(row.get('question_types', [])):
            categories.setdefault(category, []).append(scores[key]['anls'])
    seconds = [row['seconds'] for row in rows if 'seconds' in row]
    return {'count': len(rows), 'documents': len({r['document_id'] for r in rows}),
            'metric_definition': METRIC, 'official_evaluator_parity_verified': False,
            'anls': sum(s['anls'] for s in scores.values()) / len(rows),
            'exact_match': sum(s['exact_match'] for s in scores.values()) / len(rows),
            'generation_seconds_median': median(seconds) if seconds else None,
            'generation_seconds_total': sum(seconds) if seconds else None,
            'by_question_type': {k: {'count': len(v), 'anls': sum(v)/len(v)} for k,v in categories.items()},
            'per_question': [{'question_id': key, **score} for key,score in scores.items()]}

def compare(left, right):
    left, right = index_predictions(left), index_predictions(right)
    if left.keys() != right.keys():
        raise ValueError('Paired comparison requires exactly the same question IDs')
    changes = []
    for key, a in left.items():
        b = right[key]
        if a['document_id'] != b['document_id'] or a['answers'] != b['answers']:
            raise ValueError('Paired reference mismatch: ' + key)
        sa, sb = score_answer(a['prediction'], a['answers']), score_answer(b['prediction'], b['answers'])
        changes.append({'question_id': key, 'left_anls': sa['anls'], 'right_anls': sb['anls'],
                        'delta': sb['anls'] - sa['anls']})
    return {'count': len(changes), 'mean_anls_delta': sum(r['delta'] for r in changes)/len(changes),
            'improved': sum(r['delta'] > 0 for r in changes),
            'worsened': sum(r['delta'] < 0 for r in changes),
            'unchanged': sum(r['delta'] == 0 for r in changes), 'per_question': changes}
