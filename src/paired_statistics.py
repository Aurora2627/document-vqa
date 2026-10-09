"""Paired document-level bootstrap; one question per independent source document."""
import torch
from evaluation import index_predictions
from metrics import score_answer

def paired_bootstrap(left, right, samples=10000, seed=42):
    left, right = index_predictions(left), index_predictions(right)
    if left.keys() != right.keys(): raise ValueError('Mismatched paired IDs')
    if len({r['document_id'] for r in left.values()}) != len(left):
        raise ValueError('Bootstrap requires one question per source document')
    differences = []
    for key, a in left.items():
        b = right[key]
        if a['document_id'] != b['document_id'] or a['answers'] != b['answers']:
            raise ValueError('Mismatched paired references')
        sa, sb = score_answer(a['prediction'], a['answers']), score_answer(b['prediction'], b['answers'])
        differences.append([sb['anls']-sa['anls'], sb['exact_match']-sa['exact_match']])
    delta = torch.tensor(differences, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(delta), (samples, len(delta)), generator=generator)
    draws = delta[indices].mean(dim=1)
    return {'method': 'paired percentile bootstrap over independent documents', 'samples': samples, 'seed': seed,
        'metrics': {name: {'delta': delta[:, i].mean().item(),
            'ci95': torch.quantile(draws[:, i], torch.tensor([0.025,0.975], dtype=torch.float64)).tolist()}
            for i, name in enumerate(['anls','exact_match'])},
        'scope': 'sampling uncertainty within selected candidate pool; not population representativeness'}
