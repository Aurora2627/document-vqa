"""Export verified development summaries, excluding images/answers/signed URLs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    dataset = json.loads((ROOT/'runs/docvqa-dev-v030/dataset-summary.json').read_text())
    summaries = {}
    for edge in (384, 512):
        summary = json.loads((ROOT/f'runs/docvqa-baseline-{edge}-v030/answer-metrics.json').read_text())
        if summary['count'] != dataset['questions'] or summary['documents'] != dataset['source_documents']:
            raise ValueError('Unexpected counts')
        if {r['question_id'] for r in summary['per_question']} != {r['question_id'] for r in dataset['items']}:
            raise ValueError('Unexpected question IDs')
        summaries[str(edge)] = summary
    report = {'version': '0.3.0', 'date': '2026-10-09', 'launch': 'VS Code integrated terminal',
              'framework': 'PyTorch', 'device': 'MPS',
              'model': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
              'revision': '74dd0bf867a4cda7950c17663794267c60cf4b40',
              'base_dtype': 'float16', 'adapter': None,
              'decoding': 'greedy; max_new_tokens=64; full-page input',
              'dataset': dataset, 'results_by_max_image_edge': summaries,
              'limitations': ['20 development questions from 20 source documents; first viewer page, not random full-split sampling',
                              'Repository revision observed, but viewer assets not revision-pinned; raw responses and exact image hashes retained',
                              'ANLS-style implementation; official evaluator parity not verified',
                              'No fine-tuning gain or held-out benchmark claim',
                              'Generation time excludes image processing/model loading and is not an end-to-end latency benchmark']}
    with (ROOT/'reports/docvqa-development-v0.3.0.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print('DEVELOPMENT_REPORT_EXPORTED')
