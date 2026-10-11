"""Export aggregate results; verify saved artifacts without publishing document text."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run,sha

def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--ocr-study',required=True)
    p.add_argument('--audit',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    study=Path(args.study);ocr=Path(args.ocr_study)
    paths={'training':study/'training-study-summary.json','sft':study/'sft/experiment-summary.json','base_grid':ROOT/'runs/region-reading-dev-v070/experiment-summary.json',
        'trained_grid':study/'region-dev/experiment-summary.json','ocr':ocr/'experiment-summary.json','audit':Path(args.audit)/'audit.json',
        'prior_config':study/'prior-adapter-test/config.json','expanded_config':study/'sft/adapter-test/config.json'}
    out=snapshot_run(args.output,vars(args),list(paths.values())+[ocr/'retrieved-regions.jsonl'])
    reports={name:read(path) for name,path in paths.items()}
    for name,path in {'base':study/'sft/baseline-test/predictions.jsonl','prior':study/'prior-adapter-test/predictions.jsonl','expanded':study/'sft/adapter-test/predictions.jsonl'}.items():
        if sha(path)!=reports['training']['prediction_sha256'][name]:raise ValueError('Training prediction hash mismatch')
    adapter=reports['training']['adapter_sha256']
    if any(reports[name]['protocol']['adapter_sha256']!=adapter for name in ['trained_grid','ocr']):raise ValueError('Different trained readers')
    if reports['audit']['adapter_sha256']!=adapter:raise ValueError('Adapter audit mismatch')
    audits={}
    for name,directory in [('base_grid',paths['base_grid'].parent),('trained_grid',paths['trained_grid'].parent)]:
        modes=['fixed-fusion','shuffled-fusion','selected-fusion'];groups={mode:rows(directory/(mode+'.jsonl')) for mode in modes}
        for mode in modes:
            if sha(directory/(mode+'.jsonl'))!=reports[name]['prediction_sha256'][mode]:raise ValueError('Prediction hash mismatch')
        if not all(len(group)==20 for group in groups.values()):raise ValueError('Incomplete development predictions')
        for triplet in zip(*(groups[mode] for mode in modes)):
            if len({r['question_id'] for r in triplet})!=1 or len({r['input_tokens'] for r in triplet})!=1:raise ValueError('Fusion pairing/budget mismatch')
        audits[name]={'fusion_pairs':20,'matched_reader_token_budgets':True}
    for mode,digest in reports['ocr']['prediction_sha256'].items():
        if sha(ocr/(mode+'.jsonl'))!=digest or len(rows(ocr/(mode+'.jsonl')))!=20:raise ValueError('OCR prediction mismatch')
    retrieval=rows(ocr/'retrieved-regions.jsonl')
    if len(retrieval)!=20:raise ValueError('Incomplete retrieval records')
    retrieval_audit={'count':len(retrieval),'guided_nonempty':sum(bool(r['guided']['regions']) for r in retrieval),
        'shuffled_nonempty':sum(bool(r['shuffled']['regions']) for r in retrieval),
        'guided_context_truncated':sum(r['guided_budget']['truncated'] for r in retrieval),
        'records_sha256':sha(ocr/'retrieved-regions.jsonl'),
        'interpretation':'Shuffled query often yields no OCR context; not a matched-length wrong-evidence control or grounding recall'}
    audit=reports['audit'].copy()
    if reports['prior_config']['adapter_sha256']==adapter or reports['expanded_config']['adapter_sha256']!=adapter:raise ValueError('Prior/expanded checkpoint identity mismatch')
    prior={r['question_id']:r for r in rows(study/'prior-adapter-test/predictions.jsonl')}
    expanded={r['question_id']:r for r in rows(study/'sft/adapter-test/predictions.jsonl')}
    if prior.keys()!=expanded.keys():raise ValueError('Prior/expanded prediction pairing mismatch')
    audit['prior_adapter_sha256']=reports['prior_config']['adapter_sha256']
    audit['prior_to_expanded_raw_changes']=sum(prior[q]['prediction']!=expanded[q]['prediction'] for q in prior)
    audit['prediction_changes']={role:{k:v for k,v in item.items() if k!='changed_question_ids'} for role,item in audit['prediction_changes'].items()}
    with (out/'adapter-audit.json').open('x') as f:json.dump(audit,f,indent=2)
    content=['# LLaVA 参数高效微调与问题驱动区域阅读（v0.7.0）','',
        '本轮在 MacBook MPS、系统 Python/PyTorch、VS Code 集成终端运行。新增训练规模和问题驱动区域阅读消融；所有原始预测、区域、OCR、源码快照与权重保留在本机。','',
        '## 新训练与保留评测','',
        '122 道训练题来自 64 个源文档，包含上一版的 59 题；新验证 24 文档，新保留评测 64 文档。按源文档、问题 ID、图片哈希排除已知开发/评测文档。验证 NLL 选择训练轮次，保留集不用于选模型。',
        '','| 配置（均为整页 512） | 完全匹配 | ANLS 风格 |','|---|---:|---:|']
    def line(label,result,cost=False):
        count=result['count'];value=f"| {label} | {round(result['exact_match']*count)}/{count} | {result['anls']:.4f}"
        return value+(f" | {result['pipeline_wall_seconds_median']:.2f} |" if cost else ' |')
    for mode,label in [('base','原模型'),('prior','此前 59 题 LoRA'),('expanded','本轮 122 题 LoRA')]:content.append(line(label,reports['training']['results'][mode]))
    train=reports['training']['training'];validation=reports['sft']['results']['val']
    content+=['',f"两轮共 {train['optimizer_steps']} 次更新，选择第 {train['best_epoch']} 轮；验证答案 token NLL 从 {train['initial_validation_nll']:.4f} 到 {train['best_validation_nll']:.4f}。验证集 EM 从 {round(validation['baseline']['exact_match']*24)}/24 到 {round(validation['adapter']['exact_match']*24)}/24。验证改善不保证保留集生成改善。",'',
        f"参数核验：{audit['nonzero_b_matrices']}/{audit['b_matrices']} 个 B 矩阵非零，保留集 {audit['prediction_changes']['test']['raw_predictions_changed']}/64 个原始回答变化。模型确实发生更新，效果仍以配对评测为准。"]
    content+=['',f"旧、新 adapter 哈希不同；两者之间 {audit['prior_to_expanded_raw_changes']}/64 个原始回答不同。评分是否变化见逐题配对 JSON，不把相同分数误写成相同权重。"]
    content+=['','训练统计与配对区间见 [训练 JSON](expanded-sft-v0.7.0.json)。旧、新 adapter 的验证集和 checkpoint 选择不同，因此两者比较不能解释为纯数据量因果实验。64 文档来自有标签的 validation 镜像，候选池有限，不是官方盲测。','',
        '## 区域选择消融：已查看的 20 文档开发集','',
        '定位器先看整页 512 和问题，在九个区域数字的 next-token logits 中取前两项；裁原图后以 768 读局部。九格概率只是受限数字分布，未做区域监督训练，也不是校准后的证据置信度。融合使用整页和两个局部的原生三图接口。']
    for name,title in [('base_grid','冻结原模型'),('trained_grid','本轮 LoRA')]:
        content+=['','### '+title,'','| 方法 | 完全匹配 | ANLS 风格 | 单请求流程中位秒 |','|---|---:|---:|---:|']
        for mode in ['whole','selected-crop','fixed-fusion','shuffled-fusion','selected-fusion']:content.append(line(mode,reports[name]['results'][mode],True))
        content+=['',f"更换为下一道问题后，top-1 区域相同的文档数：{reports[name]['same_top1_under_shuffled_question']}/20。"]
    content+=['','固定/错配/问题选择三种融合的 reader token 数已逐题核验相同；整页单图 anyres 与三图处理路径不同，不能把整页对三图的差值完全归因于区域。定位开销计入流程耗时。','',
        '## OCR 辅助消融：同一 LoRA、同一开发集','',
        'EasyOCR 的 PyTorch CRAFT/英文识别器在 CPU 提取文字与四边形；区域选择使用 PyTorch BM25，匹配问题词与相邻文字行，返回完整宽度的横带。每带最多八行、128 个文字 token，最多两带。无匹配退回整页。不读取答案选区域。','',
        '| 方法 | 完全匹配 | ANLS 风格 | 单请求流程中位秒 |','|---|---:|---:|---:|']
    for mode,result in reports['ocr']['results'].items():content.append(line(mode,result,True))
    content+=['','question-only 使用同一语言模型且无图片，ocr-only 只加入检索文字；whole-ocr 同时输入整页与文字，shuffled-ocr 用另一问题检索文字再回答原问题。各组文字长度不同，不能宣称 token 完全匹配。',
        '',f"检索审计：原问题在 {retrieval_audit['guided_nonempty']}/20 页返回非空候选，错配问题仅 {retrieval_audit['shuffled_nonempty']}/20 页。错配控制多数退回无文字输入，不能把差值解释成同等文字预算下的问题相关性收益。没有证据框真值，非空比例不是证据召回率。本轮原问题上下文发生 token 上限裁剪的页数为 {retrieval_audit['guided_context_truncated']}/20。",
        '','OCR 结果在实验中缓存，表中流程时间仍加回该页 OCR 实测耗时；CPU OCR 当时与 MPS 训练并行，时间受到资源争用影响，仅供成本透明，不作速度基准。全部时间排除权重加载和裁剪文件保存。',
        '','## 结论边界与复现','',
        '区域/OCR 只在已查看的 20 文档上开发，不代表独立集增益；没有真实证据框，因此不报告 IoU 或已验证 grounding。新保留集评测后也成为已知数据，下一轮调参需新留出文档。ANLS 为本项目风格实现，尚未核验官方 evaluator 等价。配对 bootstrap 仅作探索分析，未校正多重比较。',
        '','完整配置、逐题配对及区间见本目录四份 v0.7.0 JSON。运行方法见 [VS Code 说明](../VSCODE_START.md)。OCR 上游：[EasyOCR](https://github.com/JaidedAI/EasyOCR)。']
    with Path(args.report).open('x') as f:f.write('\n'.join(content)+'\n')
    (out/'export-audit.json').write_text(json.dumps({'source_summary_sha256':{name:sha(path) for name,path in paths.items()},'fusion':audits,'adapter':audit,'retrieval':retrieval_audit},indent=2))
    with (ROOT/'reports/expanded-adapter-audit-v0.7.0.json').open('x') as f:json.dump(audit,f,indent=2)
    with (ROOT/'reports/ocr-retrieval-audit-v0.7.0.json').open('x') as f:json.dump(retrieval_audit,f,indent=2)
    print('DEEPENING_REPORT_EXPORTED',flush=True)

if __name__=='__main__':main()
