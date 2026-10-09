"""Build a readable report and local evidence review page from saved predictions."""
import argparse,html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import snapshot_run

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--failures',required=True)
    p.add_argument('--dataset-summary',required=True)
    p.add_argument('--output',required=True);p.add_argument('--audit-output',required=True);args=p.parse_args()
    audit=snapshot_run(args.audit_output,vars(args),[args.summary,args.failures,args.dataset_summary])
    dataset=json.loads(Path(args.dataset_summary).read_text())
    report=json.loads(Path(args.summary).read_text());results=report['results']
    lines=['# 独立文档评测：分辨率与 LoRA 四组对照','',
        '在方法和已有 adapter 固定后，新增 100 道问题，每道来自不同源文档；与之前全部真实训练、验证、保留评测和开发数据按文档 ID、问题 ID、图片哈希隔离。',
        '', '数据来自公开 DocVQA validation 镜像的固定 800 行候选池，以 SHA256(seed:文档 ID) 排序选择 100 个文档，再选各文档候选池内最早的一道问题。不是完整官方榜单或官方盲测。', '',
        '| 配置 | 完全匹配 | ANLS 风格得分 | 生成耗时中位数 |','|---|---:|---:|---:|']
    for name,result in results.items():
        lines.append('| {} | {:.0f}/{} ({:.1%}) | {:.4f} | {:.2f} 秒 |'.format(name,result['exact_match']*result['count'],result['count'],result['exact_match'],result['anls'],result['generation_seconds_median']))
    lines+=['','## 配对分析','','差值均为右侧配置减左侧配置。95% 区间使用 10,000 次配对文档 bootstrap，seed=42；每文档仅一道题。主对照是 base-512→base-768，其余对照为探索性分析，未校正多重比较。','',
        '| 对照 | ANLS 差值 [95% 区间] | EM 差值 [95% 区间] | ANLS 改善/变差/不变 |','|---|---:|---:|---:|']
    for name,result in report['comparisons'].items():
        metrics=result['uncertainty']['metrics'];a=metrics['anls'];e=metrics['exact_match'];pair=result['paired']
        lines.append('| {} | {:+.4f} [{:+.4f}, {:+.4f}] | {:+.1%} [{:+.1%}, {:+.1%}] | {}/{}/{} |'.format(name,a['delta'],*a['ci95'],e['delta'],*e['ci95'],pair['improved'],pair['worsened'],pair['unchanged']))
    lines+=['','## 题型分析','','题型可重叠，题数不能相加作为总数。小类只作诊断，不据此宣称稳定收益。','', '| 题型 | 题数 | base-512 | base-768 | adapter-512 | adapter-768 |','|---|---:|---:|---:|---:|---:|']
    categories=sorted({k for r in results.values() for k in r['by_question_type']})
    for category in categories:
        counts=[r['by_question_type'].get(category,{}).get('count',0) for r in results.values()]
        values=['{:.4f}'.format(r['by_question_type'][category]['anls']) if category in r['by_question_type'] else '—' for r in results.values()]
        lines.append('| {} | {} | {} |'.format(category,max(counts),' | '.join(values)))
    lines+=['','## 边界与复现','','- 官方 evaluator 一致性尚未验证，统一称为 ANLS 风格指标。',
        '- 模型、processor、revision、问题、解码、设备均固定；已有 LoRA 用旧验证 NLL 选择，不使用本轮结果选 checkpoint。',
        '- 各配置均在 VS Code 使用系统 Python/PyTorch/MPS 顺序运行；耗时不包含模型加载、输入处理，也不是严格的速度榜单。',
        '- 置信区间只刻画所选样本内部的不确定性；有限候选池不代表完整验证分布。',
        '- 本轮查看后，新评测数据也成为已知数据；未来调参不能继续将它称为未查看测试集。',
        '- 每轮保存源码、配置、输入哈希、日志和完整预测。公开报告不包含图片或答案标注。','',
        '数据下载前后观察到相同仓库 revision：`'+dataset['repository_revision_observed']+'`，但 viewer 接口本身没有绑定该 revision。',
        '排除已有 '+str(dataset['excluded_documents'])+' 个源文档；候选池中剩余 '+str(dataset['source_documents'])+' 个源文档，选取 100 个。',
        '本轮 manifest SHA256：`'+dataset['manifest_sha256']+'`。','']
    with Path(args.output).open('x') as output:output.write('\n'.join(lines))
    failures=json.loads(Path(args.failures).read_text());cards=[]
    for row in failures:
        details=''.join('<p><b>{}</b>: {} (ANLS {:.3f})</p>'.format(html.escape(name),html.escape(pred),row['scores'][name]['anls']) for name,pred in row['predictions'].items())
        cards.append('<article><h2>{}</h2><p>{}</p><p>参考答案：{}</p>{}<img loading="lazy" src="{}" alt="文档原图"><p>审核状态：待人工查看。不要仅根据预测字符串判断是 OCR、布局还是知识错误。</p></article>'.format(html.escape(row['question_id']),html.escape(row['question']),html.escape(' / '.join(row['references'])),details,html.escape(Path(row['image']).as_uri())))
    page='<!doctype html><meta charset="utf-8"><title>本地失败案例复核</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#f5f7fa}article{background:white;padding:24px;margin:24px 0;border-radius:12px}img{max-width:100%}p{overflow-wrap:anywhere}</style><h1>四组预测与原图复核</h1><p>仅供本机复核，包含数据标注，不上传仓库。</p>'+''.join(cards)
    (audit/'failure-review.html').write_text(page)
    print('INDEPENDENT_REPORT_RENDERED',len(failures),flush=True)
if __name__=='__main__':main()
