import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args()
b=json.loads((a.run_dir/'baseline_original.jsonl').read_text());o=json.loads((a.run_dir/'optimized_original.jsonl').read_text())
metrics=['duration','output_throughput','mean_ttft_ms','mean_tpot_ms','mean_e2e_latency_ms','p99_ttft_ms','p99_tpot_ms','accept_length']
summary={'baseline_completed':b['completed'],'optimized_completed':o['completed'],
 'input_lengths_equal':b['input_lens']==o['input_lens'],'output_lengths_equal':b['output_lens']==o['output_lens'],
 'generated_texts_equal':b['generated_texts']==o['generated_texts'],
 'equal_text_count':sum(x==y for x,y in zip(b['generated_texts'],o['generated_texts'])),
 'baseline_errors':b['errors'],'optimized_errors':o['errors'],
 'metrics':{k:{'baseline':b[k],'optimized':o[k],'relative_change_pct':100*(o[k]/b[k]-1)} for k in metrics}}
(a.run_dir/'comparison.json').write_text(json.dumps(summary,indent=2))
lines=['# 整网对照','', '单次 A/B，不代表统计显著性。两次均重启服务后沿用原脚本 warmup 和数据集。','',f"成功请求：{b['completed']} → {o['completed']}；生成文本完全相同：{summary['generated_texts_equal']}；相同条数：{summary['equal_text_count']}。",'', '| 指标 | 基线 | 优化 | 变化 |','|---|---:|---:|---:|']
for k,v in summary['metrics'].items():lines.append(f"| {k} | {v['baseline']:.4f} | {v['optimized']:.4f} | {v['relative_change_pct']:+.3f}% |")
(a.run_dir/'comparison.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
