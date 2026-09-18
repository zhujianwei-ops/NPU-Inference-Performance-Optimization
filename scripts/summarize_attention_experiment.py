"""Summarize repeated serving metrics, output drift, and per-rank FIA calls."""
import argparse,json,csv,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args();r=a.run_dir
labels=['baseline_gate_on','attention_on','baseline_repeat','attention_repeat']
raw={k:json.loads((r/(k+'.jsonl')).read_text()) for k in labels if (r/(k+'.jsonl')).exists()}
keys=['completed','duration','output_throughput','mean_tpot_ms','median_tpot_ms','p99_tpot_ms','mean_ttft_ms','mean_e2e_latency_ms','accept_length']
result={'runs':{k:{m:d[m] for m in keys} for k,d in raw.items()},'text_comparisons':{},'profile':[]}
for x,y in [('baseline_gate_on','attention_on'),('baseline_gate_on','baseline_repeat'),('attention_on','attention_repeat'),('baseline_repeat','attention_repeat')]:
 if x in raw and y in raw:
  aa,bb=raw[x],raw[y];result['text_comparisons'][x+' vs '+y]={'same_texts':sum(i==j for i,j in zip(aa['generated_texts'],bb['generated_texts'])),'n':len(aa['generated_texts']),'input_lengths_equal':aa['input_lens']==bb['input_lens'],'output_lengths_equal':aa['output_lens']==bb['output_lens']}
if len(raw)==4:
 means={}
 for name,ks in [('baseline',['baseline_gate_on','baseline_repeat']),('optimized',['attention_on','attention_repeat'])]:
  means[name]={m:statistics.mean(raw[k][m] for k in ks) for m in keys}
 result['mean_metrics']=means
 result['mean_relative_change_pct']={m:100*(means['optimized'][m]/means['baseline'][m]-1) for m in keys if means['baseline'][m]}
for label in ['baseline_repeat','attention_on']:
 for f in sorted((r/(label+'_profiles')).glob('profiles/**/kernel_details.csv')):
  rows=list(csv.DictReader(f.open()))
  if sum(x['Type']=='MatMulV2' for x in rows)!=1938:continue
  groups=defaultdict(lambda:{'calls':0,'duration_us':0.})
  for x in rows:
   if x['Type']=='FusedInferAttentionScore':
    g=groups[x['Input Shapes']];g['calls']+=1;g['duration_us']+=float(x['Duration(us)'])
  result['profile'].append({'label':label,'device':rows[0]['Device_id'],'file':str(f.relative_to(r)),'groups':dict(groups)})
probe_files=[r/'attention_on_probes.json',r/'baseline_repeat_probes.json']
if all(p.exists() for p in probe_files):
 aa,bb=[json.loads(p.read_text()) for p in probe_files]
 result['probes']=[{'name':x['name'],'baseline_pass':y['contains_expected'],'optimized_pass':x['contains_expected'],'text_equal':x['response']['text']==y['response']['text'],'baseline_prompt_tokens':y['response']['meta_info'].get('prompt_tokens'),'optimized_prompt_tokens':x['response']['meta_info'].get('prompt_tokens')} for x,y in zip(aa,bb)]
(r/'attention_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# 第二轮整网重复对照','', '| 实验 | 成功 | 输出 token/s | TPOT ms | TTFT ms | E2E ms | accept length |','|---|---:|---:|---:|---:|---:|---:|']
for name,d in result['runs'].items():lines.append(f"| {name} | {d['completed']} | {d['output_throughput']:.3f} | {d['mean_tpot_ms']:.3f} | {d['mean_ttft_ms']:.3f} | {d['mean_e2e_latency_ms']:.3f} | {d['accept_length']:.4f} |")
lines+=['','## 文本一致性','']
for label,d in result['text_comparisons'].items():lines.append(f"- {label}: {d['same_texts']}/{d['n']} 完全一致；输入长度相同 {d['input_lengths_equal']}，输出长度相同 {d['output_lengths_equal']}。")
lines+=['','两次重复只能作为复现证据，不等于充分的统计显著性检验。语义 smoke checks 不替代正式质量评测。','']
(r/'attention_comparison.md').write_text('\n'.join(lines));print(json.dumps({k:v for k,v in result.items() if k!='profile'},ensure_ascii=False,indent=2))
