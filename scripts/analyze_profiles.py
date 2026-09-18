"""Aggregate each rank/trace separately; summed kernel time is not wall time."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args()
result=[]
for f in sorted((a.run_dir/'profiles').glob('**/ASCEND_PROFILER_OUTPUT/kernel_details.csv')):
    rows=list(csv.DictReader(f.open())); groups=defaultdict(lambda: {'count':0,'duration_us':0.0})
    intervals=[]
    for r in rows:
        duration=float(r['Duration(us)']); key=(r['Type'],r['Input Shapes'],r['Input Data Types'])
        groups[key]['count']+=1;groups[key]['duration_us']+=duration
        start=float(r['Start Time(us)']); intervals.append((start,start+duration))
    if not rows: continue
    total=sum(x['duration_us'] for x in groups.values()); merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],hi)
        else:merged.append([lo,hi])
    rank={'file':str(f.relative_to(a.run_dir)), 'device':rows[0]['Device_id'],
          'stage_inferred':'decode' if any(r['Type']=='MatMulV2' for r in rows) and not any(r['Type']=='MatMulV3' for r in rows) else 'prefill',
          'kernel_sum_us':total,'interval_union_us':sum(hi-lo for lo,hi in merged),
          'span_us':max(y for x,y in intervals)-min(x for x,y in intervals),
          'groups':[dict(type=k[0],shapes=k[1],dtypes=k[2],**v,mean_us=v['duration_us']/v['count'],sum_share_pct=100*v['duration_us']/total) for k,v in sorted(groups.items(),key=lambda kv:kv[1]['duration_us'],reverse=True)]}
    result.append(rank)
(a.run_dir/'profile_summary.json').write_text(json.dumps(result,indent=2))
lines=['# Profiling 汇总','', '每个 trace、设备分别汇总。sum_share 仅为重叠 kernel 时间之和的份额，不是关键路径比例或 Amdahl 加速比。阶段由 MatMul 类型推断，需结合 server.log 复核。','']
for r in result:
    lines += [f"## {r['file']}",'',f"Device {r['device']}, 推断阶段 {r['stage_inferred']}; kernel sum {r['kernel_sum_us']/1000:.3f} ms; interval union {r['interval_union_us']/1000:.3f} ms; span {r['span_us']/1000:.3f} ms.",'','| 类型 | shape | 次数 | 总 ms | 均值 us | sum_share % |','|---|---|---:|---:|---:|---:|']
    for g in r['groups'][:20]:lines.append(f"| {g['type']} | {g['shapes']} | {g['count']} | {g['duration_us']/1000:.3f} | {g['mean_us']:.3f} | {g['sum_share_pct']:.2f} |")
    lines.append('')
(a.run_dir/'profile_summary.md').write_text('\n'.join(lines))
print('summarized',len(result),'traces')
