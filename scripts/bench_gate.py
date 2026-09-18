import argparse,json,sys,time,statistics
from pathlib import Path
import torch,torch_npu
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels'))
from fused_gate import fused_gate
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',choices=['triton','view'],default='triton');p.add_argument('--shapes',type=int,nargs='+',default=[1,4,20,80,6925,32768]);a=p.parse_args()
torch.npu.set_device(0);torch.manual_seed(42)
if a.candidate=='view':
    def fused_gate(x, gate, inplace=True):
        target=x if inplace else x.clone()
        target.view_as(gate).mul_(torch.sigmoid(gate))
        return target

def measure(fn,graph=False):
    for _ in range(5):fn()
    torch.npu.synchronize()
    if graph:
        g=torch.npu.NPUGraph()
        with torch.npu.graph(g):
            for _ in range(20): fn()
        call=g.replay; divisor=20
    else:call=fn;divisor=1
    samples=[]
    for _ in range(7):
        start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(20):call()
        end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/(20*divisor))
    return statistics.median(samples)
results=[]
for m in a.shapes:
 for layout in ['contiguous','strided']:
    x=torch.randn((m,3072),device='npu',dtype=torch.bfloat16)
    storage=torch.randn((m,12,512),device='npu',dtype=torch.bfloat16)*4
    gate=storage[:,:,256:] if layout=='strided' else torch.randn_like(x)*4
    ref=x*torch.sigmoid(gate.reshape_as(x));actual=fused_gate(x,gate,inplace=False)
    torch.npu.synchronize()
    diff=(actual.float()-ref.float()).abs()
    torch.testing.assert_close(actual,ref,atol=0.015625,rtol=0.008)
    max_abs=diff.max().item(); mismatch=(actual!=ref).float().mean().item()
    # In-place replay with zero gates would decay the input; use a stable gate
    # for timing so repeated calls do not benchmark denormal/underflow inputs.
    gate.fill_(20)
    old=lambda:x.mul_(torch.sigmoid(gate.reshape_as(x)))
    new=lambda:fused_gate(x,gate)
    row={'m':m,'hidden':3072,'layout':layout,'max_abs_error':max_abs,'mismatch_fraction':mismatch}
    for mode in ['eager','graph']:
        old_us=measure(old,mode=='graph');new_us=measure(new,mode=='graph')
        row[mode]={'old_us':old_us,'new_us':new_us,'speedup':old_us/new_us}
    results.append(row);a.output.write_text(json.dumps(results,indent=2));print(row,flush=True)
