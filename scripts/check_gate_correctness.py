"""Validate special values and real qkv row strides separately from throughput."""
import argparse,json,sys
from pathlib import Path
import torch,torch_npu
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels'))
from fused_gate import fused_gate
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.npu.set_device(0);torch.manual_seed(123)
results=[]
for m in [1,2,4,5,8,10,15,17,19,20,40,60,68,76,80]:
 for layout in ['contiguous','qkv_stride']:
    x=torch.randn((m,3072),device='npu',dtype=torch.bfloat16)
    # qkv contains two q-size branches plus two kv heads per TP rank.
    storage=torch.randn((m,7168),device='npu',dtype=torch.bfloat16)*8
    gate=storage[:,:6144].view(m,12,512)[:,:,256:] if layout=='qkv_stride' else storage[:,:3072].contiguous()
    ref=x*torch.sigmoid(gate.reshape_as(x));out=fused_gate(x,gate,False)
    torch.testing.assert_close(out,ref,rtol=0,atol=0,equal_nan=True)
    original=x.clone();ret=fused_gate(original,gate,True)
    assert ret.data_ptr()==original.data_ptr()
    torch.testing.assert_close(ret,ref,rtol=0,atol=0,equal_nan=True)
    results.append({'m':m,'layout':layout,'stride':gate.stride(),'exact':True})
x=torch.ones((1,3072),device='npu',dtype=torch.bfloat16)
g=torch.tensor([float('-inf'),-100,-20,-1,0,1,20,100,float('inf'),float('nan')],device='npu',dtype=torch.bfloat16).repeat(308)[:3072].reshape_as(x)
torch.testing.assert_close(fused_gate(x,g,False),x*torch.sigmoid(g),rtol=0,atol=0,equal_nan=True)
a.output.write_text(json.dumps({'cases':results,'special_values_exact':True},indent=2));print('PASS',len(results),'shape/layout cases and special values')
