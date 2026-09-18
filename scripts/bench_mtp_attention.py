"""Compare causal TND verify with equivalent per-token incremental attention."""
import argparse,json,statistics,time
from pathlib import Path
import torch,torch_npu
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lengths',nargs='+',type=int,default=[4096,68500]);a=p.parse_args()
torch.npu.set_device(0);torch.manual_seed(42)

def timing(fn):
 for _ in range(3):fn()
 torch.npu.synchronize();g=torch.npu.NPUGraph()
 with torch.npu.graph(g):
  for _ in range(4):fn()
 vals=[]
 for _ in range(5):
  st=torch.npu.Event(enable_timing=True);en=torch.npu.Event(enable_timing=True);st.record()
  for _ in range(10):g.replay()
  en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000/40)
 return statistics.median(vals)
res=[]
for length in a.lengths:
 for bs in [1,2]:
  width=4;pages=(length+127)//128
  q=torch.randn((bs*width,12,256),device='npu',dtype=torch.bfloat16)
  k=torch.randn((bs*pages,128,512),device='npu',dtype=torch.bfloat16);v=torch.randn_like(k)
  bt=torch.arange(bs*pages,device='npu',dtype=torch.int32).reshape(bs,pages)
  mask=torch.triu(torch.ones((2048,2048),device='npu',dtype=torch.bool),diagonal=1)
  common=dict(block_size=128,num_heads=12,num_key_value_heads=2,scale=256**-0.5)
  lens=[length-i*13 for i in range(bs)]
  old=lambda:torch_npu.npu_fused_infer_attention_score(q,k,v,block_table=bt,input_layout='TND',atten_mask=mask,sparse_mode=3,actual_seq_lengths=[width*(i+1) for i in range(bs)],actual_seq_lengths_kv=lens,**common)[0]
  btex=bt.repeat_interleave(width,dim=0);le=[max(0,l-width+j+1) for l in lens for j in range(width)]
  new=lambda:torch_npu.npu_fused_infer_attention_score(q.reshape(bs*width,1,3072),k,v,block_table=btex,input_layout='BSH',sparse_mode=0,actual_seq_lengths_kv=le,**common)[0].reshape_as(q)
  expected=old();actual=new();torch.npu.synchronize();diff=(actual.float()-expected.float()).abs();rel=diff.norm()/expected.float().norm()
  torch.testing.assert_close(actual,expected,atol=0.002,rtol=0.02)
  row=dict(length=length,bs=bs,max_abs=diff.max().item(),relative_l2=rel.item(),old_us=timing(old),new_us=timing(new));row['speedup']=row['old_us']/row['new_us'];res.append(row);a.output.write_text(json.dumps(res,indent=2));print(row,flush=True)
