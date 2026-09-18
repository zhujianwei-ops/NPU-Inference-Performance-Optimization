"""Check causal semantics, page permutation, graph Host-length rebinding."""
import argparse,json,sys
from pathlib import Path
import torch,torch_npu
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kernels'))
from mtp_incremental import expand_causal_lengths
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.npu.set_device(0);torch.manual_seed(123)
max_len=68501;page=128;pages=(max_len+page-1)//page
k=torch.randn((pages+8,page,512),device='npu',dtype=torch.bfloat16)
v=torch.randn_like(k);q=torch.randn((4,12,256),device='npu',dtype=torch.bfloat16)
bt=torch.randperm(pages+8,device='npu').to(torch.int32)[:pages].reshape(1,pages).contiguous()
btex=bt.repeat(4,1)
mask=torch.triu(torch.ones((2048,2048),device='npu',dtype=torch.bool),diagonal=1)
common=dict(block_size=128,num_heads=12,num_key_value_heads=2,scale=256**-.5)
def old(length):
 return torch_npu.npu_fused_infer_attention_score(q,k,v,block_table=bt,input_layout='TND',atten_mask=mask,sparse_mode=3,actual_seq_lengths=[4],actual_seq_lengths_kv=[length],**common)[0]
def new(length):
 return torch_npu.npu_fused_infer_attention_score(q.reshape(4,1,3072),k,v,block_table=btex,input_layout='BSH',sparse_mode=0,actual_seq_lengths_kv=expand_causal_lengths([length]),**common)[0].reshape_as(q)
def compare(x,y):
 d=(x.float()-y.float()).abs(); rel=(d.norm()/y.float().norm().clamp_min(1e-9)).item()
 torch.testing.assert_close(x,y,atol=.002,rtol=.02)
 assert rel<.006,rel
 return {'max_abs':d.max().item(),'relative_l2':rel}
results=[]
# Warmup before capture; use auto-dispatch so CPU attributes can be rebound.
new(256);torch.npu.synchronize();g=torch.npu.NPUGraph()
with torch.npu.graph(g,auto_dispatch_capture=True):captured=new(256)
for length in [4,5,127,128,129,255,256,257,2049,68424,68500,68501]:
 row={'length':length,'eager_vs_original':compare(new(length),old(length))}
 g.update(cpu_update_input=[{'actual_seq_lengths_kv':expand_causal_lengths([length])}]);g.replay();torch.npu.synchronize()
 row['graph_vs_eager']=compare(captured,new(length))
 if length<=257:
  ids=bt[0,:((length+127)//128)].long();kc=k[ids].reshape(-1,2,256)[:length].cpu().double().repeat_interleave(6,dim=1)
  vc=v[ids].reshape(-1,2,256)[:length].cpu().double().repeat_interleave(6,dim=1);qc=q.cpu().double();outs=[]
  for j in range(4):
   limit=length-4+j+1;scores=torch.einsum('hd,shd->hs',qc[j],kc[:limit])*(256**-.5)
   outs.append(torch.einsum('hs,shd->hd',scores.softmax(-1),vc[:limit]))
  oracle=torch.stack(outs).float().to('npu')
  oracle_errors={}
  for label,value in [('original',old(length)),('incremental',new(length))]:
   delta=(value.float()-oracle).abs();relative=(delta.norm()/oracle.norm()).item()
   oracle_errors[label]={'max_abs':delta.max().item(),'relative_l2':relative}
   # Both vendor paths accumulate/softmax in finite precision. Compare each
   # against FP64, and require the candidate to be no materially worse.
   assert relative < .006, oracle_errors
  assert oracle_errors['incremental']['max_abs'] <= max(.003,1.25*oracle_errors['original']['max_abs']),oracle_errors
  assert oracle_errors['incremental']['relative_l2'] <= max(.001,1.25*oracle_errors['original']['relative_l2']),oracle_errors
  row['oracle']=oracle_errors
 results.append(row);print(row,flush=True)
# Mutate q and page map between replays to prove live device inputs are read.
q.mul_(.7);bt.copy_(bt.flip(1));btex.copy_(bt.expand(4,-1));length=257
g.update(cpu_update_input=[{'actual_seq_lengths_kv':expand_causal_lengths([length])}]);g.replay();torch.npu.synchronize()
mutated=compare(captured,new(length))
# Empty graph-padding sequence must stay finite and zero.
padded=new(0);torch.npu.synchronize();assert torch.isfinite(padded).all() and torch.count_nonzero(padded)==0
assert expand_causal_lengths([0,4])==[0,0,0,0,1,2,3,4]
a.output.write_text(json.dumps({'cases':results,'mutated_input_graph':mutated,'zero_padding_passed':True},indent=2));print('PASS',flush=True)
