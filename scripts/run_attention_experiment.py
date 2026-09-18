"""Sequential validation and opt-in end-to-end experiment; logs all failures."""
import json,os,signal,subprocess,time
from pathlib import Path
b=Path(__file__).resolve().parents[1];r=Path((b/'ATTENTION_RUN').read_text().strip());old=Path((b/'CURRENT_RUN').read_text().strip())
while not (r/'baseline_gate_on_exit.json').exists():time.sleep(2)
assert json.loads((r/'baseline_gate_on_exit.json').read_text())['returncode']==0
base_env=os.environ.copy();base_env['PYTHONPATH']='/sgl-workspace/sglang/python:'+base_env.get('PYTHONPATH','')
e=base_env.copy();e['ASCEND_RT_VISIBLE_DEVICES']='6'
with (r/'attention_correctness.log').open('w') as f:
 subprocess.run(['python','-u',str(b/'scripts/check_mtp_attention.py'),'--output',str(r/'attention_correctness.json')],env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
print('CORRECTNESS_PASS',flush=True)
pid=int((old/'server_optimized.pid').read_text());os.killpg(pid,signal.SIGTERM)
import requests
s=requests.Session();s.trust_env=False
for _ in range(60):
 try:s.get('http://127.0.0.1:6688/health',timeout=1)
 except requests.RequestException:break
 time.sleep(1)
else:raise RuntimeError('Owned server did not stop')
e=base_env.copy();e['SGLANG_NPU_FUSED_ATTN_GATE']='1';e['SGLANG_NPU_MTP_INCREMENTAL']='1'
with (r/'server_attention.log').open('w') as f:
 p=subprocess.Popen(['bash',str(b/'start-qwen3.8-27b-prefix-cache.sh')],cwd='/sgl-workspace',env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'server_attention.pid').write_text(str(p.pid));print('SERVER_START',p.pid,flush=True)
subprocess.run(['python','-u',str(b/'scripts/run_benchmark.py'),'--run-dir',str(r),'--label','attention_on'],env=base_env,check=True)
print('BENCHMARK_DONE',flush=True)
