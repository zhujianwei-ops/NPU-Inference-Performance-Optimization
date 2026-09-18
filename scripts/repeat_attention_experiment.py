"""A/B/A/B replication, with isolated profiling and smoke probes."""
import json,os,signal,subprocess,time
from pathlib import Path
import requests
b=Path(__file__).resolve().parents[1];r=Path((b/'ATTENTION_RUN').read_text().strip());env=os.environ.copy();env['PYTHONPATH']='/sgl-workspace/sglang/python:'+env.get('PYTHONPATH','')
s=requests.Session();s.trust_env=False

def run_script(name,args,log):
 with (r/log).open('w') as f:subprocess.run(['python','-u',str(b/'scripts'/name),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def probes(label):run_script('attention_semantic_probes.py',['--output',str(r/(label+'_probes.json'))],label+'_probes.log')
def profile(label):
 d=r/(label+'_profiles');d.mkdir(exist_ok=True)
 run_script('collect_profile.py',['--run-dir',str(d),'--decode-only'],label+'_profile.log')
 run_script('analyze_profiles.py',[str(d)],label+'_profile_analysis.log')

def restart(pidfile,label,flag):
 pid=int((r/pidfile).read_text());os.killpg(pid,signal.SIGTERM)
 for _ in range(60):
  try:s.get('http://127.0.0.1:6688/health',timeout=1)
  except requests.RequestException:break
  time.sleep(1)
 else:raise RuntimeError('Owned server did not stop')
 e=env.copy();e['SGLANG_NPU_FUSED_ATTN_GATE']='1';e['SGLANG_NPU_MTP_INCREMENTAL']=flag
 with (r/(label+'_server.log')).open('w') as f:p=subprocess.Popen(['bash',str(b/'start-qwen3.8-27b-prefix-cache.sh')],cwd='/sgl-workspace',env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (r/(label+'_server.pid')).write_text(str(p.pid));print('START',label,p.pid,flush=True)
 run_script('run_benchmark.py',['--run-dir',str(r),'--label',label],label+'_driver.log')
 print('BENCHMARK_DONE',label,flush=True)

while not (r/'attention_on_exit.json').exists():time.sleep(2)
assert json.loads((r/'attention_on_exit.json').read_text())['returncode']==0
probes('attention_on');profile('attention_on')
restart('server_attention.pid','baseline_repeat','0')
probes('baseline_repeat');profile('baseline_repeat')
restart('baseline_repeat_server.pid','attention_repeat','1')
print('ALL_REPLICATION_DONE',flush=True)
