"""Run the original workload after readiness; persist command and exit status."""
import argparse,os,shlex,subprocess,time,json
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
s=requests.Session();s.trust_env=False
for _ in range(240):
    try:
        if s.get('http://127.0.0.1:6688/health',timeout=2).status_code==200:break
    except requests.RequestException:pass
    time.sleep(2)
else:raise RuntimeError('Server did not become ready')
base=Path(__file__).resolve().parents[1]
args=shlex.split((base/'benchmark.sh').read_text().replace('\\\n',' '))+['--output-file',str(a.run_dir/(a.label+'.jsonl')),'--output-details','--disable-tqdm']
(a.run_dir/(a.label+'_command.txt')).write_text(shlex.join(args)+'\n')
e=os.environ.copy();e['PYTHONPATH']='/sgl-workspace/sglang/python:'+e.get('PYTHONPATH','');e['NO_PROXY']='127.0.0.1,localhost';e['no_proxy']=e['NO_PROXY'];e['PYTHONUNBUFFERED']='1'
with (a.run_dir/(a.label+'.log')).open('w') as f:
    result=subprocess.run(args,cwd='/sgl-workspace',env=e,stdout=f,stderr=subprocess.STDOUT)
(a.run_dir/(a.label+'_exit.json')).write_text(json.dumps({'returncode':result.returncode}))
raise SystemExit(result.returncode)
