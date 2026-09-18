"""Run validation serially after the benchmark, avoiding device contention."""
import json,subprocess,os,time
from pathlib import Path
b=Path(__file__).resolve().parents[1];r=Path((b/'CURRENT_RUN').read_text().strip())
status=r/'optimized_original_exit.json'
while not status.exists():time.sleep(5)
if json.loads(status.read_text())['returncode']:raise RuntimeError('Benchmark failed')
e=os.environ.copy();e['ASCEND_RT_VISIBLE_DEVICES']='6';e['PYTHONPATH']='/sgl-workspace/sglang/python:'+e.get('PYTHONPATH','')
subprocess.run(['python',str(b/'scripts/check_gate_correctness.py'),'--output',str(r/'gate_correctness.json')],env=e,check=True)
optimized=r/'optimized_profiles';optimized.mkdir(exist_ok=True)
subprocess.run(['python','-u',str(b/'scripts/collect_profile.py'),'--run-dir',str(optimized),'--decode-only'],env=e,check=True)
subprocess.run(['python',str(b/'scripts/analyze_profiles.py'),str(optimized)],env=e,check=True)
print('POST_BENCHMARK_DONE',flush=True)
