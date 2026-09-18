"""Collect bounded NPU profiles using the exact cached baseline prompts."""
import argparse
import json
import pickle
import time
from pathlib import Path
import requests

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--url', default='http://127.0.0.1:6688')
p.add_argument('--decode-only', action='store_true')
a = p.parse_args()
s = requests.Session(); s.trust_env = False
out = a.run_dir / 'profiles'; out.mkdir(exist_ok=True)
cache = Path('/root/.cache/sglang/benchmark/gen_shared_prefix_42_1_20_57600_6400_1000_Qwen2Tokenizer.pkl')
with cache.open('rb') as f:
    rows = pickle.load(f)

def post(route, body=None):
    r = s.post(a.url + route, json=body, timeout=900)
    r.raise_for_status()
    try: value = r.json()
    except ValueError: value = {'text': r.text}
    if isinstance(value, dict) and value.get('success') is False:
        raise RuntimeError(value)
    return value

def generate(index, label):
    payload = {'text': rows[index].prompt, 'sampling_params': {'temperature': 0, 'max_new_tokens': 128, 'ignore_eos': True}}
    t = time.time(); result = post('/generate', payload)
    (out / (label + '_response.json')).write_text(json.dumps({'elapsed_s': time.time()-t, 'result': result}, ensure_ascii=False, indent=2))
    print(label, 'finished', time.time()-t, flush=True)

def profile(label, index, stages, steps):
    dest = out / label; dest.mkdir(exist_ok=True)
    config = {'output_dir': str(dest.resolve()), 'activities': ['CPU', 'GPU'], 'num_steps': steps,
              'profile_by_stage': True, 'profile_stages': stages, 'with_stack': True, 'record_shapes': True}
    (out / (label + '_config.json')).write_text(json.dumps(config, indent=2))
    print('start', label, post('/start_profile', config), flush=True)
    generate(index, label)
    # All captures are bounded by scheduler steps; request completion permits trace flush.
    print('profile request complete', label, flush=True)

if not a.decode_only:
    post('/flush_cache')
    profile('cold_prefill', 0, ['prefill'], 2)
    profile('prefix_hit_prefill', 1, ['prefill'], 1)
profile('decode', 2, ['decode'], 6)
print('DONE', flush=True)
