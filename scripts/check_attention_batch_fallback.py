"""Exercise concurrent requests to cover bs>1 fallback then bs=1 transition."""
import argparse,json,concurrent.futures,re
from pathlib import Path
import requests
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
tok=AutoTokenizer.from_pretrained('/home1/model/Qwen3.8-27B',trust_remote_code=True)
def generate(start):
 end=start+31
 prompt=tok.apply_chat_template([{'role':'user','content':f'Write every integer from {start} to {end} inclusive, in ascending order, separated by commas. Output only the list.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 s=requests.Session();s.trust_env=False;res=s.post('http://127.0.0.1:6688/generate',json={'text':prompt,'sampling_params':{'temperature':0,'max_new_tokens':200}},timeout=120);res.raise_for_status();body=res.json();numbers=[int(x) for x in re.findall(r'\d+',body['text'])]
 return {'start':start,'correct':numbers==list(range(start,end+1)),'response':body}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:rows=list(ex.map(generate,[1,33]))
rows.append(generate(65))
a.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2));print([(x['start'],x['correct']) for x in rows]);assert all(x['correct'] for x in rows)
