"""Small deterministic smoke set; not a replacement for model quality eval."""
import argparse,json
from pathlib import Path
import requests
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
tok=AutoTokenizer.from_pretrained('/home1/model/Qwen3.8-27B',trust_remote_code=True)
filler='Archive record: the weather station recorded ordinary readings and no special events.\n'
items=[
 ('arithmetic','Compute 17 * 23. Reply with only the integer.','391'),
 ('sorting','Sort these numbers ascending: 9, -3, 12, 0, 5. Output only the sorted list.','-3'),
 ('json','Return exactly this JSON object and no commentary: {"status":"ok","count":7}','"count"'),
 ('chinese','请只回答一个数字：一共有12盒铅笔，每盒8支，送出17支后还剩多少支？','79'),
 ('long_retrieval',filler*1800+'The secret access code is ZEBRA-7291.\n'+filler*1800+'\nWhat is the secret access code? Reply only with the code.','ZEBRA-7291'),
 ('long_arithmetic',filler*3600+'\nIgnore the archive records. Compute 123 + 456. Reply only with the integer.','579'),
]
s=requests.Session();s.trust_env=False;results=[]
for name,text,expected in items:
 prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 response=s.post('http://127.0.0.1:6688/generate',json={'text':prompt,'sampling_params':{'temperature':0,'max_new_tokens':128},'return_logprob':True,'logprob_start_len':-1,'top_logprobs_num':1},timeout=600);response.raise_for_status();body=response.json()
 row={'name':name,'expected_substring':expected,'contains_expected':expected in body.get('text',''),'response':body};results.append(row);a.output.write_text(json.dumps(results,ensure_ascii=False,indent=2));print(name,row['contains_expected'],repr(body.get('text','')[:180]),flush=True)
