#!/usr/bin/env python3
"""Check the TP2 DFlash runtime and basic request isolation; save evidence."""
import argparse,json,time
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
out=Path(__file__).resolve().parent/'backups/dflash2-tp2-20260909';out.mkdir(parents=True,exist_ok=True)
base='http://127.0.0.1:8080'
deadline=time.monotonic()+a.timeout
while True:
 try:
  r=requests.get(base+'/get_server_info',timeout=5);r.raise_for_status();info=r.json();break
 except (requests.RequestException,ValueError):
  if time.monotonic()>deadline:raise TimeoutError('DFlash2 readiness timed out')
  time.sleep(5)
(out/'server-info.json').write_text(json.dumps(info,indent=2))
args=info.get('server_args',info)
assert args['tp_size']==2 and args['nnodes']==2,args
assert args['speculative_algorithm']=='DFLASH',args
assert args['speculative_num_draft_tokens']==8,args
models=requests.get(base+'/v1/models',timeout=10);models.raise_for_status()
(out/'models.json').write_text(json.dumps(models.json(),indent=2))
checks=[('What is 17 + 25? Reply with only the number.','42'),('What is 13 + 16? Reply with only the number.','29')]
results=[]
for prompt,expected in checks:
 payload={'model':'qwen3.8-27b','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
 start=time.monotonic();r=requests.post(base+'/v1/chat/completions',json=payload,timeout=180);r.raise_for_status();body=r.json()
 results.append({'request':payload,'response':body,'elapsed_seconds':time.monotonic()-start})
 (out/'smoke.json').write_text(json.dumps(results,indent=2))
 assert body['choices'][0]['message']['content'].strip()==expected,body
payload={'model':'qwen3.8-27b','messages':[{'role':'user','content':'Write a Python function add(a, b) that returns their sum. Only output code.'}],'temperature':0,'max_tokens':128,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
r=requests.post(base+'/v1/chat/completions',json=payload,stream=True,timeout=180);r.raise_for_status()
events=[];done=False
for line in r.iter_lines():
 if not line:continue
 s=line.decode()
 if s=='data: [DONE]':done=True;break
 if s.startswith('data: '):events.append(json.loads(s[6:]))
(out/'stream.json').write_text(json.dumps(events,indent=2))
assert done and any((e.get('usage') or {}).get('completion_tokens',0)>0 for e in events),events
print('PASS: TP2 DFLASH, draft 8, two arithmetic checks, streaming with usage',flush=True)
