#!/usr/bin/env python3
"""A streaming text Demo; does not count as an accuracy benchmark."""
import json
import os
import sys
import requests
url=f'http://127.0.0.1:{os.environ.get("PORT","31014")}/v1/chat/completions'
s=requests.Session(); s.trust_env=False
prompt=sys.argv[1] if len(sys.argv)>1 else '请用中文简要介绍你的能力。'
payload={'model':'Qwen3.8-27B-Channel-INT8-w8a8','messages':[{'role':'user','content':prompt}],
         'max_tokens':512,'temperature':0,'stream':True}
print('用户：'+prompt,flush=True)
with s.post(url,json=payload,stream=True,timeout=(10,180)) as r:
    r.raise_for_status()
    for line in r.iter_lines(chunk_size=1):
        if not line or not line.startswith(b'data:'):continue
        raw=line[5:].strip()
        if raw==b'[DONE]':break
        event=json.loads(raw)
        if 'error' in event:raise RuntimeError(event['error'])
        for choice in event.get('choices',[]):
            delta=choice.get('delta',{})
            content=delta.get('reasoning_content','') or delta.get('content','')
            print(content,end='',flush=True)
print()
