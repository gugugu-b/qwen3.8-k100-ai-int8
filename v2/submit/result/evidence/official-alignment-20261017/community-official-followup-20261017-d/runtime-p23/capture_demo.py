
import json,sys,time,urllib.request
from pathlib import Path
out=Path(sys.argv[1]);port=sys.argv[2];events=[];start=time.monotonic()
payload={"model":"Qwen3.8-27B-Channel-INT8-w8a8","messages":[{"role":"user","content":"请用中文简短解释张量并行，再写一个将列表元素求和的Python函数。"}],"max_tokens":512,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}
(out/'request.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
req=urllib.request.Request('http://127.0.0.1:'+port+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=240) as r, (out/'sse-events.jsonl').open('x') as log:
 for line in r:
  row={'elapsed_s':time.monotonic()-start,'raw':line.decode('utf-8')}
  log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
  if line.startswith(b'data:'):
   raw=line[5:].strip()
   if raw==b'[DONE]':break
   value=json.loads(raw)
   if 'error' in value:raise RuntimeError(value['error'])
   for choice in value.get('choices',[]):
    delta=choice.get('delta',{});text=delta.get('reasoning_content','') or delta.get('content','')
    if text:events.append({'elapsed_s':row['elapsed_s'],'text':text});print(text,end='',flush=True)
print()
assert events
(out/'capture.json').write_text(json.dumps({'status':'REAL_API_STREAM_CAPTURED','elapsed_s':time.monotonic()-start,'events':events,'request':payload,'url':'http://127.0.0.1:'+port+'/v1/chat/completions','video_render':'PENDING_TERMINAL_REPLAY_RENDER_NOT_A_GUI_SCREEN_RECORDING'},ensure_ascii=False,indent=2)+'\n')
