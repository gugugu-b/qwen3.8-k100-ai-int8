#!/usr/bin/env python3
"""CPU filesystem observer for absolute formal-round time windows; no API/GPU use."""
import json,time,datetime
from pathlib import Path
control=Path('/data/wangzy/v2/retest-server64-128-20261017-a')
root=Path('/data/wangzy/v2/submit')
output=control/'progress-observer.jsonl'
assert control.is_dir() and not output.exists()
with output.open('x') as stream:
    while True:
        row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'samples':{}}
        for key,c in [('45',64),('67',128)]:
            p=root/f'result/runs/high-concurrency-20261017-a-p{key}-server{c}-4k-c{c}/4k-c{c}/progress.json'
            if p.exists():
                try:row['samples'][key]=json.loads(p.read_text())
                except Exception as e:row['samples'][key]={'observer_error':repr(e)}
        row['controller_finished']=(control/'finished.json').exists()
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
        if row['controller_finished']:break
        time.sleep(5)
(control/'progress-observer-finished.json').write_text(json.dumps({'status':'FINISHED','no_api_or_gpu_requests':True,'interval_s':5,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
