#!/usr/bin/env python3
"""Record CPU-only absolute formal windows for a new, explicitly named run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

p = argparse.ArgumentParser()
p.add_argument('--control-root', type=Path, required=True)
p.add_argument('--run-id', required=True)
p.add_argument('--package', type=Path, default=Path('/data/wangzy/v2/submit'))
a = p.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', a.run_id)
# The controller owns creation of this directory; observer never creates it.
for _ in range(120):
    if (a.control_root / 'execution-plan.json').is_file():
        break
    time.sleep(1)
else:
    raise SystemExit('Controller did not create execution-plan.json within 120s')
assert json.loads((a.control_root / 'execution-plan.json').read_text())['run_id'] == a.run_id
with (a.control_root / 'progress-observer.jsonl').open('x') as stream:
    while True:
        row = {'utc': datetime.now(timezone.utc).isoformat(), 'samples': {}}
        for key, c in [('45', 64), ('67', 128)]:
            f = a.package / f'result/runs/{a.run_id}-p{key}-server{c}-4k-c{c}/4k-c{c}/progress.json'
            if f.exists():
                try:
                    row['samples'][key] = json.loads(f.read_text())
                except (ValueError, OSError) as e:
                    row['samples'][key] = {'observer_error': repr(e)}
        row['controller_finished'] = (a.control_root / 'finished.json').exists()
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        if row['controller_finished']:
            break
        time.sleep(5)
(a.control_root / 'progress-observer-finished.json').write_text(json.dumps({
    'status': 'FINISHED', 'no_api_or_gpu_requests': True, 'interval_s': 5,
    'utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
