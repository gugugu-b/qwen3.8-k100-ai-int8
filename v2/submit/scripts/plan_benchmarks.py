#!/usr/bin/env python3
"""Print a CPU-only test plan. Generated commands require an already started service."""
import argparse
from datetime import datetime,timezone
p=argparse.ArgumentParser()
p.add_argument('kind',choices=['missing','official-aligned','sla-probe','sharegpt'])
a=p.parse_args()
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
print('# Plan only; no service or benchmark was started.')
if a.kind=='sharegpt':
    for c in [1,4,16,32,64]:print(f'RUN_ID=sharegpt-{tag}-c{c} bash scripts/run.sh sharegpt {c}')
else:
    grids={'4k':[1,8,16,32,64,128],'64k':[1,8,16,32]}
    if a.kind=='missing':grids={'4k':[1,8,16,32,64,128],'64k':[1,8,16,32]}
    if a.kind=='sla-probe':grids={s:[1,2,4,8,16,32,64,128] for s in ['4k','64k']}
    if a.kind=='official-aligned':print('# Uses exact SourceFind reference client; this remains a contestant measurement, not an organizer-issued score.')
    for scene,cs in grids.items():
        for c in cs:print(f'RUN_ID={a.kind}-{tag}-{scene}-c{c} bash scripts/run.sh benchmark {scene} {c}')
    if a.kind=='sla-probe':print('# Run ascending one at a time; inspect each result. Refine integer points between pass/fail; repeat boundary. No maximum is established by this plan.')
