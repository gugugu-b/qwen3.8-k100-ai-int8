#!/usr/bin/env python3
"""Audit all raw responses against frozen indices, prompts, scores and summary."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate import reference, score, save, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
    refs={r['index']:r for r in reference()}
    records=[json.loads(s) for s in (out/'requests.jsonl').read_text().splitlines()]
    assert len(records)==len({r['index'] for r in records})==1319
    assert {r['index'] for r in records}==set(refs)
    total=0
    for r in records:
        ref=refs[r['index']]
        prompt=ref['input'][len('**User**: \n'):]
        assert r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
        assert r['target']==ref['target']
        if 'response' in r:
            file=out/r['response'];assert file.resolve().is_relative_to(out.resolve())
            assert sha(file)==r['response_sha256']
        if r['error']:
            assert r['correct']==0
            continue
        data=json.loads(file.read_text());choice=data['choices'][0];text=choice['message'].get('content') or ''
        answer,correct=score(text,ref['target'])
        assert answer==r['answer'] and correct==r['correct']
        assert r['content_replacement_chars']==text.count('\ufffd')
        assert r['reasoning_replacement_chars']==(choice['message'].get('reasoning_content') or '').count('\ufffd')
        assert r['finish_reason']==choice.get('finish_reason') and r['usage']==data.get('usage')
        total+=correct
    summary=json.loads((out/'summary.json').read_text())
    assert summary['correct']==total and summary['accuracy']==total/1319
    assert summary['threshold_pass']==(total/1319>=.95)
    assert summary['errors']==sum(bool(r['error']) for r in records)
    save(out/'independent-review.json',dict(status='RAW_RESPONSE_RESCORING_AND_1319_INDEX_PROMPT_SHA_PASS',
        requests=1319,correct=total,accuracy=total/1319,threshold_pass=total/1319>=.95,
        note='Integrity and local-protocol rescore only; final official data/script equivalence remains pending'))
    print('Full raw-response re-score and prompt/index integrity PASS',flush=True)
if __name__=='__main__':main()
