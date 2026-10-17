#!/usr/bin/env python3
"""Frozen full GSM8K replay; unmodified EvalScope 1.6.1 scorer, no prompt search."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE/'quality-vendor'))
spec = importlib.util.spec_from_file_location('original_math_parser', SOURCE/'evalscope_math_parser_1_6_1.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)
def score(text, target):
    answer = parser.extract_answer(text)
    return answer, float(parser.math_equal(answer, parser.strip_answer_string(target)))
def reference():
    ref = SOURCE/'gsm8k-reference.jsonl'
    assert sha(ref) == 'b676b0128b716b38d3200ae07f21f51a9f8e5b116eda760a3cc28ea5e060fed2'
    assert sha(SOURCE/'evalscope_math_parser_1_6_1.py') == '809d4d47631ccb3b7da8c751f08d3ee94df5273a1ef1aeb252ec2f6b260f8b87'
    rows = [json.loads(s) for s in ref.read_text().splitlines()]
    assert len(rows) == len({r['index'] for r in rows}) == 1319
    for r in rows:
        assert r['input'].startswith('**User**: \n')
        a, c = score(r['sample_score']['score']['prediction'], r['target'])
        assert a == r['sample_score']['score']['extracted_prediction'] and c == r['sample_score']['score']['value']['acc'], r['index']
    return sorted(rows, key=lambda r:r['index'])

async def evaluate(args, rows):
    import aiohttp
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'responses').mkdir()
    old = sum(r['sample_score']['score']['value']['acc'] for r in rows)
    save(out/'manifest.json', dict(protocol='Frozen archived EvalScope 1.6.1 full GSM8K main, 4-shot',
        reference_sha256=sha(SOURCE/'gsm8k-reference.jsonl'), scorer_sha256=sha(SOURCE/'evalscope_math_parser_1_6_1.py'),
        reference_scorer_reproduction='1319/1319 PASS', old_correct=old, old_accuracy=old/len(rows),
        base_url=args.base_url, concurrency=32, temperature=0.0, max_tokens='OMITTED, matching archived EvalScope GenerateConfig',
        scoring_field='message.content only, reasoning excluded as in original EvalScope completion parser',
        requests=1319, threshold_accuracy=0.95, minimum_correct=math.ceil(1319*.95), errors_in_denominator=True,
        final_official_precision_protocol='NOT_VERIFIED; official exact data revision/subset/script not supplied',
        dedicated_instance=True, simultaneous_other_TP2_performance_instances=True,
        baseline_comparison='Archived v1 4-shot outputs, reproduced scorer; historical comparison only'))
    limit = asyncio.Semaphore(32)
    results = []
    started = time.monotonic()
    def progress():
        p=dict(completed=len(results), total=len(rows), correct=sum(r['correct'] for r in results),
            errors=sum(bool(r['error']) for r in results), elapsed_s=time.monotonic()-started)
        save(out/'progress.json', p)
        print(json.dumps(p), flush=True)
    with (out/'requests.jsonl').open('x') as log:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200),
                connector=aiohttp.TCPConnector(limit=32), trust_env=False) as client:
            async def one(r):
                async with limit:
                    payload=dict(model='Qwen3.8-27B-Channel-INT8-w8a8', temperature=0.0,
                        messages=[dict(role='user', content=r['input'][len('**User**: \n'):])])
                    begin=time.monotonic()
                    rec=dict(index=r['index'], target=r['target'], correct=0, answer=None, error=None,
                        old_correct=r['sample_score']['score']['value']['acc'],
                        prompt_sha256=hashlib.sha256(payload['messages'][0]['content'].encode()).hexdigest())
                    try:
                        async with client.post(args.base_url+'/v1/chat/completions', json=payload) as response:
                            raw=await response.read()
                            responsepath=out/'responses'/f"{r['index']}.json"
                            responsepath.write_bytes(raw)
                            rec.update(http_status=response.status, response=str(responsepath.relative_to(out)), response_sha256=sha(responsepath))
                            data=json.loads(raw)
                            assert response.status==200 and not data.get('error'), (response.status, data.get('error'))
                        choice=data['choices'][0]
                        message=choice['message'];text=message.get('content') or ''
                        rec.update(finish_reason=choice.get('finish_reason'), usage=data.get('usage'),
                            content_replacement_chars=text.count('\ufffd'),
                            reasoning_replacement_chars=(message.get('reasoning_content') or '').count('\ufffd'))
                        rec['answer'],rec['correct']=score(text,r['target'])
                    except Exception as e:
                        rec['error']=repr(e)
                    rec['latency_s']=time.monotonic()-begin
                    results.append(rec)
                    log.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                    if len(results)%10==0:progress()
            await asyncio.gather(*(one(r) for r in rows))
    progress()
    correct=sum(r['correct'] for r in results)
    errors=[r['index'] for r in results if r['error']]
    truncated=[r['index'] for r in results if r.get('finish_reason')=='length']
    textflags=[r['index'] for r in results if r.get('content_replacement_chars',0) or r.get('reasoning_replacement_chars',0)]
    threshold=correct/len(rows)>=.95
    clean=not errors and not truncated and not textflags
    summary=dict(status='LOCAL_PROTOCOL_PASS' if threshold and clean else 'REVIEW_REQUIRED',
        benchmark='GSM8K', requests=len(rows),completed=len(results),correct=correct,accuracy=correct/len(rows),
        community_threshold=.95,threshold_pass=threshold,minimum_correct=math.ceil(len(rows)*.95),
        errors=len(errors),error_indices=errors, text_flag_indices=textflags,truncated_indices=truncated,
        extraction_failed_indices=[r['index'] for r in results if not r['error'] and not r['answer']],
        old_correct=old,old_accuracy=old/len(rows),delta_percentage_points=(correct-old)/len(rows)*100,
        regressed_indices=[r['index'] for r in results if not r['error'] and r['old_correct']==1 and r['correct']==0],
        fixed_indices=[r['index'] for r in results if not r['error'] and r['old_correct']==0 and r['correct']==1],
        elapsed_s=time.monotonic()-started, official_precision_admission='PENDING_EXACT_OFFICIAL_PROTOCOL',
        comparison='Historical v1, identical frozen prompts and original scorer; not a contemporaneous A/B')
    save(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--base-url',default='http://127.0.0.1:31400');p.add_argument('--check-reference',action='store_true')
    a=p.parse_args();rows=reference();print('Frozen scorer reproduction 1319/1319 PASS',flush=True)
    if not a.check_reference:
        assert a.output
        asyncio.run(evaluate(a,rows))
