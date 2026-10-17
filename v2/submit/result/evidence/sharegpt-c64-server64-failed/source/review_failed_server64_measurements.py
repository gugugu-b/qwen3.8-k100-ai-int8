#!/usr/bin/env python3
"""CPU-only forensic measurement report; preserves and never waives original text failure."""
import argparse,hashlib,json,math,re,statistics
from pathlib import Path
RUN='submit-current-c64-server64-pair01-20261016-a'
RT='experiments/performance-validation-c64-server64-20261016'
MUTABLE={'artifact-sha256.json','performance-summary.json','finished','final-state.json','failed-state.json','control-sha256.json'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def q(a,p):
 a=sorted(a);x=(len(a)-1)*p;i=int(x);return a[i]*(1-x+i)+a[min(i+1,len(a)-1)]*(x-i)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();out=root/'result'/RUN
 assert not a.out.exists()
 manifests={n:read(out/n) for n in ['artifact-sha256.json','control-sha256.json']}
 for name,m in manifests.items():
  for rel,digest in m.items():
   p=root/rel;assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and p.resolve().is_relative_to(root)
   assert p.is_file() and sha(p)==digest,(name,rel)
 raw=root/RT/'result'/(RUN+'-c64-candidate')/'sharegpt-c64'
 expected={str(p.relative_to(root)) for p in out.rglob('*') if p.is_file() and p.name not in MUTABLE and not any(part.startswith('review-') for part in p.relative_to(out).parts)}
 expected.update(str(p.relative_to(root)) for p in raw.rglob('*') if p.is_file())
 assert set(manifests['artifact-sha256.json'])==expected
 assert set(manifests['control-sha256.json'])=={str(p.relative_to(root)) for p in out.rglob('*') if p.is_file() and p.name!='control-sha256.json'}
 state=read(out/'failed-state.json');assert state['run']==RUN and state['phase']=='failed' and state['physical_pair']==[0,1]
 assert not (out/'finished').exists() and not (out/'stages/c64-candidate.json').exists() and not (raw/'validated_summary.json').exists()
 release=read(out/'card-release.json');assert release['queue_success'] is False and release['running'] is False
 pre=read(out/'preflight.json');assert pre['scenes']==[64] and pre['kinds']==['candidate'] and pre['physical_pair']==[0,1]
 assert pre['server_max_running_requests']==pre['cuda_graph_max_bs']==64 and pre['accuracy']=='NOT_RUN'
 assert pre['seed']==20260912 and pre['prompts_per_round']==1000 and pre['measure_rounds']==3 and pre['full_warmup_rounds']==1
 assert pre['dataset_sha256']=='35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
 ev=out/'evidence/c64-candidate';info=read(ev/'server-info.json');args=info.get('server_args',info)
 required={'tp_size':2,'pp_size':1,'context_length':67584,'mem_fraction_static':.85,'max_running_requests':64,'max_prefill_tokens':65536,'chunked_prefill_size':8192,'mamba_scheduler_strategy':'no_buffer','cuda_graph_max_bs':64,'disable_cuda_graph':False,'disable_radix_cache':True,'attention_backend':'fa3','mm_attention_backend':'fa3','page_size':64,'speculative_num_steps':1,'speculative_eagle_topk':1,'speculative_num_draft_tokens':2,'custom_all_reduce_backend':'native','disable_custom_all_reduce':False}
 for key,value in required.items():assert args.get(key)==value,(key,args.get(key),value)
 assert args['speculative_algorithm'] in ['NEXTN','EAGLE']
 candidate=out/'source/candidate01'
 assert sha(candidate/'sitecustomize.py')=='da5b59bd28ea434356c7c96744c47f19493e540da7ba6f9bb890410e168c1610'
 assert sha(candidate/'native_bf16_ar_guard.py')=='f9d5ec98c70724dcb80857228681a6c03c619606643cdd6918c9084af51cd7f4'
 refpath=out/'source/v1-report-reference.json';assert sha(refpath)=='54c5ff586396dd90760eeb3fc3686f741927a5ef065294148ce293919ab4ef2d';reference=read(refpath)['sharegpt']['64'];assert reference['output_tps']==417.42
 text=(ev/'failure-server.log').read_text();assert text.count('[Native BF16 AR] Installed')>=2
 assert all('[Native BF16 AR] Eligible rank='+str(i) in text for i in [0,1]);assert 'cuda graph: True' in text
 error_markers=[x for x in ['Traceback (most recent call last)','VMFault','Watchdog caught collective operation timeout','out of memory'] if x in text];assert not error_markers,error_markers
 validator=(ev/'benchmark-command.log').read_text();assert 'validate_sharegpt.py' in validator and 'assert not any' in validator and 'generated_texts' in validator and 'AssertionError:' in validator
 rows=[];inputs=None;outputs=None
 labels=['full-warmup-01','round-01','round-02','round-03']
 assert {p.stem for p in raw.glob('*.jsonl')}==set(labels)
 for label in labels:
  p=raw/(label+'.jsonl');lines=[x for x in p.read_text().splitlines() if x.strip()];assert len(lines)==1
  d=json.loads(lines[0]);assert d['completed']==1000 and d['max_concurrency']==64 and d['dataset_name']=='sharegpt'
  for k in ['input_lens','output_lens','generated_texts','ttfts','itls','errors']:assert len(d[k])==1000
  assert not any(d['errors']) and all(isinstance(t,str) for t in d['generated_texts'])
  assert all(isinstance(v,int) and v>0 for v in d['input_lens']+d['output_lens'])
  assert all(math.isfinite(x) and x>=0 for x in d['ttfts'])
  assert all(math.isfinite(x) and x>=0 for xs in d['itls'] for x in xs)
  assert d['total_input_tokens']==sum(d['input_lens'])==331978 and d['total_output_tokens']==sum(d['output_lens'])==196501
  assert math.isclose(d['output_throughput'],sum(d['output_lens'])/d['duration'],rel_tol=1e-10)
  assert math.isclose(d['p99_ttft_ms'],q([x*1000 for x in d['ttfts']],.99),rel_tol=1e-8,abs_tol=1e-5)
  if inputs is None:inputs=d['input_lens'];outputs=d['output_lens']
  assert inputs==d['input_lens'] and outputs==d['output_lens']
  log=(raw/(label+'.log')).read_text();blines='\n'.join(x for x in log.splitlines() if 'benchmark_args' in x);assert blines
  for pat in [r'seed[=:]\s*20260912\b',r'num_prompts[=:]\s*1000\b',r'max_concurrency[=:]\s*64\b',r'warmup_requests[=:]\s*64\b',r'sharegpt_context_len[=:]\s*67584\b',r'profile[=:]\s*False\b']:assert re.search(pat,blines),pat
  bad=[i for i,t in enumerate(d['generated_texts']) if '\ufffd' in t];assert bad==[24,663],bad
  failure_details=[dict(index=i,characters=d['generated_texts'][i].count('\ufffd'),at_string_end=d['generated_texts'][i].endswith('\ufffd'),last_60_chars=d['generated_texts'][i][-60:],input_tokens=d['input_lens'][i],output_tokens=d['output_lens'][i]) for i in bad]
  assert all(x['at_string_end'] and x['characters']==1 for x in failure_details)
  lat={k:v for k,v in d.items() if k.endswith(('_ttft_ms','_tpot_ms','_itl_ms','_e2e_latency_ms')) and isinstance(v,(int,float))};assert all(math.isfinite(v) and v>=0 for v in lat.values())
  rows.append(dict(label=label,completed=1000,request_errors=0,input_tokens=d['total_input_tokens'],output_tokens=d['total_output_tokens'],duration_s=d['duration'],output_tps=d['output_throughput'],raw_sha256=sha(p),saved_latency_statistics_ms=lat,text_integrity='FAILED',replacement_character_requests=failure_details))
 formal=rows[1:];vs=[x['output_tps'] for x in formal];mean=statistics.mean(vs);lat={k:statistics.mean(x['saved_latency_statistics_ms'][k] for x in formal) for k in formal[0]['saved_latency_statistics_ms']}
 comparison={k:{'historical_reference':v,'latest':mean if k=='output_tps' else lat[k],'change_percent':((mean if k=='output_tps' else lat[k])/v-1)*100} for k,v in reference.items()}
 result=dict(status='MEASUREMENTS_COMPLETE_ORIGINAL_TEXT_INTEGRITY_FAILED',run=RUN,physical_pair=[0,1],client_concurrency=64,server_max_running_requests=64,cuda_graph_max_bs=64,formal_requests=3000,full_warmup_requests=1000,short_warmup_per_invocation=64,request_errors=0,output_tps_rounds=vs,mean_output_tps=mean,spread_percent=(max(vs)-min(vs))/mean*100,three_round_mean_saved_latency_statistics_ms=lat,historical_report_reference_comparison=comparison,prior_server32_descriptive_comparison={'mean_output_tps':417.09167316715815,'physical_pair':[6,7],'server_max_running_requests':32,'cuda_graph_max_bs':32,'change_percent':(mean/417.09167316715815-1)*100,'causal_attribution':False},rows=rows,original_queue_status='FAILED',original_validation_gate_pass=False,original_failed_state_sha256=sha(out/'failed-state.json'),original_artifact_manifest_sha256=sha(out/'artifact-sha256.json'),original_control_manifest_sha256=sha(out/'control-sha256.json'),artifact_count=len(manifests['artifact-sha256.json']),control_count=len(manifests['control-sha256.json']),evidence_exact_size_sha_and_manifest_coverage=True,original_raw_or_gate_modified=False,validator_failure='validate_sharegpt.py line26 rejects U+FFFD in generated_texts; same indices24,663 in all four complete invocations, each one trailing U+FFFD',root_cause_of_response_character='UNPROVEN: final token IDs and raw SSE text_chunks were not saved',accuracy='NOT_RUN',frozen=False,default_patch_changed=False,formal_acceptance=False,actual_model_graph_gate='REVIEW_REQUIRED',caveats=['Numeric measurements are complete, but original text integrity gate FAILED. No PASS is fabricated and no assertion is relaxed in the original validator.','Historical report and previous other-pair server32 comparisons are descriptive, not same-pair contemporaneous causal A/B.','Throughput and TTFT P99 reconstructed exactly. TPOT/ITL/E2E retain installed benchmark metrics; text_chunks and per-request final latency are absent, preventing exact reconstruction.','Mean of per-round percentiles is not a pooled percentile; observed spread is not a statistical noise upper bound.','No model quality/GSM8K evaluation, new GPU test, optimizer change, default promotion or freeze was performed.'])
 a.out.mkdir(parents=True);data=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 with (a.out/'measurement-report.json').open('x') as f:f.write(data)
 print(json.dumps({'status':result['status'],'mean_output_tps':mean,'historical_report_gain_percent':comparison['output_tps']['change_percent'],'original_validation_gate_pass':False,'report_sha256':sha(a.out/'measurement-report.json')}))
if __name__=='__main__':main()
