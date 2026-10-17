#!/usr/bin/env python3
"""One-time CPU forensic archive; never edits or restarts original failed RUN."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,traceback,datetime
from pathlib import Path
ROOT=Path('/data/wangzy/submit1009');RUN='submit-current-c64-server64-pair01-20261016-a'
OUT=ROOT/'result'/RUN;POST=ROOT/'result'/(RUN+'-failure-archive-20261016-a')
SOURCE=ROOT/'02-测试与结果/performance-validation-c64-server64-20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write(chr(10))
def copy(src,dst,h):
 assert sha(src)==h and not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(dst)==h==sha(src)
def main():
 assert not POST.exists();made=False
 with (ROOT/'result/gpu01-owner.lock').open('rb') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);lock_bytes=f.read()
  try:
   statepath=ROOT/'result'/(RUN+'-state.json');state=json.loads(statepath.read_text());assert state['phase']=='failed' and state['run']==RUN and state['physical_pair']==[0,1]
   assert not Path('/proc/'+str(state['pid'])).exists()
   assert statepath.read_bytes()==(OUT/'failed-state.json').read_bytes() and not (OUT/'finished').exists()
   c=json.loads(subprocess.check_output(['docker','inspect','qwen38-submit1009-perf01-server64-20261016'],text=True))[0]
   assert c['Name']=='/qwen38-submit1009-perf01-server64-20261016' and c['Config']['Labels']['qwen38.submit1009.server64']=='pair01'
   assert c['Image']=='sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
   assert not c['State']['Running'] and c['State']['Pid']==0 and 'HIP_VISIBLE_DEVICES=0,1' in c['Config']['Env']
   manifests={n:json.loads((OUT/n).read_text()) for n in ['artifact-sha256.json','control-sha256.json']};union={}
   for name,m in manifests.items():
    for rel,h in m.items():
     assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and (ROOT/rel).resolve().is_relative_to(ROOT)
     assert sha(ROOT/rel)==h and (rel not in union or union[rel]==h),rel;union[rel]=h
   assert set(manifests['control-sha256.json'])=={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file() and p.name!='control-sha256.json'}
   for p in [statepath,ROOT/'logs'/(RUN+'-job.log'),OUT/'artifact-sha256.json',OUT/'control-sha256.json']:union[str(p.relative_to(ROOT))]=sha(p)
   POST.mkdir();made=True;snap=POST/'submit1009'
   for rel,h in sorted(union.items()):copy(ROOT/rel,snap/rel,h)
   (POST/'control').mkdir();save(POST/'control/owned-container-stopped.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'name':c['Name'],'id':c['Id'],'state':c['State'],'physical_pair':[0,1],'queue_pid_absent':True,'lock_was_free':True})
   gpu=json.loads(subprocess.check_output(['/opt/dtk/bin/rocm-smi','-d','0','1','--showuse','--showmeminfo','vram','--json'],text=True));save(POST/'control/gpu01-observation.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':gpu,'own_container_stopped':True})
   for src in [Path(__file__).resolve(),SOURCE/'review_failed_server64_measurements.py']:copy(src,POST/'source'/src.name,sha(src))
   diag=ROOT/'result'/(RUN+'-failed-postcheck')
   for src in sorted(diag.rglob('*')):
    if src.is_file():copy(src,POST/'source-diagnostic'/src.relative_to(diag),sha(src))
   with (POST/'independent-review.log').open('x') as log:
    subprocess.run([sys.executable,'-B',str(POST/'source/review_failed_server64_measurements.py'),'--root',str(snap),'--out',str(POST/'remote-independent-review')],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
   report=POST/'remote-independent-review/measurement-report.json';obj=json.loads(report.read_text());assert obj['status']=='MEASUREMENTS_COMPLETE_ORIGINAL_TEXT_INTEGRITY_FAILED' and obj['original_validation_gate_pass'] is False
   for rel,h in union.items():assert sha(ROOT/rel)==h==sha(snap/rel)
   assert not Path('/proc/'+str(state['pid'])).exists();f.seek(0);assert f.read()==lock_bytes
   save(POST/'archive-review.json',{'status':'FAILED_RUN_EVIDENCE_FROZEN_MEASUREMENTS_AUDITED','original_run_status':'FAILED','original_text_integrity_gate':'FAILED','original_sources_raw_manifests_unchanged':True,'queue_pid_absent':True,'owned_container_stopped':True,'lock_contents_unchanged':True,'union_files':len(union),'report_sha256':sha(report),'accuracy':'NOT_RUN','root_cause':'UNPROVEN','gpu_retest_performed':False})
   files={str(p.relative_to(POST)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(POST.rglob('*')) if p.is_file()}
   save(POST/'archive-inventory.json',{'status':'FAILED_RUN_ARCHIVE_COMPLETE','files':files,'keys_relative_to':'POST','inventory_excludes_itself':True,'large_remote_only':[k for k,v in files.items() if v['bytes']>=16*1024*1024]})
   print(json.dumps({'status':'FAILED_RUN_ARCHIVE_COMPLETE','post':str(POST),'files':len(files),'report_sha256':sha(report),'mean_output_tps':obj['mean_output_tps'],'original_text_integrity_gate_pass':False}))
  except BaseException as e:
   if made and not (POST/'archive-failure.json').exists():save(POST/'archive-failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'original_modified':False})
   raise
  finally:fcntl.flock(f,fcntl.LOCK_UN)
if __name__=='__main__':main()
