"""Bounded CPU scheduler; preserves the existing per-UID geometry contract."""
import argparse, concurrent.futures as futures, json, os, signal, sys, time, traceback
from pathlib import Path
import prepare_geometry as geometry
from core import ROOT, setup, cfg, dirs, path, read, readlines, sha, fp, append, atomic, lock, now, check_upstream
STOP = False

def request_stop(signum, frame):
    global STOP
    STOP = True

def terminal_records(rows, success, failures):
    known = {r['uid']: r for r in rows}
    good, bad = {}, {}
    for event in success:
        uid = event['uid']
        if uid not in known or uid in good or uid in bad:
            raise RuntimeError('unknown or duplicate successful UID: '+uid)
        good[uid] = event
    for event in failures:
        uid = event['uid']
        if uid not in known or uid in good or uid in bad:
            raise RuntimeError('unknown or duplicate failed UID: '+uid)
        bad[uid] = event
    return good, bad, [r for r in rows if r['uid'] not in good and r['uid'] not in bad]

def verify_reused(row, event, config, inventory, tools):
    folder = path(event['folder'])
    receipt = read(folder/'complete.json')
    conversion, water = geometry.contracts(row, config, tools, path('src/Hunyuan3D-2.1/hy3dshape/tools/render/render.py'), path('src/Hunyuan3D-2.1/hy3dshape/tools/watertight/watertight_and_sample.py'))
    water['input_ply_sha256'] = receipt['files']['converted/mesh.ply']['sha256']
    validation = {'water_receipt_sha256':sha(folder/'.stages/water.json'),'quality_sha256':sha(Path(geometry.__file__).with_name('quality.py')),'vae_extent':1.9998,'vae_points':81920,'points_seed':geometry.seed(config['seed'],'vae-points',row['uid'])}
    expected = {'schema':2,'uid':row['uid'],'raw_sha256':row['raw_sha256'],'conversion_fingerprint':fp(conversion),'water_fingerprint':fp(water),'validation_fingerprint':fp(validation),'caption_hash':row['caption_hash'],'inventory_sha256':inventory}
    if receipt['contract'] != expected or receipt['contract_sha256'] != fp(expected):
        raise RuntimeError('completed UID contract changed: '+row['uid'])
    geometry.verify_files(folder, receipt['files'])

def worker_init():
    setup()
    original = geometry.append
    def resource_append(file, record):
        import fcntl
        mutex = ROOT/'environment/abo_hssd_resource_append.lock'
        with mutex.open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            original(file, record)
    geometry.append = resource_append

def process_one(row, config, inventory, blender, python, tools):
    try:
        with lock('environment/abo_hssd_uid_locks/'+row['uid']+'.lock'):
            result = geometry.prepare_one(row, config, inventory, Path(blender), Path(python), tools, path('src/Hunyuan3D-2.1/hy3dshape/tools/render/render.py'), path('src/Hunyuan3D-2.1/hy3dshape/tools/watertight/watertight_and_sample.py'), dirs(config)[0]/'processed')
        return True, result
    except Exception as exc:
        return False, {'uid':row['uid'],'error':repr(exc),'traceback':traceback.format_exc()}

def main():
    import multiprocessing as mp
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    setup(); config = cfg(); check_upstream(config)
    if not 1 <= args.workers <= min(2,config['max_workers']): raise ValueError('workers must be 1 or 2')
    if args.limit is not None and args.limit < 1: raise ValueError('limit must be positive')
    assets, results = dirs(config); audit = results/'geometry_full'
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    with lock('environment/abo_hssd_geometry.lock'):
        rows = readlines(assets/'metadata/candidates.jsonl'); inventory = sha(results/'migration_20261027/source_inventory.json')
        if inventory != 'df419ae645debf29ab79b8ca6cb7f1c98b47cad1045207c967201f2e3c95fc94': raise RuntimeError('inventory fingerprint changed')
        tools = read(results/'preparation/tool_identity.json')['tools']
        good,bad,pending = terminal_records(rows,readlines(audit/'events.jsonl'),readlines(audit/'failures.jsonl'))
        def status(state, active=(), **extra):
            atomic(audit/'status.json',{'status':state,'pid':os.getpid(),'workers':args.workers,'completed':len(good),'failed':len(bad),'total':len(rows),'active_uids':list(active),'data_ready':False,**extra,**now()})
        status('verifying_resume',verified_completed=0)
        for i,row in enumerate(rows):
            if row['uid'] in good: verify_reused(row,good[row['uid']],config,inventory,tools)
            if i % 100 == 0: status('verifying_resume',verified_candidates=i+1)
            if STOP: status('paused'); return
        atomic(audit/'parallel_resume_verification.json',{'success_receipts_verified':len(good),'failed_preserved_without_retry':len(bad),'pending':len(pending),'candidate_sha256':sha(assets/'metadata/candidates.jsonl'),'workers':args.workers,'payloads_recomputed':False,**now()})
        atomic(audit/'process.json',{'pid':os.getpid(),'command':[sys.executable,'-B',str(Path(__file__).resolve())]+sys.argv[1:],'workers':args.workers,'data_ready':False,**now()})
        if args.limit is not None: pending = pending[:args.limit]
        iterator = iter(pending); finished = 0
        with futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
            active = {}
            def fill():
                if STOP: return
                while len(active) < args.workers:
                    row = next(iterator,None)
                    if row is None: break
                    job = pool.submit(process_one,row,config,inventory,str(ROOT/config['blender']),str(ROOT/config['runtime']/'bin/python'),tools)
                    active[job] = row['uid']
            fill(); status('running',active.values())
            while active:
                done,_ = futures.wait(active,timeout=10,return_when=futures.FIRST_COMPLETED)
                for job in done:
                    uid = active.pop(job); success, record = job.result()
                    if record['uid'] != uid: raise RuntimeError('UID result mismatch')
                    if success: append(audit/'events.jsonl',record); good[uid] = record
                    else: append(audit/'failures.jsonl',record); bad[uid] = record
                    finished += 1
                fill(); status('draining' if STOP else 'running',active.values(),session_attempted=finished)
        if len(good)+len(bad) == len(rows): status('complete_with_failures' if bad else 'complete')
        else: status('paused',reason='requested_stop' if STOP else 'bounded_batch_complete',session_attempted=finished)

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        atomic('results/abo_hssd_v1/geometry_full/parallel_failure.json',{'pid':os.getpid(),'error':repr(exc),'traceback':traceback.format_exc(),'data_ready':False,**now()})
        raise
