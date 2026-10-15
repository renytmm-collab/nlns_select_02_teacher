from pathlib import Path
import numpy as np
from .core import read,verify,sha,digest,ROOT,stream

STAGES=('dev-states','dev-candidates','dev-features','collect-dev-a','collect-dev-v','fit-oof','select-models','diagnostics','benchmark')

def old_artifact(source,name):
    d=Path(source)/'stages'/name; r=verify(read(d/'complete.json'))
    for f,h in r['files'].items():
        if Path(f).name!=f or sha(d/f)!=h: raise ValueError('Historical receipt/file mismatch: '+name+'/'+f)
    return verify(read(d/'data.json'))

def audit_source(c):
    source=Path(c['development_source']); repo=Path(c['source_repo'])
    m=verify(read(source/'manifest.json')); mismatches=[]; raw_mismatches=[]; missing=[]
    for f,h in m['identity']['files'].items():
        p=repo/f
        if not p.is_file(): missing.append(f); continue
        normalized=__import__('hashlib').sha256(p.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()
        if sha(p)!=h: raw_mismatches.append({'path':f,'expected_normalized':h,'actual_bytes':sha(p),'actual_normalized':normalized})
        if normalized!=h: mismatches.append({'path':f,'expected':h,'actual':normalized})
    if missing or any(x['path']!='protocol/UPSTREAM_BASELINE.json' for x in mismatches):
        raise ValueError('Core historical source drift or missing files')
    verified={}
    for s in STAGES:
        old_artifact(source,s); verified[s]=verify(read(source/'stages'/s/'complete.json'))['digest']
    cm=verify(read(Path(c['confirmation_source'])/'manifest.json'))
    calibration=Path(cm['references']['calibration'])
    provenance=read(ROOT/'vendor/PROVENANCE.json')
    for r in provenance:
        if sha(ROOT/'vendor'/r['path'])!=r['sha256']: raise ValueError('Copied source drift')
    return {'source_manifest':m['digest'],'source_config':m['config'],'stage_receipts':verified,
            'source_mismatches':mismatches,'raw_hash_differences':raw_mismatches,'missing':missing,'copied_sources':provenance,
            'confirmation_manifest':cm['digest'],'historical_calibration_path':str(calibration),
            'historical_calibration_present':calibration.is_dir(),
            'historical_calibration_completely_verified':False,
            'history_warning':'Referenced server calibration is not verified by a copied manifest; new paired calibration mandatory',
            'historical_costs':{s:verify(read(source/'stages'/s/'export.json'))['costs'] for s in ('dev-states','collect-dev-a','collect-dev-v','benchmark')}}

def load_labels(source,name,tables,repeats,root,old=True,states=None):
    from point_select.identity import words as old_words
    from .core import words
    d=Path(source)/'stages'/name; group={'collect-dev-a':'A','collect-dev-v':'V','collect-test-b':'B'}[name]
    expected={(t['instance'],t['step'],cid) for t in tables for cid in t['unique']}
    stats={k:{'seen':set(),'z':0.,'events':0,'y':0.} for k in expected}
    action_identity={}
    if states is not None:
        from point_select.features import audit as feature_audit
        from point_select.data import cost
        for t in tables:
            state=states[t['instance'],t['step']]
            for cid,customers in t['unique'].items():
                action_identity[t['instance'],t['step'],cid]=(cost(state),feature_audit(state,customers,[0.,0.])['input_points'])
    batches={}
    path=d/('rows.jsonl' if old else 'labels.jsonl')
    for r in stream(path):
        key=(r['instance'],r['step'],r['candidate']); rep=r['repeat']
        if key not in stats or rep in stats[key]['seen'] or not 0<=rep<repeats: raise ValueError('Label coverage identity')
        if r['status']!='ok' or r.get('fixture') is not False or r.get('repair_invoked') is not True:
            raise ValueError('Failed/uncertain/synthetic labels')
        if r['group']!=group: raise ValueError('Wrong label partition')
        seed=old_words(root,group,*key,rep) if old else words(root,group,*key,rep)
        if r['seed_words']!=seed: raise ValueError('Repair semantic seed mismatch')
        before,after=r['cost_before'],r['cost_after']
        if states is not None:
            frozen_before,natural=action_identity[key]
            if abs(before-frozen_before)>1e-10*max(1,frozen_before) or r['input_points']!=natural:
                raise ValueError('Label snapshot cost or natural Destroy input differs')
        if not np.isfinite([before,after]).all() or min(before,after)<=0: raise ValueError('Invalid costs')
        y=(before-after)/before
        if abs(r['y']-y)>1e-14 or r['z']!=max(r['y'],0): raise ValueError('Per-repeat truncation mismatch')
        if r['input_points']!=r['padded_input_points'] or r['batch_size']!=16: raise ValueError('Padding/batch mismatch')
        b=batches.setdefault(r['batch'],{'key':key,'reps':set()})
        if b['key']!=key or rep in b['reps']: raise ValueError('Heterogeneous/duplicate batch')
        b['reps'].add(rep); st=stats[key]; st['seen'].add(rep); st['z']+=r['z']; st['y']+=r['y']; st['events']+=r['y']>0
    if any(len(s['seen'])!=repeats for s in stats.values()) or any(len(b['reps'])!=16 for b in batches.values()):
        raise ValueError('Incomplete label coverage')
    return {k:{'target':s['z']/repeats,'event_probability':s['events']/repeats,'mean_y':s['y']/repeats} for k,s in stats.items()}
