import copy
import math
import time
from pathlib import Path
import numpy as np
from .core import *
from .source import audit_source,old_artifact,load_labels
from .features import build_table,vector,pool
from .models import specs,fit,predict,metric
from .analysis import interval,planning,evaluate
from point_select.data import cost

def imported(run): return artifact(run,'import-development')

def initialize_development(run,c):
    evidence=audit_source(c); source=Path(c['development_source'])
    sd=old_artifact(source,'dev-states'); tables=old_artifact(source,'dev-candidates')['tables']
    fd=old_artifact(source,'dev-features'); selected=old_artifact(source,'select-models')
    old_oof=old_artifact(source,'fit-oof')
    if sd.get('fixture',False) or evidence['source_config']['development_instances']!=64: raise ValueError('Real 64-instance development required')
    states=sd['states']; folds=sd['split']['folds']
    if len(states)!=128 or set(folds)!={str(i) for i in range(64)} or set(folds.values())!={0,1,2,3}: raise ValueError('Development state/fold scope')
    frozen_states={(s['instance'],s['step']):s['state'] for s in states}
    aa=load_labels(source,'collect-dev-a',tables,32,evidence['source_config']['root_seed'],states=frozen_states)
    vv=load_labels(source,'collect-dev-v',tables,64,evidence['source_config']['root_seed'],states=frozen_states)
    order=[[t['instance'],t['step'],s['slot'],s['candidate']] for t in fd['tables'] for s in t['slots']]
    if order!=old_oof['row_order']: raise ValueError('Historical OOF row-order mismatch')
    legal={}; unavailable={}
    for method,model in selected['models'].items():
        oo=next((x for x in old_oof['specs'] if x['spec']==model['spec']),None)
        if oo is None: unavailable[method]='No matching legal OOF'; continue
        for f in oo['folds']:
            train=set(f['train_instances']); val=set(f['validation_instances'])
            if train&val or train|val!=set(range(64)) or any(folds[str(i)]!=f['fold'] for i in val): raise ValueError('Historical OOF leakage')
        if len(oo['scores'])!=len(order) or selected['selected_oof_scores'][method]!=oo['scores']: raise ValueError('OOF score mismatch')
        legal[method]=oo['scores']
    snap={(s['instance'],s['step']):s for s in states}
    ordered_states=[snap[t['instance'],t['step']] for t in fd['tables']]
    by_table={(t['instance'],t['step']):t for t in tables}
    ordered_tables=[by_table[t['instance'],t['step']] for t in fd['tables']]
    historical=set()
    coverage=[]
    for path in c['historical_catalogs']:
        data=verify(read(path)) if read(path).get('digest') else read(path)
        count=0
        for r in data['instances']:
            h=r.get('content_sha256')
            if h: historical.add(h); count+=1
        coverage.append({'path':str(path),'sha256':sha(path),'entries':count})
    # Development and historical 192 catalogs mandatory; no test A/B files read.
    for src in (source,Path(c['confirmation_source'])):
        stage_name='dev-states' if src==source else 'test-states'
        path=src/'stages'/stage_name/'catalog.json'
        receipt_data=verify(read(src/'stages'/stage_name/'complete.json'))
        if sha(path)!=receipt_data['files']['catalog.json']: raise ValueError('Historical catalog receipt')
        cat=verify(read(path)); hashes=[r['content_sha256'] for r in cat['instances']]
        historical.update(hashes); coverage.append({'path':str(path),'sha256':sha(path),'entries':len(hashes)})
    if len(historical)<256: raise ValueError('Required historical catalogs incomplete/overlap')
    init(run,c)
    with stage(run,'import-development') as d:
        rows=[]
        for t in ordered_tables:
            for s in t['slots']:
                key=t['instance'],t['step'],s['candidate']
                rows.append({'instance':key[0],'step':key[1],'slot':s['slot'],'candidate':key[2],'A':aa[key],'V':vv[key]})
        write(d/'data.json',sealed({'audit':evidence,'states':ordered_states,'tables':ordered_tables,'folds':folds,
                                   'label_means':rows,'old_models':selected['models'],'old_oof':legal,
                                   'old_oof_unavailable':unavailable,'historical_hashes':sorted(historical),'history_coverage':coverage,
                                   'fixture':False,'read_scope':'Old development A/V only; historical confirmation catalog and model identity'}))

def references(run,role,args):
    name=f'generate-{role}-references'; m=manifest(run); c=m['config']
    states=imported(run)['states'] if role=='dev' else artifact(run,'generate-test-states')['states']
    instances={s['instance']:s['state'] for s in states}
    cap_check(name,args.solver_budget,len(instances)*15)
    if args.solver_seconds!=5 or args.solver_threads!=1 or args.solver_seeds!=c['solver_seeds']: raise ValueError('Explicit solver budget required')
    from .reference import solve
    with stage(run,name,{'cpu_budget':args.solver_budget,'seeds':args.solver_seeds,'threads':1}) as d:
        j=Journal(d,args.solver_budget)
        try:
            result={str(i):solve(state,d/f'instance-{i}',c,j,i) for i,state in sorted(instances.items())}
            write(d/'data.json',sealed({'references':result,'cpu_limit_reserved':j.used,'selection':'minimum recomputed feasible cost, seed order tie','fixture':False}))
        finally: j.close()

def features(run,role):
    name=f'build-{role}-features'; c=manifest(run)['config']
    data=imported(run) if role=='dev' else artifact(run,'generate-test-states')
    tables=imported(run)['tables'] if role=='dev' else artifact(run,'build-test-candidates')['tables']
    states={(s['instance'],s['step']):s['state'] for s in data['states']}
    refs=artifact(run,f'generate-{role}-references')['references']; ft=[]
    for t in tables:
        primary=refs[str(t['instance'])]['primary']; rr=primary['routes'] if primary else None
        ft.append(build_table(states[t['instance'],t['step']],t,rr,c['root_seed']))
    with stage(run,name) as d: write(d/'data.json',sealed({'tables':ft,'history_features':[],'schema':sha(ROOT/'FEATURE_SPEC.md')}))

def dev_rows(run,group,label='A'):
    if label not in ('A','V'): raise ValueError('Development-only label scope')
    ft=artifact(run,'build-dev-features')['tables']; data=imported(run)
    means={(r['instance'],r['step'],r['slot'],r['candidate']):r for r in data['label_means']}
    return [{'instance':t['instance'],'step':t['step'],'slot':s['slot'],'candidate':s['candidate'],
             'values':vector(s,group),'target':means[t['instance'],t['step'],s['slot'],s['candidate']][label]['target'],
             'weight':1/64/2/16} for t in ft for s in t['slots']]

def fit_oof(run,max_fits):
    cap_check('fit-oof',max_fits,120); folds=imported(run)['folds']
    result={}
    with stage(run,'fit-oof',{'max_fits':max_fits,'read_labels':'A only'}) as d:
        journal=Journal(d,max_fits)
        try:
            for group in GROUPS:
                rows=dev_rows(run,group); options=[]
                for index,spec in enumerate(specs()):
                    scores=[None]*len(rows); lineage=[]
                    for f in range(4):
                        train=[r for r in rows if folds[str(r['instance'])]!=f]
                        jj=[j for j,r in enumerate(rows) if folds[str(r['instance'])]==f]
                        key=f'{group}/{index}/{f}'; journal.begin(key,1,{'spec':spec,'fold':f,'group':group})
                        model=fit(train,spec); val=[rows[j] for j in jj]; values=predict(model,[r['values'] for r in val])
                        for j,v in zip(jj,values): scores[j]=v
                        lineage.append({'fold':f,'train_instances':sorted({r['instance'] for r in train}),
                                        'validation_instances':sorted({r['instance'] for r in val}),'preprocessing':model['preprocessing']})
                        journal.event({'key':key,'status':'ok'})
                    options.append({'spec':spec,'scores':scores,'fold_lineage':lineage})
                result[group]=options
            write(d/'data.json',sealed({'groups':result,'fits':journal.used,'read_scope':'development A32 only','row_order':[[r['instance'],r['step'],r['slot'],r['candidate']] for r in rows]}))
        finally: journal.close()

def assess(run,max_fits):
    cap_check('assess-development',max_fits,5); data=imported(run); c=manifest(run)['config']
    ft=artifact(run,'build-dev-features')['tables']; oof=artifact(run,'fit-oof')['groups']; vv=dev_rows(run,'B0','V')
    allscores={}; selected={}; allmetrics={}; aux={}
    def gain(sc): return float(np.mean(list(metric(vv,sc).values())))
    def choose(options):
        best=max(gain(x['scores']) for x in options)
        return next(x for x in reversed(options) if best-gain(x['scores'])<=1e-12)
    for g in ('B0','Bcap'):
        selected[g]=choose(oof[g]); allscores[g]=selected[g]['scores']
    allscores.update(data['old_oof'])
    allscores['native_expectation']=[0.]*len(vv)
    for name in ('contact_edges','release_concentration'):
        allscores[name]=[s['rules'][name] for t in ft for s in t['slots']]
    baseline=max(allscores,key=lambda n:(gain(allscores[n]),-list(allscores).index(n)))
    fallback=[t['fallback'] for t in ft for _ in t['slots']]
    def fallback_scores(sc): return [allscores[baseline][j] if fallback[j] else v for j,v in enumerate(sc)]
    for g in ('Tdiff','Tready','Tshuffle'):
        selected[g]=choose([{**x,'raw_scores':x['scores'],'scores':fallback_scores(x['scores'])} for x in oof[g]])
        allscores[g]=selected[g]['scores']
    teacher=max(('Tdiff','Tready'),key=lambda n:(gain(allscores[n]),n=='Tdiff'))
    for name,sc in allscores.items(): allmetrics[name]=metric(vv,sc)
    for group in GROUPS:
        sc=selected[group].get('raw_scores',selected[group]['scores'])
        aux[group]={'mse_on_V':float(np.mean((np.asarray(sc)-np.asarray([r['target'] for r in vv]))**2)),
                    'correlation_on_V':float(np.corrcoef(sc,[r['target'] for r in vv])[0,1]) if np.std(sc)>0 and np.std([r['target'] for r in vv])>0 else None}
    contrasts={b:[100*(allmetrics[teacher][i]-allmetrics[b][i]) for i in allmetrics[teacher]] for b in dict.fromkeys([baseline,'Bcap'])}
    proceed=all(np.mean(v)>0 for v in contrasts.values())
    with stage(run,'assess-development',{'max_fits':max_fits,'read_scope':'development A/V; exploratory'}) as d:
        j=Journal(d,max_fits); models={}
        try:
            for g in GROUPS:
                j.begin(g,1,{'group':g,'spec':selected[g]['spec']}); models[g]=fit(dev_rows(run,g),selected[g]['spec']); j.event({'key':g,'status':'ok'})
            write(d/'data.json',sealed({'models':models,'selected_specs':{g:selected[g]['spec'] for g in GROUPS},
                    'selected_scores':allscores,'per_instance':allmetrics,'mean_pp':{n:100*gain(s) for n,s in allscores.items()},
                    'teacher':teacher,'baseline':baseline,'may_continue':proceed,'development_status':'DEVELOPMENT_GATE_PASS' if proceed else 'DEVELOPMENT_NOT_SUPPORTED',
                    'power':{b:planning(v,c['root_seed'],len(contrasts)) for b,v in contrasts.items()},'auxiliary':aux,
                    'fallback_fraction':sum(fallback)/len(fallback),'fits':j.used,'exploratory':True}))
        finally: j.close()

def calibrate(run,cap):
    cap_check('calibrate',cap,1024); c=manifest(run)['config']; data=imported(run)
    from .backend import Backend,repair_batch
    # Preselected lexicographic first 32 state/action identities; never label-dependent.
    actions=sorted((t['instance'],t['step'],cid,t['unique'][cid]) for t in data['tables'] for cid in t['unique'])[:32]
    states={(s['instance'],s['step']):s['state'] for s in data['states']}
    backend=Backend(c)
    with stage(run,'calibrate',{'cap':cap,'actions_digest':digest(actions),'tolerance_relative':1e-8}) as d:
        j=Journal(d,cap); comparisons=[]
        try:
            for i,step,cid,ids in actions:
                seeds=[words(c['root_seed'],'calibration',i,step,cid,r) for r in range(16)]
                single=[]
                for r,seed in enumerate(seeds):
                    _,rows=repair_batch(backend,states[i,step],ids,[seed],j,f'{i}/{step}/{cid}/single/{r}',{'instance':i,'step':step,'candidate':cid,'repeat_start':r,'profile':1})
                    single+=rows
                _,batched=repair_batch(backend,states[i,step],ids,seeds,j,f'{i}/{step}/{cid}/batch',{'instance':i,'step':step,'candidate':cid,'profile':16})
                for a,b in zip(single,batched):
                    if a['seed_words']!=b['seed_words'] or abs(a['cost_after']-b['cost_after'])>1e-8*max(1,abs(a['cost_after'])): raise ValueError('Paired calibration mismatch; real tests blocked')
                    comparisons.append({'single':a,'homogeneous':b})
            write(d/'data.json',sealed({'passed':True,'sampling_not_proof':True,'comparisons':comparisons,'charged':j.used,'environment':backend.environment,'fixture':False}))
        finally: j.close()

def freeze(run):
    a=artifact(run,'assess-development'); cal=artifact(run,'calibrate'); m=manifest(run); data=imported(run)
    if not a['may_continue'] or not cal['passed'] or cal['fixture'] or m['fixture']: raise ValueError('Development/calibration gate')
    # Test labels/state generation cannot exist before this receipt.
    if any((Path(run)/'stages'/n).exists() for n in DEPS if 'test' in n): raise ValueError('Premature test exposure')
    primary=list(dict.fromkeys([a['baseline'],'Bcap']))
    with stage(run,'freeze-plan') as d:
        write(d/'data.json',sealed({'teacher':a['teacher'],'baseline':a['baseline'],'models':a['models'],
              'old_models':data['old_models'],'test_ids':list(range(1000,1128)),
              'primary_comparisons':[[a['teacher'],b] for b in primary],
              'fallback':'B* if no feasible reference or ref_cost>=current_cost','delta_pp':.01,'family_alpha':.05,
              'bootstrap_repeats':10000,'root_seed':m['config']['root_seed'],'checkpoint':CHECKPOINT,
              'power':a['power'],'schema':sha(ROOT/'FEATURE_SPEC.md'),'history_coverage':data['history_coverage'],
              'missing_policy':'no exclusion; missing/uncertain Repair invalid, no feasible ref fallback',
              'sensitivity':'stages50/200,gap<=.05/>.05,structure<=.5/>.5,all3refs,drop1/drop2,Tshuffle',
              'budget':LIMITS,'mechanism':'Tready-Tdiff stable positive t/bootstrap, exploratory; GO mechanism only if T*=Tready'}))

def states(run,cap):
    c=manifest(run)['config']; plan=artifact(run,'freeze-plan'); data=imported(run); cap_check('generate-test-states',cap,25600)
    from .backend import Backend,payload,repair_batch
    backend=Backend(c)
    if backend.environment!=artifact(run,'calibrate')['environment']: raise ValueError('Calibration runtime identity drift')
    with stage(run,'generate-test-states',{'cap':cap}) as d:
        j=Journal(d,cap); seen=set(data['historical_hashes']); snaps=[]; catalog=[]
        try:
            for i in plan['test_ids']:
                current=payload(backend.generate(i)); h=digest({k:current[k] for k in ('original_locations','demand','capacity')})
                if h in seen: raise ValueError('Instance content collision; no replacement')
                seen.add(h); catalog.append({'instance':i,'initial':copy.deepcopy(current),'content_sha256':h,'seed_words':words(c['root_seed'],'instance',i)})
                from point_select.data import point
                for step in range(1,201):
                    ids,_=point(current,rng(c['root_seed'],'state_destroy',i,step))
                    vs,rows=repair_batch(backend,current,ids,[words(c['root_seed'],'state_repair',i,step)],j,f'{i}/{step}',{'instance':i,'step':step})
                    if rows[0]['cost_after']<=rows[0]['cost_before']: current=payload(vs[0])
                    if step in (50,200):
                        s={'instance':i,'step':step,'state':copy.deepcopy(current),'cost':cost(current),'content_sha256':h}
                        snaps.append(s); write(d/f'snapshot-{i}-{step}.json',sealed(s))
            write(d/'catalog.json',sealed({'instances':catalog,'history_coverage':data['history_coverage']}))
            write(d/'data.json',sealed({'states':snaps,'environment':backend.environment,'fixture':False}))
        finally: j.close()

def candidates(run):
    c=manifest(run)['config']; data=artifact(run,'generate-test-states')
    tables=[pool(s['state'],c['root_seed'],s['instance'],s['step']) for s in data['states']]
    with stage(run,'build-test-candidates') as d: write(d/'data.json',sealed({'tables':tables}))

def score_table(t,plan,c):
    from point_select.models import predict as old_predict,tie_indices
    scores={g:predict(plan['models'][g],[vector(s,g) for s in t['slots']]) for g in GROUPS}
    for name,model in plan['old_models'].items(): scores[name]=old_predict(model,[s['X'] for s in t['slots']]).tolist()
    scores['native_expectation']=[0.]*16
    for name in ('contact_edges','release_concentration'): scores[name]=[s['rules'][name] for s in t['slots']]
    if t['fallback']:
        for g in ('Tdiff','Tready','Tshuffle'): scores[g]=scores[plan['baseline']].copy()
    ties={n:tie_indices(sc).tolist() for n,sc in scores.items()}
    chosen={n:int(rng(c['root_seed'],'tie',t['instance'],t['step'],n).choice(ii)) for n,ii in ties.items()}
    return scores,ties,chosen

def predictions(run):
    c=manifest(run)['config']; plan=artifact(run,'freeze-plan'); ft=artifact(run,'build-test-features')['tables']
    refs=artifact(run,'generate-test-references')['references']; states={(s['instance'],s['step']):s['state'] for s in artifact(run,'generate-test-states')['states']}
    scores={}; ties={}; chosen={}; alternate={}; raw_new={}
    for t in ft:
        key=f"{t['instance']}/{t['step']}"; scores[key],ties[key],chosen[key]=score_table(t,plan,c)
        raw_new[key]={g:predict(plan['models'][g],[vector(s,g) for s in t['slots']]) for g in GROUPS}
        alternate[key]={}
        for index,o in enumerate(refs[str(t['instance'])]['outcomes']):
            alt=build_table(states[t['instance'],t['step']],t,o['routes'] if o['status']=='ok' else None,c['root_seed'])
            alt_scores,_,_=score_table(alt,plan,c); alternate[key][str(index)]={g:alt_scores[g] for g in GROUPS}
    with stage(run,'freeze-test-predictions') as d:
        write(d/'data.json',sealed({'scores':scores,'ties':ties,'chosen':chosen,'alternate_scores':alternate,'raw_new_scores':raw_new,
              'labels_read':False,'plan_digest':plan['digest'],'feature_digest':artifact(run,'build-test-features')['digest']}))

def collect(run,cap):
    c=manifest(run)['config']; ft=artifact(run,'build-test-features')['tables']; preds=artifact(run,'freeze-test-predictions')
    need=sum(len(t['unique']) for t in ft)*64; cap_check('collect-test-b',cap,need)
    states={(s['instance'],s['step']):s['state'] for s in artifact(run,'generate-test-states')['states']}
    from .backend import Backend,repair_batch
    backend=Backend(c)
    if backend.environment!=artifact(run,'calibrate')['environment']: raise ValueError('Calibration runtime identity drift')
    with stage(run,'collect-test-b',{'cap':cap,'predictions_digest':preds['digest']}) as d:
        j=Journal(d,cap)
        try:
            with (d/'labels.jsonl').open('x',encoding='utf-8',newline='\n') as f:
                for t in ft:
                    for cid,ids in sorted(t['unique'].items()):
                        for offset in range(0,64,16):
                            seeds=[words(c['root_seed'],'B',t['instance'],t['step'],cid,r) for r in range(offset,offset+16)]
                            _,rows=repair_batch(backend,states[t['instance'],t['step']],ids,seeds,j,f"{t['instance']}/{t['step']}/{cid}/{offset}",
                              {'instance':t['instance'],'step':t['step'],'candidate':cid,'group':'B','repeat_start':offset})
                            for row in rows: f.write(canonical(row)+'\n')
                            f.flush(); __import__('os').fsync(f.fileno())
            write(d/'data.json',sealed({'repeats':64,'charged':j.used,'environment':backend.environment,'fixture':False}))
        finally: j.close()
