import math
from statistics import NormalDist
import numpy as np
from point_select.statistics import t_quantile
from .core import rng,artifact,manifest,receipt,read,verify,sha,stream,DEPS,LIMITS,VERSION
from .models import metric

def interval(values,root,tag,m=1):
    x=np.asarray(values,float)
    if len(x)<2 or not np.isfinite(x).all(): raise ValueError('Independent complete instances required')
    mean=float(x.mean()); sd=float(x.std(ddof=1)); se=sd/math.sqrt(len(x)); alpha=.05/m
    q=t_quantile(1-alpha,len(x)-1); r=rng(root,'bootstrap',tag)
    boot=np.array([x[r.randint(len(x),size=len(x))].mean() for _ in range(10000)])
    sdboot=np.array([x[r.randint(len(x),size=len(x))].std(ddof=1) for _ in range(10000)])
    ordered=np.sort(x)
    return {'n':len(x),'mean_pp':mean,'sd_pp':sd,'median_pp':float(np.median(x)),
            'wins':int((x>0).sum()),'losses':int((x<0).sum()),'ties':int((x==0).sum()),
            'alpha':alpha,'t_lower_pp':mean-q*se,'t_upper_pp':mean+q*se,
            'bootstrap95_pp':np.quantile(boot,[.025,.975]).tolist(),
            'planning_sd_pp':max(sd,float(np.quantile(sdboot,.8))),
            'drop_largest_mean_pp':float(ordered[:-1].mean()),
            'drop_two_largest_mean_pp':float(ordered[:-2].mean()) if len(x)>2 else None,
            'true_gain_gt_delta_proven':mean-q*se>.01}

def planning(values,root,m):
    it=interval(values,root,'power',m); sigma=it['planning_sd_pp']; nd=NormalDist(); z=nd.inv_cdf(1-.05/m)
    if sigma==0: powers={str(d):1. for d in (.01,.02,.03)}; detectable=0.; required={str(d):2 for d in (.01,.02,.03)}
    else:
        powers={str(d):nd.cdf(d*math.sqrt(128)/sigma-z) for d in (.01,.02,.03)}
        detectable=(z+nd.inv_cdf(.8))*sigma/math.sqrt(128)
        required={str(d):math.ceil(((z+nd.inv_cdf(.8))*sigma/d)**2) for d in (.01,.02,.03)}
    return {'n':128,'detectable_effect_80pct_pp':detectable,'approx_power':powers,'required_n_80pct':required,
            'sigma_pp':sigma,'caveat':'Development V reused historically; selection/variance bias and heavy tails; no automatic expansion'}

def decide(comparisons,mechanism,teacher,valid=True,fixture=False):
    if not valid or fixture: return 'INVALID'
    passing=lambda v: (v['mean_pp']>=.01 and v['t_lower_pp']>0 and v['bootstrap95_pp'][0]>0 and
                       v['drop_largest_mean_pp']>0 and v['drop_two_largest_mean_pp'] is not None and v['drop_two_largest_mean_pp']>0)
    if all(passing(v) for v in comparisons.values()):
        proven=teacher=='Tready' and mechanism['t_lower_pp']>0 and mechanism['bootstrap95_pp'][0]>0 and mechanism['drop_two_largest_mean_pp']>0
        return 'GO_REFERENCE' if proven else 'GO_REFERENCE_BUT_MECHANISM_UNPROVEN'
    if any(v['t_upper_pp']<.01 and v['bootstrap95_pp'][1]<.01 for v in comparisons.values()): return 'STOP_PRACTICAL'
    return 'INCONCLUSIVE'

def audit(run,final=False):
    m=manifest(run); found={}; costs={}; pending=[]; stage_wall={}
    for name in DEPS:
        d=__import__('pathlib').Path(run)/'stages'/name
        if not d.exists(): pending.append(name); continue
        r=receipt(run,name); found[name]=r['digest']; started=verify(read(d/'started.json'))
        if started['manifest']!=m['digest'] or r['time']<started['time']: raise ValueError('Stage lineage/time')
        stage_wall[name]=r['time']-started['time']
        if started['dependencies']!={n:receipt(run,n)['digest'] for n in DEPS[name]}: raise ValueError('Dependencies changed')
        if any(receipt(run,n)['time']>started['time'] for n in DEPS[name]): raise ValueError('Freeze/stage order')
        if (d/'calls.jsonl').exists():
            calls={}; terminal={}
            for row in stream(d/'calls.jsonl'):
                if row['status']=='started':
                    if row['key'] in calls: raise ValueError('Duplicate invocation')
                    calls[row['key']]=row['charge']
                else:
                    if row['key'] not in calls or row['key'] in terminal: raise ValueError('Invocation state transition')
                    terminal[row['key']]=row['status']
            charged=sum(calls.values()); costs[name]=charged
            if name not in LIMITS or charged>LIMITS[name]: raise ValueError('Budget exceeded')
            # Feasible-reference absence is allowed; failed/uncertain Repair or fits is not.
            if len(terminal)!=len(calls) or (name not in ('generate-dev-references','generate-test-references') and any(s!='ok' for s in terminal.values())):
                raise ValueError('Incomplete/uncertain calls')
    if sum(costs.get(n,0) for n in ('calibrate','generate-test-states','collect-test-b'))>288768: raise ValueError('Total Repair budget')
    if sum(costs.get(n,0) for n in ('fit-oof','assess-development'))>125: raise ValueError('Total fit budget')
    if sum(costs.get(n,0) for n in ('generate-dev-references','generate-test-references'))>2880: raise ValueError('Solver budget')
    if final and pending: raise ValueError('Incomplete experiment: '+','.join(pending))
    return {'valid_engineering_audit':True,'complete':not pending,'pending':pending,'stage_receipts':found,'charged_budgets':costs,'stage_wall_seconds':stage_wall,
            'fixture':m['fixture'],'evidence':'Privileged-reference information-value experiment, not a deployable student'}

def evaluate(tables,labels,scores):
    rows=[]; flat={}
    for t in tables:
        key=f"{t['instance']}/{t['step']}"
        for s in t['slots']:
            rows.append({'instance':t['instance'],'step':t['step'],'target':labels[t['instance'],t['step'],s['candidate']]['target']})
        for name,values in scores[key].items(): flat.setdefault(name,[]).extend(values)
    return {n:{i:100*v for i,v in metric(rows,s).items()} for n,s in flat.items()}

def report(run):
    try:
        aud=audit(run,True); m=manifest(run)
        if m['fixture']: raise ValueError('Synthetic fixture cannot produce production confirmation conclusions')
        c=m['config']; plan=artifact(run,'freeze-plan')
        from .source import load_labels
        ft=artifact(run,'build-test-features')['tables']; predictions=artifact(run,'freeze-test-predictions')
        state_data=artifact(run,'generate-test-states')
        states={(s['instance'],s['step']):s['state'] for s in state_data['states']}
        labs=load_labels(run,'collect-test-b',ft,64,c['root_seed'],False,states)
        from point_select.models import tie_indices
        if predictions['plan_digest']!=plan['digest'] or predictions['feature_digest']!=artifact(run,'build-test-features')['digest'] or predictions['labels_read']:
            raise ValueError('Prediction freeze linkage')
        for key,sc in predictions['scores'].items():
            for name,values in sc.items():
                ii=tie_indices(values).tolist()
                if predictions['ties'][key][name]!=ii or predictions['chosen'][key][name] not in ii:
                    raise ValueError('Frozen ties/choices inconsistent')
        ev=evaluate(ft,labs,predictions['scores']); teacher=plan['teacher']; baseline=plan['baseline']
        if set(ev[teacher])!={str(i) for i in plan['test_ids']}: raise ValueError('128 instance coverage')
        contrasts={f'{teacher}-{b}':[ev[teacher][i]-ev[b][i] for i in ev[teacher]] for b in dict.fromkeys([baseline,'Bcap'])}
        ci={n:interval(v,c['root_seed'],n,len(contrasts)) for n,v in contrasts.items()}
        mech=interval([ev['Tready'][i]-ev['Tdiff'][i] for i in ev[teacher]],c['root_seed'],'mechanism')
        decision=decide(ci,mech,teacher,True,m['fixture'])
        auxiliary={}
        for group in ('B0','Bcap','Tdiff','Tready','Tshuffle'):
            target=[]; pred=[]
            for t in ft:
                key=f"{t['instance']}/{t['step']}"
                target.extend(labs[t['instance'],t['step'],s['candidate']]['target'] for s in t['slots'])
                pred.extend(predictions['raw_new_scores'][key][group])
            target=np.asarray(target); pred=np.asarray(pred)
            auxiliary[group]={'mse_on_B':float(np.mean((pred-target)**2)),
                              'correlation_on_B':float(np.corrcoef(pred,target)[0,1]) if np.std(pred)>0 and np.std(target)>0 else None,
                              'scope':'raw new model outputs before fallback; slot-weighted; auxiliary only'}
        precision={}
        for name,v in ci.items():
            sigma=v['planning_sd_pp']; z=NormalDist().inv_cdf(1-.05/len(ci))+NormalDist().inv_cdf(.8)
            n=max(2,math.ceil((z*sigma/.01)**2))
            additional=max(0,n-128)
            precision[name]={'assumed_effect_pp':.01,'approx_total_independent_instances_80pct':n,
                             'additional_instances':additional,'additional_repair_upper':2248*additional,
                             'additional_reference_cpu_seconds':15*additional,'authorized':False,
                             'caveat':'Post-test precision suggestion only; requires separate review and new protocol; heavy tails/variance uncertainty'}
        # Frozen stage/gap/structure strata are descriptive, never primary success rules.
        strata={}
        for tag,predicate in {'step50':lambda t:t['step']==50,'step200':lambda t:t['step']==200,
                             'fallback':lambda t:t['fallback'],'reference_better':lambda t:not t['fallback'],
                             'gap_le_5pct':lambda t:t['reference_gap'] is not None and t['reference_gap']<=.05,
                             'gap_gt_5pct':lambda t:t['reference_gap'] is not None and t['reference_gap']>.05,
                             'structure_le_half':lambda t:t['structure_distance'] is not None and t['structure_distance']<=.5,
                             'structure_gt_half':lambda t:t['structure_distance'] is not None and t['structure_distance']>.5}.items():
            per={}
            from point_select.models import tie_indices
            for t in ft:
                if not predicate(t): continue
                sc=predictions['scores'][f"{t['instance']}/{t['step']}"]
                z=np.array([labs[t['instance'],t['step'],s['candidate']]['target'] for s in t['slots']])
                diff=100*(z[tie_indices(sc[teacher])].mean()-z[tie_indices(sc[baseline])].mean())
                per.setdefault(t['instance'],[]).append(float(diff))
            vv=[float(np.mean(x)) for x in per.values()]
            strata[tag]={'instances':len(vv),'mean_pp':float(np.mean(vv)) if vv else None,'exploratory':True}
        refs=artifact(run,'generate-test-references')['references']
        ref_quality=[]; solver_runtime=[]
        from .features import edges
        for i,r in refs.items():
            feasible=[o for o in r['outcomes'] if o['status']=='ok']
            solver_runtime.extend({'instance':i,'seed':o['seed'],'status':o['status'],'wall_seconds':o['wall_seconds'],
                                   'actual_child_cpu_seconds':o['actual_child_cpu_seconds'],'time_limit_cpu':o['time_limit_cpu']} for o in r['outcomes'])
            ref_quality.append({'instance':i,'costs':[o.get('cost') for o in r['outcomes']],
                                'pairwise_structure_disagreement':[sum((edges(a['routes'])-edges(b['routes'])).values())/sum(edges(a['routes']).values()) for j,a in enumerate(feasible) for b in feasible[j+1:]]})
        alt=predictions.get('alternate_scores',{})
        alt_ev={}
        for index in range(3):
            sc={key:values[str(index)] for key,values in alt.items()}
            if sc: alt_ev[str(index)]=evaluate(ft,labs,sc)
        no_reference=sum(r['primary'] is None for r in refs.values())/len(refs)
        fallback_fraction=sum(t['fallback'] for t in ft)/len(ft)
        accuracy_better=auxiliary[teacher]['mse_on_B']<auxiliary['Bcap']['mse_on_B']
        if no_reference==1 or fallback_fraction==1:
            failure_layer='REFERENCE_AVAILABILITY_OR_NO_POSITIVE_REFERENCE_HEADROOM'
        elif decision.startswith('GO_REFERENCE'):
            failure_layer='NO_REFERENCE_SELECTION_FAILURE' if decision=='GO_REFERENCE' else 'ACTIONABILITY_INCREMENT_UNPROVEN'
        elif accuracy_better:
            failure_layer='PREDICTION_IMPROVEMENT_NOT_CONVERTED_TO_CONFIRMED_SELECTION_GAIN'
        else:
            failure_layer='REFERENCE_INFORMATION_INCREMENT_NOT_ESTABLISHED_UNDER_THIS_MODEL_AND_BUDGET'
        return {'decision':decision,'valid':True,'audit':aud,'teacher':teacher,'baseline':baseline,
                'primary_comparisons':ci,'mechanism':mech,'method_mean_pp':{n:float(np.mean(list(v.values()))) for n,v in ev.items()},
                'auxiliary':auxiliary,'precision_review':precision,'solver_runtime':solver_runtime,
                'historical_costs':artifact(run,'import-development')['audit']['historical_costs'],
                'per_instance_pp':ev,'sensitivity':strata,'reference_quality':ref_quality,
                'alternate_reference_per_instance_pp':alt_ev,'fallback_fraction':fallback_fraction,
                'failure_diagnosis':{'layer':failure_layer,'no_feasible_reference_instance_fraction':no_reference,
                                     'reference_not_better_or_missing_state_fraction':fallback_fraction,
                                     'causal_attribution_proven':False,'caveat':'Layer is a diagnostic interpretation; wide intervals cannot prove absence of information'},
                'answers':{'worth_continuing':decision.startswith('GO_REFERENCE'),
                           'actionability_proven':decision=='GO_REFERENCE','next_step':{'GO_REFERENCE':'Separate student migration experiment',
                           'GO_REFERENCE_BUT_MECHANISM_UNPROVEN':'Student transfer; mechanism novelty remains unproven',
                           'STOP_PRACTICAL':'Stop this teacher version','INCONCLUSIVE':'Separate precision review; no larger model authorization','INVALID':'Repair engineering/identity defects'}[decision],
                           'prediction_only':{'accuracy_better_than_Bcap':accuracy_better,
                                              'selection_success':decision.startswith('GO_REFERENCE'),
                                              'interpretation':'If accuracy improves without GO selection evidence, prediction accuracy alone does not support continuation'}},
                'limitations':['Not a deployable student','Not full-search benefit','No single-customer causal value','No universal rejection of heuristic teachers','t/bootstrap approximate under heavy tails']}
    except (ValueError,FileNotFoundError,KeyError) as e:
        return {'decision':'INVALID','valid':False,'reason':str(e),'answers':{'worth_continuing':None,'actionability_proven':None,'next_step':'Complete/repair engineering audit; no scientific conclusion'}}

def markdown(r):
    a=r.get('answers',{})
    return ('# 启发式参考解诊断\n\n'+f"当前实验有效吗：{r['valid']}\n\n决策：**{r['decision']}**\n\n"+
            '这是特权教师信息价值实验，不能证明学生可部署或完整搜索有效。\n\n'+
            f"参考解值得继续吗：{a.get('worth_continuing')}\n\n可操作性优于纯差异吗：{a.get('actionability_proven')}（机制探索性，不是额外确认性检验）\n\n"+
            f"是否只有预测更准：{a.get('prediction_only')}\n\n下一步：{a.get('next_step')}\n\n"+
            '新增收益和机制增量（单位pp；主比较为冻结最强无参考基线及Bcap）：\n\n```json\n'+
            __import__('json').dumps({k:r[k] for k in ('primary_comparisons','mechanism','method_mean_pp','reason','failure_diagnosis','precision_review') if k in r},indent=2,ensure_ascii=False)+
            '\n```\n\n不能成立的结论：学生迁移有效、完整NLNS改善、单客户因果价值，以及所有启发式教师均有效/无效。\n')
