from collections import Counter
import itertools
import math
import numpy as np
from point_select.data import routes, partial_solution, validate_deletion, cost
from point_select.features import cache, fast, rules
from .core import digest, rng

DIFF_NAMES=('released_nonref_count','released_nonref_length','released_shared_count','released_shared_length','missing_incident','coroute_xor')
READY_NAMES=('visible_fraction','merge_fraction','capacity_fraction','capacity_length','affected_fraction','locked_fraction')
CURRENT_NAMES=('contact_length','contact_fraction','kept_length','incomplete_count','complete_count','released_count','open_count','fragment_mean','fragment_max','fragment_std','load_mean','pair_capacity')

def edges(rr):
    return Counter(tuple(sorted((a,b))) for r in rr for a,b in zip([0,*r],[*r,0]))

def fragments(state,deleted):
    pp=partial_solution(state,deleted); out=[]; mapping={}
    for t in pp:
        ids=[e[0] for e in t]
        if all(i==0 for i in ids): continue
        open_=ids[0]!=0 or ids[-1]!=0
        end=set([ids[0],ids[-1]])-{0} if open_ else set()
        f={'ids':ids,'ends':end,'load':sum(state['demand'][i] for i in ids),
           'open':open_,'anchored':0 in ids,'n':sum(i!=0 for i in ids)}
        j=len(out); out.append(f)
        for i in ids:
            if i: mapping[i]=j
    return out,mapping

def eligible(a,b,ff,mm,capacity):
    def visible(i): return i==0 or i in ff[mm[i]]['ends']
    vis=visible(a) and visible(b)
    if not vis: return False,False,False
    if a==0 or b==0:
        f=ff[mm[b if a==0 else a]]
        return True,True,f['load']<=capacity
    ia,ib=mm[a],mm[b]
    different=ia!=ib
    return True,different,different and ff[ia]['load']+ff[ib]['load']<=capacity

def extra(state,deleted,reference=None):
    validate_deletion(deleted); d=set(deleted)
    rr=[list(r) for r in sorted(min(tuple(r),tuple(reversed(r))) for r in routes(state))]
    ee=edges(rr)
    xy=np.asarray(state['original_locations'],float)
    length=lambda e: float(np.linalg.norm(xy[e[0]]-xy[e[1]]))
    total=math.fsum(length(e)*n for e,n in sorted(ee.items()))
    ff,mm=fragments(state,deleted); inc=[f for f in ff if f['open']]
    touched=Counter({e:n for e,n in ee.items() if d.intersection(e)})
    lens=[f['n'] for f in inc]; loads=[f['load'] for f in inc]
    pairs=list(itertools.combinations(inc,2)); cap=state['capacity']
    current=[sum(length(e)*n for e,n in touched.items())/total,sum(touched.values())/sum(ee.values()),
             sum(length(e)*n for e,n in (ee-touched).items())/total,len(inc)/101,
             sum(not f['open'] for f in ff)/101,sum(f['ids']==[i] for f in ff for i in d)/101,
             sum(len(f['ends']) for f in inc)/101,
             np.mean(lens)/100 if lens else 0,max(lens,default=0)/100,np.std(lens)/100 if lens else 0,
             np.mean(loads)/cap if loads else 0,sum(a['load']+b['load']<=cap for a,b in pairs)/len(pairs) if pairs else 0]
    if reference is None: return list(map(float,current)),[0.]*6,[0.]*6,{'missing':True}
    reference=[list(r) for r in sorted(min(tuple(r),tuple(reversed(r))) for r in reference)]
    re=edges(reference); missing=re-ee; non=(ee-re)&touched; shared=(ee&re)&touched
    def route_map(rs): return {i:j for j,r in enumerate(rs) for i in r}
    cm,rm=route_map(rr),route_map(reference)
    rel=[(a,b) for a,b in itertools.combinations(range(1,101),2) if a in d or b in d]
    diff=[sum(non.values())/sum(ee.values()),sum(length(e)*n for e,n in non.items())/total,
          sum(shared.values())/sum(ee.values()),sum(length(e)*n for e,n in shared.items())/total,
          sum(n for e,n in missing.items() if d.intersection(e))/sum(re.values()),
          sum((cm[a]==cm[b])!=(rm[a]==rm[b]) for a,b in rel)/len(rel)]
    den=max(1,sum(missing.values())); visible=merge=ready=locked=0; ready_length=0.; affected=set()
    for (a,b),n in missing.items():
        v,m,c=eligible(a,b,ff,mm,cap)
        visible+=n*v; merge+=n*m; ready+=n*c; locked+=n*(not v)
        if c:
            ready_length+=n*length((a,b))
            affected.update(mm[i] for i in (a,b) if i and ff[mm[i]]['open'])
    reference_cost=sum(length(e)*n for e,n in re.items())
    ready_values=[visible/den,merge/den,ready/den,ready_length/reference_cost,len(affected)/max(1,len(inc)),locked/den]
    return list(map(float,current)),list(map(float,diff)),list(map(float,ready_values)),{'missing':False}

def build_table(state,table,reference,root):
    cc=cache(state); slots=[]
    for s in table['slots']:
        deleted=table['unique'][s['candidate']]
        gc,gd,gr,_=extra(state,deleted,reference)
        slots.append({**s,'X':fast(cc,deleted,s['center']),'current':gc,'diff':gd,'ready':gr,'rules':rules(cc,deleted)})
    perm=rng(root,'shuffle',table['instance'],table['step']).permutation(16).tolist()
    for j,s in enumerate(slots): s['shuffle']=slots[perm[j]]['diff']+slots[perm[j]]['ready']
    from .reference import route_cost
    fallback=reference is None or edges(reference)==edges(routes(state)) or route_cost(state,reference)>=cost(state)
    return {**table,'slots':slots,'shuffle_permutation':perm,'fallback':fallback,
            'reference_gap':None if reference is None else (cost(state)-route_cost(state,reference))/cost(state),
            'structure_distance':None if reference is None else sum((edges(routes(state))-edges(reference)).values())/sum(edges(routes(state)).values())}

def vector(s,group):
    return s['X']+({'B0':[],'Bcap':s['current'],'Tdiff':s['diff'],
                   'Tready':s['diff']+s['ready'],'Tshuffle':s['shuffle']}[group])

def pool(state,root,i,step):
    from point_select.data import point
    sd=digest(state); slots=[]; unique={}
    for j in range(16):
        ids,center=point(state,rng(root,'candidate',i,step,j)); cid=digest([sd,ids]); unique[cid]=ids
        slots.append({'slot':j,'candidate':cid,'center':center,'seed_words':__import__('teacher_select.core',fromlist=['words']).words(root,'candidate',i,step,j)})
    return {'instance':i,'step':step,'state_digest':sd,'slots':slots,'unique':unique}
