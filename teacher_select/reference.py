"""HGS adapter: no automatic installation, compilation, retries or seed extension."""
import math
import os
import re
import subprocess
import time
from pathlib import Path
import numpy as np
from .core import sha, digest

def route_cost(state,rr):
    xy=np.asarray(state['original_locations'],float)
    return float(math.fsum(np.linalg.norm(xy[a]-xy[b]) for r in rr for a,b in zip([0,*r],[*r,0])))

def validate(state,rr):
    if len(state['demand'])!=101 or state['capacity']!=50: raise ValueError('ALTR identity')
    if sorted(i for r in rr for i in r)!=list(range(1,101)) or any(not r for r in rr):
        raise ValueError('Service duplication, depot mapping or split delivery')
    if any(sum(state['demand'][i] for i in r)>50 for r in rr): raise ValueError('Capacity infeasible')
    c=route_cost(state,rr)
    if not math.isfinite(c) or c<=0: raise ValueError('Invalid reference cost')
    return c

def encode(state):
    xy=state['original_locations']; q=state['demand']
    # CVRPLIB node 1 is depot, output Route customers use their internal 1..100 IDs.
    text='NAME : ALTR\nTYPE : CVRP\nDIMENSION : 101\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : 50\nNODE_COORD_SECTION\n'
    text+=''.join(f'{i+1} {x:.17g} {y:.17g}\n' for i,(x,y) in enumerate(xy))
    text+='DEMAND_SECTION\n'+''.join(f'{i+1} {int(v)}\n' for i,v in enumerate(q))
    text+='DEPOT_SECTION\n1\n-1\nEOF\n'
    # Input precision round-trip before passing to solver.
    lines=text.split('NODE_COORD_SECTION\n')[1].split('DEMAND_SECTION')[0].splitlines()
    parsed=[[float(x) for x in l.split()[1:]] for l in lines]
    if parsed!=xy: raise ValueError('Coordinate precision loss')
    if any(int(v)!=v for v in q): raise ValueError('Demand precision loss')
    return text

def parse(text,state):
    rr=[]; advertised=None
    for line in text.splitlines():
        if line.startswith('Route #'):
            rr.append([int(x) for x in line.split(':',1)[1].split()])
        elif line.startswith('Cost '): advertised=line.split()[1]
        elif line.strip(): raise ValueError('Unknown HGS solution line')
    c=validate(state,rr)
    if advertised is None: raise ValueError('HGS cost missing')
    # HGS commonly prints rounded costs; allow exactly half the last printed unit.
    from decimal import Decimal
    v=Decimal(advertised); tolerance=float(Decimal(10)**v.as_tuple().exponent)/2+1e-10
    if tolerance>1e-3 or abs(float(v)-c)>tolerance: raise ValueError('HGS cost precision/inconsistency')
    return {'routes':rr,'cost':c,'advertised_cost':float(v),'printed_cost_tolerance':tolerance}

def select(outcomes):
    feasible=[o for o in outcomes if o['status']=='ok']
    return min(feasible,key=lambda o:(o['cost'],o['seed_index'])) if feasible else None

def solve(state,d,c,journal,i):
    exe=Path(c['solver_executable']).resolve()
    if not exe.is_file() or sha(exe)!=c['solver_sha256'] or not c['solver_version']:
        raise ValueError('Pinned HGS executable/version required')
    d.mkdir(); inp=d/'instance.vrp'; inp.write_text(encode(state),encoding='utf-8')
    outcomes=[]; env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for index,seed in enumerate(c['solver_seeds']):
        out=d/f'seed-{seed}.sol'; log=d/f'seed-{seed}.log'
        cmd=[str(exe),str(inp),str(out),'-t','5','-seed',str(seed),'-round','0','-veh','100','-log','0']
        meta={'instance':i,'seed':seed,'seed_index':index,'command':cmd,'solver_sha256':c['solver_sha256'],
              'version':c['solver_version'],'threads':1,'time_limit_cpu':5,'input_sha256':sha(inp),'instance_digest':digest(state)}
        key=f'reference/{i}/{seed}'; journal.begin(key,5,meta); tick=time.perf_counter(); cpu_start=os.times()
        try:
            with log.open('x',encoding='utf-8') as f:
                p=subprocess.run(cmd,cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=15,check=False)
            result={**meta,'exit_code':p.returncode,'wall_seconds':time.perf_counter()-tick}
            if p.returncode==0 and out.exists(): result.update(parse(out.read_text(),state),status='ok')
            else: result.update(status='failed',reason='No feasible output or nonzero exit')
        except subprocess.TimeoutExpired:
            result={**meta,'status':'uncertain','reason':'Wall watchdog killed solver','wall_seconds':time.perf_counter()-tick}
        except ValueError as e:
            result={**meta,'status':'invalid','reason':str(e),'wall_seconds':time.perf_counter()-tick}
        cpu_end=os.times(); result['actual_child_cpu_seconds']=(cpu_end.children_user+cpu_end.children_system-cpu_start.children_user-cpu_start.children_system) if os.name!='nt' else None
        # CPU limit is solver-native, with OS measurement independently disclosed.
        result['input_sha256_after']=sha(inp)
        result['output_sha256']=sha(out) if out.is_file() else None
        result['log_sha256']=sha(log)
        if result['input_sha256_after']!=meta['input_sha256']: raise ValueError('Solver mutated frozen input')
        journal.event({'key':key,**result}); outcomes.append(result)
    if any(o['status']=='invalid' for o in outcomes): raise ValueError('Reference identity/precision invalid')
    best=select(outcomes)
    return {'outcomes':outcomes,'primary':best,'no_feasible_reference':best is None}
