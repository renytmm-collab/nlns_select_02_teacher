"""Independent boundary around byte-identical legacy actor/Repair/VRP core."""
import copy
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .core import ROOT,CHECKPOINT,sha,words,rng,digest,write,canonical
from point_select.data import partial_solution,cost

def payload(v):
    return {'locations':v.locations.tolist(),'original_locations':v.original_locations.tolist(),
            'demand':v.demand.tolist(),'capacity':int(v.capacity),'solution':v.get_solution_copy()}

def restore(state):
    from vrp.vrp_problem import VRPInstance
    v=VRPInstance(100,np.asarray(state['locations']),np.asarray(state['original_locations']),np.asarray(state['demand'],int),state['capacity'])
    v.solution=copy.deepcopy(state['solution']); return v

def destroy(v,ids):
    expected=partial_solution(payload(v),ids); v.destroy(sorted(ids))
    if v.solution!=expected: raise ValueError('Destroy serialization mismatch')
    return digest(expected)

class Backend:
    def __init__(self,c):
        if np.__version__!='1.26.4': raise ValueError('Real backend requires numpy==1.26.4; no semantic fallback')
        checkpoint=Path(c['checkpoint']).resolve()
        if sha(checkpoint)!=CHECKPOINT: raise ValueError('Checkpoint identity')
        for r in __import__('teacher_select.core',fromlist=['read']).read(ROOT/'vendor/PROVENANCE.json'):
            if sha(ROOT/'vendor'/r['path'])!=r['sha256']: raise ValueError('Vendor identity drift')
        os.environ['CUDA_CACHE_DISABLE']='1'; os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        from actor import VrpActorModel
        self.torch=torch; self.c=c
        self.config=SimpleNamespace(device=torch.device(c['device']),split_delivery=False,round_distances=False)
        if self.config.device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA missing; no CPU fallback')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'): raise ValueError('CUBLAS deterministic setting')
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        data=torch.load(checkpoint,map_location=self.config.device,weights_only=False)
        if data.get('destroy_operation')!='P' or data.get('p_destruction')!=.2 or data.get('ots_mode')!='random-seq': raise ValueError('Checkpoint metadata')
        self.actor=VrpActorModel(self.config.device,128).to(self.config.device)
        self.actor.load_state_dict(data['parameters'],strict=True); self.actor.eval()
        self.environment={'numpy':np.__version__,'torch':torch.__version__,'device':str(self.config.device),
                          'cuda':torch.version.cuda,'checkpoint':CHECKPOINT,'fixture':False,'threads':1}
        if self.config.device.type=='cuda':
            self.environment.update(gpu=torch.cuda.get_device_name(self.config.device),capability=list(torch.cuda.get_device_capability(self.config.device)),arches=torch.cuda.get_arch_list())
    def sync(self):
        if self.config.device.type=='cuda': self.torch.cuda.synchronize(self.config.device)
    def repair(self,vs,seeds):
        import repair
        with self.torch.no_grad(): repair.repair(vs,self.actor,self.config,rngs=[np.random.RandomState(s) for s in seeds])
        for v in vs: v.verify_solution(self.config)
        self.sync()
    def generate(self,i):
        from vrp.data_utils import create_dataset
        old=np.random.get_state()
        try:
            np.random.seed(words(self.c['root_seed'],'instance',i))
            return create_dataset(1,SimpleNamespace(instance_blueprint='ALTR_100'),create_solution=True)[0]
        finally: np.random.set_state(old)

def repair_batch(backend,state,ids,seeds,journal,key,meta):
    journal.begin(key,len(seeds),{**meta,'seeds':seeds,'environment':backend.environment})
    tick=time.perf_counter()
    try:
        vs=[restore(state) for _ in seeds]
        for v in vs: destroy(v,ids)
        natural=[v.get_max_nb_input_points() for v in vs]
        if len(set(natural))!=1: raise ValueError('Heterogeneous batch blocked')
        before=cost(state); backend.sync(); backend.repair(vs,seeds)
        after=[float(v.get_costs(False)) for v in vs]
        if not np.isfinite(after).all() or min(after)<=0: raise ValueError('Invalid repaired cost')
        rows=[]
        for j,(seed,a) in enumerate(zip(seeds,after)):
            y=(before-a)/before
            rows.append({**meta,'repeat':meta.get('repeat_start',0)+j,'seed_words':seed,'batch':key,
                         'batch_size':len(seeds),'status':'ok','repair_invoked':True,'fixture':False,
                         'input_points':natural[j],'padded_input_points':max(natural),'cost_before':before,
                         'cost_after':a,'y':y,'z':max(y,0),'wall_seconds':time.perf_counter()-tick})
        journal.event({'key':key,'status':'ok','completed':len(rows),'wall_seconds':time.perf_counter()-tick})
        return vs,rows
    except BaseException as e:
        journal.event({'key':key,'status':'uncertain','charged':len(seeds),'error_type':type(e).__name__,'wall_seconds':time.perf_counter()-tick})
        raise
