import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'vendor'))
VERSION = 'point-reference-diagnostic-v1'
CHECKPOINT = '777178c30819f73b387674c29a2430cc06b0fda876792d5e08137d5ea3f94cf8'
GROUPS = ('B0', 'Bcap', 'Tdiff', 'Tready', 'Tshuffle')
LIMITS = {'generate-dev-references': 960, 'generate-test-references': 1920,
          'generate-test-states': 25600, 'collect-test-b': 262144, 'calibrate': 1024,
          'fit-oof': 120, 'assess-development': 5}
DEPS = {
 'import-development': [], 'generate-dev-references': ['import-development'],
 'build-dev-features': ['generate-dev-references'], 'fit-oof': ['build-dev-features'],
 'assess-development': ['fit-oof'], 'calibrate': ['import-development'],
 'freeze-plan': ['assess-development', 'calibrate'],
 'generate-test-states': ['freeze-plan'], 'generate-test-references': ['generate-test-states'],
 'build-test-candidates': ['generate-test-references'], 'build-test-features': ['build-test-candidates'],
 'freeze-test-predictions': ['build-test-features'], 'collect-test-b': ['freeze-test-predictions']}

def canonical(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(v):
    return hashlib.sha256(canonical(v).encode('utf-8')).hexdigest()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sealed(v):
    return {**v, 'digest': digest(v)}

def verify(v):
    if v.get('digest') != digest({k: x for k, x in v.items() if k != 'digest'}):
        raise ValueError('Sealed digest mismatch')
    return v

def inside(p):
    p = Path(p).resolve()
    if p == ROOT or not p.is_relative_to(ROOT): raise ValueError('Output outside new project')
    return p

def write(p, v):
    p = inside(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(canonical(v)+'\n'); f.flush(); os.fsync(f.fileno())

def stream(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def words(root, name, *parts):
    raw = hashlib.sha256(canonical([root, VERSION, name, *parts]).encode()).digest()[:16]
    return [int.from_bytes(raw[j:j+4], 'little') for j in range(0,16,4)]

def rng(root, name, *parts):
    import numpy as np
    return np.random.RandomState(words(root, name, *parts))

def code_identity():
    paths = [ROOT/'teacher_cli.py', *sorted((ROOT/'teacher_select').glob('*.py')),
             *sorted(ROOT.glob('*.md')), *sorted((ROOT/'configs').glob('*.json')),
             *sorted((ROOT/'vendor').rglob('*.py')), ROOT/'vendor/LICENSE', ROOT/'vendor/PROVENANCE.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def receipt(run, name):
    d = Path(run)/'stages'/name
    if (d/'failed.json').exists(): raise ValueError('Failed stage blocks: '+name)
    r = verify(read(d/'complete.json'))
    for f,h in r['files'].items():
        target=(d/f).resolve()
        if Path(f).is_absolute() or not target.is_relative_to(d.resolve()) or sha(target) != h:
            raise ValueError('Stage artifact changed: '+name+'/'+f)
    return r

def artifact(run, name):
    receipt(run,name)
    return verify(read(Path(run)/'stages'/name/'data.json'))

def manifest(run):
    m = verify(read(Path(run)/'manifest.json'))
    if m['code'] != code_identity(): raise ValueError('Source/config identity drift')
    if 'runtime' in m:
        import numpy as np
        if m['runtime']!={'python':sys.version,'numpy':np.__version__}: raise ValueError('Run Python/NumPy runtime drift')
    return m

@contextlib.contextmanager
def stage(run, name, meta=None):
    run = inside(run); m=manifest(run)
    deps={n: receipt(run,n)['digest'] for n in DEPS[name]}
    d=run/'stages'/name
    # Exclusive directory and run lock prevent simultaneous stages/fit-budget races.
    lock=run/'active.lock'
    with lock.open('x',encoding='utf-8') as f: f.write(name)
    try:
        if d.exists(): raise ValueError('Stage already started; no automatic rerun: '+name)
        d.mkdir(parents=True)
        write(d/'started.json',sealed({'stage':name,'manifest':m['digest'],'dependencies':deps,
                                      'time':time.time(),'authorization':meta or {}}))
        try: yield d
        except BaseException as e:
            write(d/'failed.json',sealed({'type':type(e).__name__,'message':str(e),'time':time.time()}))
            raise
        else:
            # Only scan the newly created stage, including solver inputs/outputs/logs.
            files={p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()}
            write(d/'complete.json',sealed({'stage':name,'files':files,'time':time.time()}))
    finally: lock.unlink()

class Journal:
    """Fsync before invocation; unfinished started records remain charged and blocking."""
    def __init__(self,d,cap):
        self.f=inside(Path(d)/'calls.jsonl').open('x',encoding='utf-8',newline='\n')
        self.cap=cap; self.used=0; self.keys=set()
    def event(self,row):
        self.f.write(canonical(row)+'\n'); self.f.flush(); os.fsync(self.f.fileno())
    def begin(self,key,count,meta):
        if key in self.keys or count<=0 or self.used+count>self.cap: raise ValueError('Duplicate call or budget exhausted')
        self.keys.add(key); self.used+=count
        self.event({'key':key,'status':'started','charge':count,**meta})
    def close(self): self.f.close()

def cap_check(name, value, need):
    if value is None or not need<=value<=LIMITS[name]: raise ValueError('Explicit cap insufficient or exceeds frozen limit')

def init(run,c):
    run=inside(run)
    if run.exists(): raise ValueError('New run required')
    if c['experiment']!=VERSION or c['root_seed'] in (2026100401,2026101001): raise ValueError('Protocol/root seed collision')
    if c['test_instances']!=128 or c['checkpoint_sha256']!=CHECKPOINT or c['history_features']!=[] or c['device']!='cuda':
        raise ValueError('Scientific specification changed; review required')
    if c['solver_seeds']!=[271828,314159,161803] or c['solver_seconds']!=5 or c['solver_threads']!=1:
        raise ValueError('Reference budget/specification changed')
    import numpy as np
    write(run/'manifest.json',sealed({'experiment':VERSION,'config':c,'code':code_identity(),
                                      'runtime':{'python':sys.version,'numpy':np.__version__},
                                      'fixture':False,'created':time.time()}))
