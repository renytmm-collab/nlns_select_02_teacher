import argparse
import importlib.util
import json
import sys
from pathlib import Path
from .core import read,ROOT,LIMITS,DEPS,manifest,write,inside,canonical

def main(argv=None):
    p=argparse.ArgumentParser(description='Fixed Repair privileged-reference diagnostic; default read-only')
    p.add_argument('command',choices=['check','audit-source','budget','audit','analyze',*DEPS])
    p.add_argument('--config',default=str(ROOT/'configs/experiment.json'))
    p.add_argument('--run',default=str(ROOT/'runs/reference-v1'))
    p.add_argument('--start',action='store_true'); p.add_argument('--cap',type=int)
    p.add_argument('--max-fits',type=int); p.add_argument('--solver-budget',type=int)
    p.add_argument('--solver-seconds',type=float); p.add_argument('--solver-threads',type=int)
    p.add_argument('--solver-seeds',type=int,nargs=3); p.add_argument('--output')
    a=p.parse_args(argv); readonly=a.command in ('check','audit-source','budget','audit','analyze')
    if (not readonly or a.output) and not a.start: p.error('Writing requires explicit --start')
    if a.output and a.command!='analyze': p.error('--output only supported by analyze')
    try:
        c=read(a.config)
        if a.command=='check':
            from .core import CHECKPOINT,sha
            result={'project':str(ROOT),'python':sys.version,'numpy_available':importlib.util.find_spec('numpy') is not None,
                    'torch_available':importlib.util.find_spec('torch') is not None,
                    'checkpoint_present':Path(c['checkpoint']).is_file(),'expected_checkpoint':CHECKPOINT,
                    'solver_present':Path(c['solver_executable']).is_file(),'real_runtime_verified':False,
                    'no_install_or_real_compute':True}
            if result['checkpoint_present']: result['checkpoint_hash_matches']=sha(c['checkpoint'])==CHECKPOINT
            import numpy as np
            result.update(numpy=np.__version__,production_numpy_matches=np.__version__=='1.26.4')
        elif a.command=='budget':
            result={'stage_limits':LIMITS,'new_repair_upper':288768,'solver_cpu_limit_seconds':2880,
                    'test_n':128,'research_fits_upper':125,'historical_cost':'separate, not zero','failures_uncertain_charged':True}
        elif a.command=='audit-source':
            from .source import audit_source
            result=audit_source(c)
        elif a.command=='audit':
            from .analysis import audit
            result=audit(a.run)
        elif a.command=='analyze':
            from .analysis import report,markdown
            result=report(a.run)
            if a.output:
                out=inside(a.output)
                if out.exists(): raise ValueError('Report directory exists; no overwrite')
                write(out/'report.json',result)
                (out/'report.md').write_text(markdown(result),encoding='utf-8')
        else:
            from . import pipeline as q
            command=a.command
            if command=='import-development': q.initialize_development(a.run,c)
            else:
                if manifest(a.run)['fixture']: raise ValueError('Production CLI rejects synthetic manifests')
                if command in ('generate-dev-references','generate-test-references'): q.references(a.run,'dev' if 'dev' in command else 'test',a)
                elif command in ('build-dev-features','build-test-features'): q.features(a.run,'dev' if 'dev' in command else 'test')
                elif command=='fit-oof': q.fit_oof(a.run,a.max_fits)
                elif command=='assess-development': q.assess(a.run,a.max_fits)
                elif command=='calibrate': q.calibrate(a.run,a.cap)
                elif command=='freeze-plan': q.freeze(a.run)
                elif command=='generate-test-states': q.states(a.run,a.cap)
                elif command=='build-test-candidates': q.candidates(a.run)
                elif command=='freeze-test-predictions': q.predictions(a.run)
                elif command=='collect-test-b': q.collect(a.run,a.cap)
                else: raise ValueError('Unknown command')
            result={'stage':command,'completed':True,'research_success':False}
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    except (ValueError,RuntimeError,OSError,KeyError,ImportError) as e:
        print(canonical({'blocked':True,'error_type':type(e).__name__,'reason':str(e)}),file=sys.stderr)
        raise SystemExit(2)
