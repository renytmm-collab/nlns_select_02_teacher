import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.dont_write_bytecode=True
from teacher_select.core import *
from teacher_select.features import extra,edges,fragments,eligible,build_table,pool
from teacher_select.reference import parse,encode,validate,select,route_cost
from teacher_select.models import fit,predict,metric
from teacher_select.analysis import decide,interval,report
from teacher_select.source import load_labels
from point_select.data import validate_deletion,routes,partial_solution

def state():
    xy=[[((i*37)%101)/101,((i*61)%103)/103] for i in range(101)]
    rr=[list(range(i,i+10)) for i in range(1,101,10)]
    return {'locations':xy,'original_locations':xy,'demand':[0]+[1]*100,'capacity':50,
            'solution':[[[i,0 if i==0 else 1,0] for i in [0,*r,0]] for r in rr]}

class Properties(unittest.TestCase):
    def setUp(self): self.s=state(); self.d=list(range(1,21)); self.ref=[list(reversed(r)) for r in routes(self.s)]
    def test_route_order_and_reversal(self):
        base=extra(self.s,self.d,self.ref)[:3]; s=copy.deepcopy(self.s)
        s['solution']=[list(reversed(t)) for t in reversed(s['solution'])]
        import numpy as np
        for a,b in zip(base,extra(s,self.d,list(reversed(self.ref)))[:3]): np.testing.assert_allclose(a,b,atol=1e-15)
    def test_depot_singleton_multiplicity(self): self.assertEqual(edges([[1],[2]])[(0,1)],2)
    def test_many_singleton_routes(self):
        s=copy.deepcopy(self.s); s['solution']=[[[0,0,0],[i,1,0],[0,0,0]] for i in range(1,101)]
        a=extra(s,self.d,[[i] for i in range(1,101)])
        self.assertEqual(a[1][2],.2)
        self.assertEqual(a[1][0],0.)
        self.assertEqual(len(fragments(s,self.d)[0]),100)
        s['solution'].reverse(); self.assertEqual(a,extra(s,self.d,[[i] for i in range(1,101)]))
    def test_whole_retained_routes_locked(self):
        ff,mm=fragments(self.s,self.d); self.assertFalse(ff[mm[31]]['open']); self.assertEqual(eligible(31,1,ff,mm,50),(False,False,False))
    def test_fixed_twenty_unique(self):
        for d in ([0,*range(1,20)],list(range(1,20)),[1]*20,list(range(81,102))):
            with self.assertRaises(ValueError): validate_deletion(d)
        validate_deletion(self.d)
    def test_same_fragment_not_merge(self):
        ff=[{'ends':{1,2},'load':10,'open':True}]; mm={1:0,2:0}
        self.assertEqual(eligible(1,2,ff,mm,50),(True,False,False))
    def test_internal_customer_locked(self):
        ff=[{'ends':{1,3},'load':10,'open':True}]; mm={1:0,2:0,3:0}
        self.assertEqual(eligible(2,0,ff,mm,50),(False,False,False))
    def test_capacity_necessary_not_sufficient(self):
        ff=[{'ends':{1},'load':30,'open':True},{'ends':{2},'load':30,'open':True}]
        self.assertEqual(eligible(1,2,ff,{1:0,2:1},50),(True,True,False))
    def test_depot_closure(self):
        ff=[{'ends':{1},'load':30,'open':True}]; self.assertEqual(eligible(1,0,ff,{1:0},50),(True,True,True))
    def test_partial_boundary_matches_semantics(self):
        d=list(range(2,22)); pp=partial_solution(self.s,d)
        self.assertTrue(any(t[0][0]==0 and t[-1][0]==1 for t in pp))
        self.assertTrue(any(t[0][0]==22 and t[-1][0]==0 for t in pp))
    def test_difference_not_gain(self):
        _,diff,ready,_=extra(self.s,self.d,[[i] for i in range(1,101)])
        self.assertGreater(sum(diff),0); self.assertEqual(len(ready),6)
        # No label or success output exists in feature computation.
    def test_shuffle_and_no_history(self):
        t=pool(self.s,1,0,50); a=build_table(self.s,t,self.ref,1); b=build_table(self.s,t,self.ref,1)
        self.assertEqual(a,b); self.assertEqual(len(a['slots'][0]['X']),12)
        self.s['future_history']=[{'z':1000}]; self.assertEqual(extra(self.s,self.d,self.ref)[:3],extra(state(),self.d,self.ref)[:3])
    def test_fold_preprocess(self):
        rows=[{'values':[float(i),1.],'target':i/100,'weight':1} for i in range(4)]
        m=fit(rows,{'family':'ridge','alpha':.1}); before=copy.deepcopy(m['preprocessing'])
        predict(m,[[1e9,-1e9]]); self.assertEqual(before,m['preprocessing']); self.assertEqual(before['mean'][0],1.5)
    def test_constant_native_and_duplicate_slot_weight(self):
        rows=[]
        for step in (50,200):
            for j in range(16): rows.append({'instance':0,'step':step,'target':1. if j<3 else 0.,'candidate':'same' if j<3 else str(j)})
        self.assertEqual(metric(rows,[0]*32)['0'],3/16)
        self.assertEqual(metric(rows,[1 if j%16<3 else 0 for j in range(32)])['0'],1.)
    def test_truncate_then_mean(self):
        y=[-.2,.1]; self.assertEqual(sum(max(v,0) for v in y)/2,.05)
        self.assertNotEqual(max(sum(y)/2,0),.05)
    def test_reference_precision(self):
        self.assertIn('NODE_COORD_SECTION',encode(self.s)); rr=routes(self.s); c=route_cost(self.s,rr)
        text='\n'.join(f'Route #{j+1}: '+ ' '.join(map(str,r)) for j,r in enumerate(rr))+f'\nCost {c:.8f}\n'
        self.assertAlmostEqual(parse(text,self.s)['cost'],c)
        with self.assertRaises(ValueError): parse(text.replace(f'{c:.8f}','1.00000000'),self.s)
    def test_reference_infeasible_and_duplicates(self):
        with self.assertRaises(ValueError): validate(self.s,[[1]*100])
        s=copy.deepcopy(self.s); s['demand']=[0]+[9]*100
        with self.assertRaises(ValueError): validate(s,self.ref)
    def test_reference_selection_has_no_label_access(self):
        a={'status':'ok','cost':5,'seed_index':0,'future_B':0}; b={'status':'ok','cost':6,'seed_index':1,'future_B':1e9}
        self.assertEqual(select([a,b]),a); self.assertIsNone(select([{'status':'failed'}]))
    def test_decision_computed_fixture_invalid(self):
        v={'mean_pp':.02,'t_lower_pp':.001,'t_upper_pp':.03,'bootstrap95_pp':[.001,.03],'drop_largest_mean_pp':.01,'drop_two_largest_mean_pp':.01}
        self.assertEqual(decide({'C1':v},v,'Tready'),'GO_REFERENCE')
        self.assertEqual(decide({'C1':v},v,'Tdiff'),'GO_REFERENCE_BUT_MECHANISM_UNPROVEN')
        self.assertEqual(decide({'C1':v},v,'Tready',fixture=True),'INVALID')
        bad={**v,'t_lower_pp':-.01,'t_upper_pp':.009,'bootstrap95_pp':[-.01,.009]}
        self.assertEqual(decide({'C1':bad},v,'Tready'),'STOP_PRACTICAL')
        bad['bootstrap95_pp']=[-.01,.03]; self.assertEqual(decide({'C1':bad},v,'Tready'),'INCONCLUSIVE')
    def test_budget_caps(self):
        for cap in (None,1023,1025):
            with self.assertRaises(ValueError): cap_check('calibrate',cap,1024)
    def test_stream_and_missing_label_invalid(self):
        self.assertFalse(report(ROOT/'tests/nonexistent-run')['valid'])
    def test_fallback_freeze_no_labels(self):
        t=pool(self.s,4,0,50); ft=build_table(self.s,t,self.ref,4)
        self.assertTrue(ft['fallback'])
    def test_journal_fail_closed_and_stage_identity(self):
        temp_root=ROOT/'tests/_tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            d=Path(td); j=Journal(d,1); j.begin('call',1,{'fixture':True})
            with self.assertRaises(ValueError): j.begin('call',1,{})
            with self.assertRaises(ValueError): j.begin('second',1,{})
            j.close(); self.assertEqual(list(stream(d/'calls.jsonl'))[0]['status'],'started')
            write(d/'manifest.json',sealed({'config':{},'fixture':True,'code':code_identity()}))
            with stage(d,'import-development') as sd: write(sd/'data.json',sealed({'fixture':True}))
            with self.assertRaises(ValueError):
                with stage(d,'import-development'): pass
            with self.assertRaises(FileNotFoundError):
                with stage(d,'collect-test-b'): pass
            receipt(d,'import-development')
            (d/'stages/import-development/data.json').write_text('{}')
            with self.assertRaises(ValueError): receipt(d,'import-development')
    def test_failed_stage_blocks(self):
        temp_root=ROOT/'tests/_tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            d=Path(td); write(d/'manifest.json',sealed({'fixture':True,'config':{},'code':code_identity()}))
            with self.assertRaises(RuntimeError):
                with stage(d,'import-development'): raise RuntimeError('synthetic failure')
            self.assertTrue((d/'stages/import-development/failed.json').exists())
            with self.assertRaises(ValueError): receipt(d,'import-development')
    def test_outside_paths_and_tampered_seals(self):
        with self.assertRaises(ValueError): inside(ROOT.parent/'old')
        with self.assertRaises(ValueError): verify({'digest':'success'})
    def test_report_generator_and_readonly_cli(self):
        from teacher_select.analysis import markdown
        from teacher_select.cli import main
        from unittest.mock import patch
        self.assertIn('当前实验有效吗',markdown({'valid':False,'decision':'INVALID','reason':'fixture','answers':{}}))
        with patch('teacher_select.cli.write',side_effect=AssertionError('read-only wrote')),patch('builtins.print'):
            main(['check']); main(['budget']); main(['analyze','--run',str(ROOT/'tests/nonexistent-run')])
        with patch('sys.stderr'),self.assertRaises(SystemExit): main(['fit-oof'])
    def test_group_dimensions(self):
        from teacher_select.features import vector
        t=build_table(self.s,pool(self.s,1,0,50),self.ref,1)
        self.assertEqual([len(vector(t['slots'][0],g)) for g in GROUPS],[12,24,18,24,24])

    def test_synthetic_oof_and_final_fit_pipeline(self):
        from teacher_select.pipeline import fit_oof,assess
        temp_root=ROOT/'tests/_tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            d=Path(td); write(d/'manifest.json',sealed({'config':{'root_seed':7},'fixture':True,'code':code_identity()}))
            tables=[]; means=[]
            for i in range(8):
                for step in (50,200):
                    slots=[]
                    for j in range(16):
                        s={'slot':j,'candidate':str(j),'X':[float(j%3)]*12,'current':[float(j%3)]*12,
                           'diff':[float(j%3)]*6,'ready':[float(j%3)]*6,'shuffle':[float((j+1)%3)]*12,
                           'rules':{'contact_edges':float(j%3),'release_concentration':float(j%3)}}
                        slots.append(s); lab={'target':(j%3)*.001}
                        means.append({'instance':i,'step':step,'slot':j,'candidate':str(j),'A':lab,'V':lab})
                    tables.append({'instance':i,'step':step,'slots':slots,'fallback':False})
            with stage(d,'import-development') as sd:
                write(sd/'data.json',sealed({'label_means':means,'folds':{str(i):i%4 for i in range(8)},'old_oof':{},'old_models':{},'fixture':True}))
            with stage(d,'generate-dev-references') as sd: write(sd/'data.json',sealed({'fixture':True}))
            with stage(d,'build-dev-features') as sd: write(sd/'data.json',sealed({'tables':tables,'fixture':True}))
            fit_oof(d,120); oo=artifact(d,'fit-oof'); self.assertEqual(oo['fits'],120)
            for group in oo['groups'].values():
                for option in group:
                    for fold in option['fold_lineage']:
                        self.assertFalse(set(fold['train_instances'])&set(fold['validation_instances']))
            assess(d,5); a=artifact(d,'assess-development'); self.assertEqual(a['fits'],5)
            self.assertEqual(set(a['models']),set(GROUPS)); self.assertTrue(a['exploratory'])

    def test_fixture_label_and_incomplete_coverage_block(self):
        temp_root=ROOT/'tests/_tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as td:
            d=Path(td); target=d/'stages/collect-test-b'; target.mkdir(parents=True)
            tables=[{'instance':0,'step':50,'unique':{'a':list(range(1,21))}}]
            seed=words(7,'B',0,50,'a',0)
            row={'instance':0,'step':50,'candidate':'a','repeat':0,'status':'ok','fixture':True,'repair_invoked':True,
                 'group':'B','seed_words':seed,'cost_before':1.,'cost_after':1.,'y':0.,'z':0.,'input_points':21,'padded_input_points':21,'batch_size':16,'batch':'b'}
            (target/'labels.jsonl').write_text(canonical(row)+'\n')
            with self.assertRaisesRegex(ValueError,'synthetic'): load_labels(d,'collect-test-b',tables,64,7,False)
            row['fixture']=False; (target/'labels.jsonl').write_text(canonical(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Incomplete'): load_labels(d,'collect-test-b',tables,64,7,False)

if __name__=='__main__': unittest.main()
