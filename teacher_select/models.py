import numpy as np
from point_select.trees import fit_tree,predict_tree
from point_select.models import tie_indices,groups

def specs():
    return ([{'family':'ridge','alpha':a} for a in (.01,.1,1.)]+
            [{'family':'gbdt','depth':d,'trees':t,'min_leaf':64,'learning_rate':.05} for d,t in ((1,32),(2,32),(2,64))])

def fit(rows,spec):
    x=np.asarray([r['values'] for r in rows],float); y=np.asarray([r['target'] for r in rows],float)
    if not len(x) or not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid training target')
    w=np.asarray([r['weight'] for r in rows],float)
    if (w<=0).any() or not np.isfinite(w).all(): raise ValueError('Invalid weights')
    w=w/w.sum()
    med=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in x.T])
    xx=np.where(np.isfinite(x),x,med); mean=np.average(xx,axis=0,weights=w)
    scale=np.sqrt(np.average((xx-mean)**2,axis=0,weights=w)); scale=np.where(scale>1e-12,scale,1)
    xx=(xx-mean)/scale; b=float(np.average(y,weights=w))
    m={'spec':spec,'features':x.shape[1],'preprocessing':{'median':med.tolist(),'mean':mean.tolist(),'scale':scale.tolist()},'intercept':b}
    if spec['family']=='ridge':
        beta=np.linalg.solve(xx.T@(w[:,None]*xx)+spec['alpha']*np.eye(x.shape[1]),xx.T@(w*(y-b)))
        m['coefficients']=beta.tolist()
    else:
        pred=np.full(len(y),b); trees=[]
        for _ in range(spec['trees']):
            t=fit_tree(xx,y-pred,w,spec['depth'],spec['min_leaf']); pred+=.05*predict_tree(t,xx); trees.append(t)
        m['trees']=trees
    return m

def predict(m,x):
    x=np.asarray(x,float); p=m['preprocessing']
    if x.ndim!=2 or x.shape[1]!=m['features']: raise ValueError('Feature identity')
    xx=(np.where(np.isfinite(x),x,p['median'])-p['mean'])/p['scale']
    if m['spec']['family']=='ridge': v=xx@np.asarray(m['coefficients'])+m['intercept']
    else:
        v=np.full(len(x),m['intercept'])
        for t in m['trees']: v+=.05*predict_tree(t,xx)
    if not np.isfinite(v).all(): raise ValueError('Nonfinite scores')
    return v.tolist()

def metric(rows,scores):
    by={}
    for (i,step),ii in groups(rows).items():
        if len(ii)!=16: raise ValueError('Native slots missing')
        tied=tie_indices([scores[j] for j in ii]); value=np.mean([rows[ii[j]]['target'] for j in tied])
        by.setdefault(i,[]).append((step,float(value)))
    if any(len(v)!=2 or {s for s,_ in v}!={50,200} for v in by.values()): raise ValueError('State completeness')
    return {str(i):float(np.mean([x for _,x in v])) for i,v in sorted(by.items())}
