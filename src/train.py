"""Train on 2014-21, select on 2022-23, then evaluate once on 2024-25."""
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from src.model import RainQDA, transform_rain
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, roc_curve)
from src.data import ROOT, FEATURES, LABELS, make_frame, splits

SEED = 269220
OUT = ROOT/'results'
PLOTS = OUT/'plots'

def metrics(y, p, threshold):
    pred = p >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    two_classes = len(np.unique(y)) == 2
    return dict(auc=float(roc_auc_score(y,p)) if two_classes else None, ap=float(average_precision_score(y,p)),
        accuracy=float(accuracy_score(y,pred)), balanced_accuracy=float(balanced_accuracy_score(y,pred)) if two_classes else None,
        precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred)),
        f1=float(f1_score(y,pred)), brier=float(brier_score_loss(y,p)),
        specificity=float(tn/(tn+fp)) if tn+fp else None, threshold=float(threshold), tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def threshold_for(y,p):
    # A balanced criterion gives wet and dry days equal weight.
    grid = np.linspace(.2,.8,121)
    return float(max(grid, key=lambda t: (balanced_accuracy_score(y,p>=t),-abs(t-.5))))

def save_plot(name):
    plt.tight_layout()
    plt.savefig(PLOTS/f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close()

def run():
    OUT.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
    (ROOT/'models').mkdir(exist_ok=True)
    frame=make_frame(); train,val,test=splits(frame)
    frame.to_csv(ROOT/'data/model_dataset.csv',index=False)
    ytr,yv,yt=[x.wet_tomorrow.to_numpy() for x in (train,val,test)]
    Xtr,Xv,Xt=[x[FEATURES] for x in (train,val,test)]
    candidates=[]; best=None
    scaler=StandardScaler().fit(transform_rain(Xtr))
    ztr=scaler.transform(transform_rain(Xtr)); zv=scaler.transform(transform_rain(Xv))
    for regularisation in [0.0,.01,.05,.1,.25,.5,.75,1.0]:
        candidate=QuadraticDiscriminantAnalysis(reg_param=regularisation)
        candidate.fit(ztr,ytr)
        score=float(roc_auc_score(yv,candidate.predict_proba(zv)[:,1]))
        candidates.append(dict(reg_param=regularisation,validation_auc=score))
        if best is None or score>best[0]: best=(score,candidate)
    qda=best[1]
    model=RainQDA.from_fitted(scaler,qda,Xtr)
    # Verify the portable quadratic score against scikit-learn before export.
    for x in [Xtr,Xv,Xt]:
        np.testing.assert_allclose(model.decision_function(x),
            qda.decision_function(scaler.transform(transform_rain(x))),rtol=1e-10,atol=1e-10)
    pval=model.predict_proba(Xv)[:,1]
    threshold=threshold_for(yv,pval)
    logistic=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,random_state=SEED))
    logistic.fit(Xtr,ytr)
    plv=logistic.predict_proba(Xv)[:,1]; tl=threshold_for(yv,plv)
    month_rate=train.groupby(train.target_date.dt.month).wet_tomorrow.mean()
    pvmonth=val.target_date.dt.month.map(month_rate).to_numpy()
    tm=threshold_for(yv,pvmonth)
    # No test row has been used in model or threshold selection above this line.
    probabilities={'Regularised QDA':model.predict_proba(Xt)[:,1],
        'Logistic regression':logistic.predict_proba(Xt)[:,1],
        'Monthly climatology':test.target_date.dt.month.map(month_rate).to_numpy(),
        'Rain persistence':(test.rain_today>=1).astype(float).to_numpy(),
        'Constant train prevalence':np.full(len(test),ytr.mean())}
    thresholds={'Regularised QDA':threshold,'Logistic regression':tl,'Monthly climatology':tm,
                'Rain persistence':.5,'Constant train prevalence':.5}
    scores={k:metrics(yt,p,thresholds[k]) for k,p in probabilities.items()}
    p=probabilities['Regularised QDA']; pred=p>=threshold
    test_export=test[['observation_date','target_date','target_rain_mm','wet_tomorrow']].copy()
    for name,probs in probabilities.items(): test_export[name]=probs
    test_export.to_csv(OUT/'test_predictions.csv',index=False)
    # Exact interventional SHAP of the quadratic log-odds, training background.
    sampled=np.linspace(0,len(test)-1,min(240,len(test)),dtype=int)
    contributions,base=model.explain(Xt.iloc[sampled])
    shap=np.column_stack([contributions,np.full(len(sampled),base)])
    raw=model.decision_function(Xt.iloc[sampled])
    assert np.allclose(shap.sum(axis=1),raw,atol=1e-8)
    importance=np.abs(contributions).mean(axis=0); order=np.argsort(importance)
    pd.DataFrame({'feature':FEATURES,'label':LABELS,'mean_abs_shap':importance}).sort_values('mean_abs_shap',ascending=False).to_csv(OUT/'shap_importance.csv',index=False)
    groups={'Correct wet day':(yt==1)&pred,'Correct dry day':(yt==0)&~pred,
            'Missed wet day':(yt==1)&~pred,'False wet alert':(yt==0)&pred}
    examples=[]
    for name,mask in groups.items():
        inds=np.flatnonzero(mask)
        if not len(inds): continue
        i=int(inds[len(inds)//2]); row=test.iloc[i]
        local,base=model.explain(Xt.iloc[[i]])
        sv=np.r_[local[0],base]
        examples.append(dict(name=name,date=str(row.target_date.date()),actual=int(yt[i]),
            actual_rain=float(row.target_rain_mm),probability=float(p[i]),
            features=[float(v) for v in row[FEATURES]],shap=sv[:-1].tolist(),base=float(sv[-1])))
    summary=dict(seed=SEED,features=FEATURES,feature_labels=LABELS,raw_rows=len(pd.read_csv(ROOT/'data/colombo_daily.csv')),
        usable_rows=len(frame),splits={k:dict(n=len(v),start=str(v.target_date.min().date()),
        end=str(v.target_date.max().date()),wet=int(v.wet_tomorrow.sum()),prevalence=float(v.wet_tomorrow.mean()))
        for k,v in [('train',train),('validation',val),('test',test)]},
        candidates=candidates,selected_params=dict(reg_param=float(qda.reg_param),preprocessing="log1p of three rainfall features; train-only standardisation",priors="training class frequencies"),
        validation=metrics(yv,pval,threshold),test=scores,
        shap=dict(n=len(sampled),background_n=len(train),method='exact interventional SHAP of quadratic score',space='raw log odds',max_additivity_error=float(np.max(np.abs(shap.sum(axis=1)-raw)))),
        examples=examples,versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','matplotlib','streamlit']},
        python=platform.python_version())
    (OUT/'metrics.json').write_text(json.dumps(summary,indent=2))
    model.save(ROOT/'models/qda.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#222222','#555555','#888888','#444444','#aaaaaa']
    plt.figure(figsize=(7,4))
    for (name,probs),color,style in zip(probabilities.items(),colors,['-','--','-.',':','-']):
        fpr,tpr,_=roc_curve(yt,probs); plt.plot(fpr,tpr,label=f'{name}: {scores[name]["auc"]:.3f}',color=color,linestyle=style)
    plt.plot([0,1],[0,1],'k--',alpha=.3);plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.legend(fontsize=8);save_plot('roc')
    plt.figure(figsize=(7,3.6));plt.barh(np.array(LABELS)[order],importance[order],color=colors[0]);plt.xlabel('Mean absolute SHAP value (log odds)');save_plot('shap_global')
    print(json.dumps({'rows':len(frame),'regularisation':float(qda.reg_param),'threshold':threshold,'test':scores},indent=2))

if __name__=='__main__': run()
