# Bias-variance vs complexity
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

BASE_DIR = r'C:\Users\Admin\Documents\thesis_sending\PythonProject'
CLEAN_DIR = os.path.join(BASE_DIR,'Data','cleaned')
PLOTS_DIR = os.path.join(BASE_DIR,'Plots','bias_variance')
RES_DIR = os.path.join(BASE_DIR,'Results','bias_variance')
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
RS=42; TEST_SIZE=.2; VAL_SIZE=.1; NBOOT=40
EXCLUDE={'Close','Open','High','Low','Volume','returns','log_returns','obv','volume_ma_5',
         'Close_lag1','Close_lag7','Close_lag21','High_lag1','High_lag7','High_lag21',
         'Low_lag1','Low_lag7','Low_lag21','Open_lag1','Open_lag7','Open_lag21'}
fp=os.path.join(CLEAN_DIR,'AAPL_cleaned.csv')
df=pd.read_csv(fp,index_col='Date',parse_dates=True)
df['target']=(df['Close'].shift(-1)>df['Close']).astype(int); df=df.dropna()
feats=[c for c in df.columns if c not in EXCLUDE and c!='target']
X=df[feats].values;y=df['target'].values
n=len(y); nt=int(n*TEST_SIZE); nv=int(n*VAL_SIZE)
Xtr,Xv,Xte = X[:-nt-nv], X[-nt-nv:-nt], X[-nt:]
ytr,yv,yte = y[:-nt-nv], y[-nt-nv:-nt], y[-nt:]
sz = min(1000,len(Xtr))
rows=[]
# LR
sc=StandardScaler().fit(Xtr[:sz]); Xtz=sc.transform(Xtr[:sz]); Xtez=sc.transform(Xte)
cs=[0.01,0.1,1.0,10,100]
biases=[]; vars=[]; errs=[]
for c in cs:
    preds=[]
    for b in range(NBOOT):
        Xb,yb=resample(Xtr[:sz],ytr[:sz],random_state=RS+b)
        m=LogisticRegression(C=c,max_iter=3000,random_state=RS+b)
        m.fit(sc.transform(Xb),yb)
        preds.append(m.predict_proba(Xtez)[:,1])
    Phat=np.mean(np.array(preds),axis=0); yhat_bar=(Phat>=.5).astype(int)
    bias=np.mean(yhat_bar!=yte)
    var=np.mean([np.mean(((p>=.5).astype(int))!=yhat_bar) for p in preds])
    biases.append(bias); vars.append(var); errs.append(bias+var)
    rows.append({'model':'LR','param':'C','val':c,'bias':bias,'var':var,'err':bias+var})
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(cs,biases,'o-',label='bias'); ax.plot(cs,vars,'o-',label='variance'); ax.plot(cs,errs,'o--',label='bias+var approx')
ax.set_xscale('log'); ax.set_xlabel('C (log)'); ax.set_ylabel('0-1'); ax.set_title('LR Bias-Variance vs C (AAPL)')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(os.path.join(PLOTS_DIR,'lr_bv_vs_c.png'), dpi=120, bbox_inches='tight'); plt.close(fig)
print('lr ok')
# RF
depths=[2,3,5,8,12,15]
biases=[]; vars=[]
for d in depths:
    preds=[]
    for b in range(NBOOT):
        Xb,yb=resample(Xtr[:sz],ytr[:sz],random_state=RS+b)
        m=RandomForestClassifier(n_estimators=300,max_depth=d,min_samples_leaf=5,random_state=RS+b,n_jobs=-1)
        m.fit(Xb,yb); preds.append(m.predict_proba(Xte)[:,1])
    Phat=np.mean(np.array(preds),axis=0); yhat_bar=(Phat>=.5).astype(int)
    bias=np.mean(yhat_bar!=yte); var=np.mean([np.mean(((p>=.5).astype(int))!=yhat_bar) for p in preds])
    biases.append(bias); vars.append(var)
    rows.append({'model':'RF','param':'max_depth','val':d,'bias':bias,'var':var,'err':bias+var})
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(depths,biases,'o-',label='bias'); ax.plot(depths,vars,'o-',label='variance'); ax.plot(depths,[b+v for b,v in zip(biases,vars)],'o--',label='bias+var')
ax.set_xlabel('max_depth'); ax.set_ylabel('0-1'); ax.set_title('RF Bias-Variance vs max_depth (AAPL)')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(os.path.join(PLOTS_DIR,'rf_bv_vs_depth.png'), dpi=120, bbox_inches='tight'); plt.close(fig)
print('rf ok')
# XGB
depths=[1,2,3,4,5,6]
biases=[]; vars=[]
for d in depths:
    preds=[]
    for b in range(NBOOT):
        Xb,yb=resample(Xtr[:sz],ytr[:sz],random_state=RS+b)
        m=XGBClassifier(n_estimators=300,max_depth=d,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',use_label_encoder=False,random_state=RS+b,n_jobs=-1)
        m.fit(Xb,yb,verbose=False); preds.append(m.predict_proba(Xte)[:,1])
    Phat=np.mean(np.array(preds),axis=0); yhat_bar=(Phat>=.5).astype(int)
    bias=np.mean(yhat_bar!=yte); var=np.mean([np.mean(((p>=.5).astype(int))!=yhat_bar) for p in preds])
    biases.append(bias); vars.append(var)
    rows.append({'model':'XGB','param':'max_depth','val':d,'bias':bias,'var':var,'err':bias+var})
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(depths,biases,'o-',label='bias'); ax.plot(depths,vars,'o-',label='variance'); ax.plot(depths,[b+v for b,v in zip(biases,vars)],'o--',label='bias+var')
ax.set_xlabel('max_depth'); ax.set_ylabel('0-1'); ax.set_title('XGB Bias-Variance vs max_depth (AAPL)')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(os.path.join(PLOTS_DIR,'xgb_bv_vs_depth.png'), dpi=120, bbox_inches='tight'); plt.close(fig)
pd.DataFrame(rows).to_csv(os.path.join(RES_DIR,'bias_variance_summary.csv'), index=False)
print('done')
