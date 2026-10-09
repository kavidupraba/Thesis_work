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
PLOTS_DIR = os.path.join(BASE_DIR,'Plots','bias_variance_all')
RES_DIR   = os.path.join(BASE_DIR,'Results','bias_variance')
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
TEST_SIZE=.2; VAL_SIZE=.1; RS=42; NBOOT=20
EXCLUDE={'Close','Open','High','Low','Volume','returns','log_returns','obv','volume_ma_5',
         'Close_lag1','Close_lag7','Close_lag21','High_lag1','High_lag7','High_lag21',
         'Low_lag1','Low_lag7','Low_lag21','Open_lag1','Open_lag7','Open_lag21'}
STOCKS = sorted(f.replace('_cleaned.csv','') for f in os.listdir(CLEAN_DIR) if f.endswith('_cleaned.csv'))
all_rows=[]
for tk in STOCKS:
    try:
        fp=os.path.join(CLEAN_DIR,f'{tk}_cleaned.csv')
        df=pd.read_csv(fp,index_col='Date',parse_dates=True)
        df['target']=(df['Close'].shift(-1)>df['Close']).astype(int); df=df.dropna()
        feats=[c for c in df.columns if c not in EXCLUDE and c!='target']
        X=df[feats].values;y=df['target'].values
        n=len(y); nt=int(n*TEST_SIZE)
        if nt>=n: continue
        Xtr,Xte = X[:-nt], X[-nt:]; ytr,yte = y[:-nt], y[-nt:]
        sz=min(800,len(Xtr))
        # collect curves per model for this stock
        lr_bias,lr_var,lr_x = [],[],[]
        sc=StandardScaler().fit(Xtr[:sz]); Xtez=sc.transform(Xte)
        for c in [0.1,1.0,10.0]:
            preds=[]
            for b in range(NBOOT):
                Xb,yb=resample(Xtr[:sz],ytr[:sz],random_state=RS+b)
                m=LogisticRegression(C=c,max_iter=2000,random_state=RS+b)
                m.fit(sc.transform(Xb),yb); preds.append(m.predict_proba(Xtez)[:,1])
            Phat=np.mean(np.array(preds),axis=0); yhat=(Phat>=.5).astype(int)
            bias=np.mean(yhat!=yte); var=np.mean([np.mean(((p>=.5).astype(int))!=yhat) for p in preds])
            lr_bias.append(bias); lr_var.append(var); lr_x.append(c); all_rows.append({'stock':tk,'model':'LR','param':'C','val':c,'bias':bias,'var':var,'err':bias+var})
        rf_bias,rf_var,rf_x=[],[],[]
        for d in [3,5,12]:
            preds=[]
            for b in range(NBOOT):
                Xb,yb=resample(Xtr[:sz],ytr[:sz],random_state=RS+b)
                m=RandomForestClassifier(n_estimators=200,max_depth=d,min_samples_leaf=5,random_state=RS+b,n_jobs=-1)
                m.fit(Xb,yb); preds.append(m.predict_proba(Xte)[:,1])
            Phat=np.mean(np.array(preds),axis=0); yhat=(Phat>=.5).astype(int)
            bias=np.mean(yhat!=yte); var=np.mean([np.mean(((p>=.5).astype(int))!=yhat) for p in preds])
            rf_bias.append(bias); rf_var.append(var); rf_x.append(d); all_rows.append({'stock':tk,'model':'RF','param':'max_depth','val':d,'bias':bias,'var':var,'err':bias+var})
        xg_bias,xg_var,xg_x=[],[],[]
        for d in [2,3,5]:
            preds=[]
            for b in range(NBOOT):
                Xb,yb=resample(Xtr[:sz],ytr[:sz],random_state=RS+b)
                m=XGBClassifier(n_estimators=200,max_depth=d,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',use_label_encoder=False,random_state=RS+b,n_jobs=-1,verbosity=0)
                m.fit(Xb,yb,verbose=False); preds.append(m.predict_proba(Xte)[:,1])
            Phat=np.mean(np.array(preds),axis=0); yhat=(Phat>=.5).astype(int)
            bias=np.mean(yhat!=yte); var=np.mean([np.mean(((p>=.5).astype(int))!=yhat) for p in preds])
            xg_bias.append(bias); xg_var.append(var); xg_x.append(d); all_rows.append({'stock':tk,'model':'XGB','param':'max_depth','val':d,'bias':bias,'var':var,'err':bias+var})
        # plot
        fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(14,4))
        ax1.plot(lr_x,lr_bias,'o-',label='bias'); ax1.plot(lr_x,lr_var,'o-',label='variance'); ax1.plot(lr_x,[b+v for b,v in zip(lr_bias,lr_var)],'o--',label='bias+var'); ax1.set_xscale('log'); ax1.set_title(f'LR {tk}')
        ax1.legend(); ax1.grid(alpha=.2)
        ax2.plot(rf_x,rf_bias,'o-',label='bias'); ax2.plot(rf_x,rf_var,'o-',label='variance'); ax2.plot(rf_x,[b+v for b,v in zip(rf_bias,rf_var)],'o--',label='bias+var'); ax2.set_title(f'RF {tk}'); ax2.legend(); ax2.grid(alpha=.2)
        ax3.plot(xg_x,xg_bias,'o-',label='bias'); ax3.plot(xg_x,xg_var,'o-',label='variance'); ax3.plot(xg_x,[b+v for b,v in zip(xg_bias,xg_var)],'o--',label='bias+var'); ax3.set_title(f'XGB {tk}'); ax3.legend(); ax3.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR,f'bv_{tk}.png'), dpi=120, bbox_inches='tight'); plt.close(fig)
        print('ok',tk)
    except Exception as e: print('err',tk); continue
pd.DataFrame(all_rows).to_csv(os.path.join(RES_DIR,'bias_variance_all.csv'), index=False)
print('done')
