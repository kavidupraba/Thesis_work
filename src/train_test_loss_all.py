import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

BASE_DIR = r'C:\Users\Admin\Documents\thesis_sending\PythonProject'
CLEAN_DIR = os.path.join(BASE_DIR,'Data','cleaned')
PLOTS_DIR = os.path.join(BASE_DIR,'Plots','train_test_loss_all')
RES_DIR   = os.path.join(BASE_DIR,'Results','train_test_loss')
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
TEST_SIZE=.2; RS=42
EXCLUDE={'Close','Open','High','Low','Volume','returns','log_returns','obv','volume_ma_5',
         'Close_lag1','Close_lag7','Close_lag21','High_lag1','High_lag7','High_lag21',
         'Low_lag1','Low_lag7','Low_lag21','Open_lag1','Open_lag7','Open_lag21'}
STOCKS = sorted(f.replace('_cleaned.csv','') for f in os.listdir(CLEAN_DIR) if f.endswith('_cleaned.csv'))
rows=[]
tr_sizes = [50,100,200,400,800,1200]
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
        szs = [s for s in tr_sizes if s<=len(Xtr)]
        if not szs: szs=[len(Xtr)]
        lr_tr,lr_te,lr_n=[],[],[]
        sc=StandardScaler().fit(Xtr); Xtrz=sc.transform(Xtr); Xtez=sc.transform(Xte)
        for s in szs:
            m=LogisticRegression(C=1.0,max_iter=2000,random_state=RS)
            m.fit(Xtrz[:s],ytr[:s])
            trl=log_loss(ytr[:s],m.predict_proba(Xtrz[:s])[:,1]); tel=log_loss(yte,m.predict_proba(Xtez)[:,1])
            lr_tr.append(trl); lr_te.append(tel); lr_n.append(s); rows.append({'stock':tk,'model':'LR','ntrain':s,'train_logloss':trl,'test_logloss':tel})
        rf_tr,rf_te,rf_n=[],[],[]
        for s in szs:
            m=RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=5,random_state=RS,n_jobs=-1)
            m.fit(Xtr[:s],ytr[:s])
            trl=log_loss(ytr[:s],m.predict_proba(Xtr[:s])[:,1]); tel=log_loss(yte,m.predict_proba(Xte)[:,1])
            rf_tr.append(trl); rf_te.append(tel); rf_n.append(s); rows.append({'stock':tk,'model':'RF','ntrain':s,'train_logloss':trl,'test_logloss':tel})
        xg_tr,xg_te,xg_n=[],[],[]
        for s in szs:
            m=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',use_label_encoder=False,random_state=RS,n_jobs=-1,verbosity=0)
            m.fit(Xtr[:s],ytr[:s],verbose=False)
            trl=log_loss(ytr[:s],m.predict_proba(Xtr[:s])[:,1]); tel=log_loss(yte,m.predict_proba(Xte)[:,1])
            xg_tr.append(trl); xg_te.append(tel); xg_n.append(s); rows.append({'stock':tk,'model':'XGB','ntrain':s,'train_logloss':trl,'test_logloss':tel})
        # plots
        fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(14,4))
        ax1.plot(lr_n,lr_tr,'o-',label='train'); ax1.plot(lr_n,lr_te,'o-',label='test'); ax1.set_title(f'LR {tk}'); ax1.set_xlabel('ntrain'); ax1.set_ylabel('logloss'); ax1.legend(); ax1.grid(alpha=.2)
        ax2.plot(rf_n,rf_tr,'o-',label='train'); ax2.plot(rf_n,rf_te,'o-',label='test'); ax2.set_title(f'RF {tk}'); ax2.set_xlabel('ntrain'); ax2.set_ylabel('logloss'); ax2.legend(); ax2.grid(alpha=.2)
        ax3.plot(xg_n,xg_tr,'o-',label='train'); ax3.plot(xg_n,xg_te,'o-',label='test'); ax3.set_title(f'XGB {tk}'); ax3.set_xlabel('ntrain'); ax3.set_ylabel('logloss'); ax3.legend(); ax3.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR,f'tl_{tk}.png'), dpi=120, bbox_inches='tight'); plt.close(fig)
        print('ok',tk)
    except Exception as e: print('err',tk); continue
pd.DataFrame(rows).to_csv(os.path.join(RES_DIR,'train_test_loss_all.csv'), index=False)
print('done')
