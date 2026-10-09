"""
Learning_curves.py (extended)
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = r'C:\Users\Admin\Documents\thesis_sending\PythonProject'
CLEAN_DIR = os.path.join(BASE_DIR,'Data','cleaned')
PLOTS_DIR = os.path.join(BASE_DIR,'Plots','learning_curves')
os.makedirs(PLOTS_DIR, exist_ok=True)

TICKER='AAPL'; TEST_SIZE=.2; VAL_SIZE=.1; RS=42
EXCLUDE={'Close','Open','High','Low','Volume','returns','log_returns','obv','volume_ma_5',
         'Close_lag1','Close_lag7','Close_lag21','High_lag1','High_lag7','High_lag21',
         'Low_lag1','Low_lag7','Low_lag21','Open_lag1','Open_lag7','Open_lag21'}

fp = os.path.join(CLEAN_DIR,f'{TICKER}_cleaned.csv')
df = pd.read_csv(fp,index_col='Date',parse_dates=True)
df['target']=(df['Close'].shift(-1)>df['Close']).astype(int)
df=df.dropna()
feats=[c for c in df.columns if c not in EXCLUDE and c!='target']
X=df[feats].values; y=df['target'].values
n=len(y); nt=int(n*TEST_SIZE); nv=int(n*VAL_SIZE)
Xtr,Xv,Xte = X[:-nt-nv], X[-nt-nv:-nt], X[-nt:]
ytr,yv,yte = y[:-nt-nv], y[-nt-nv:-nt], y[-nt:]
sc = StandardScaler().fit(Xtr)
Xtrz, Xvz = sc.transform(Xtr), sc.transform(Xv)
tr_sizes = [50,100,200,400,800,1200,1435]
lr_tr, lr_val = [], []
for sz in tr_sizes:
    m = LogisticRegression(C=1.0, max_iter=3000, random_state=RS)
    m.fit(Xtrz[:sz], ytr[:sz])
    p_tr = m.predict_proba(Xtrz[:sz])[:,1]; p_v = m.predict_proba(Xvz)[:,1]
    lr_tr.append(log_loss(ytr[:sz], p_tr)); lr_val.append(log_loss(yv, p_v))
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(tr_sizes, lr_tr, 'o-', label='train logloss'); ax.plot(tr_sizes, lr_val, 'o-', label='val logloss')
ax.set_xlabel('training samples'); ax.set_ylabel('logloss'); ax.set_title('LR Learning Curve (C=1.0, AAPL 70/10/20)')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(os.path.join(PLOTS_DIR,'lr_learning_curve.png'), dpi=120, bbox_inches='tight')
plt.close(fig)
print('lr done')
# RF
rf_tr, rf_val = [], []
for sz in tr_sizes:
    m = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=RS, n_jobs=-1)
    m.fit(Xtr[:sz], ytr[:sz])
    p_tr = m.predict_proba(Xtr[:sz])[:,1]; p_v = m.predict_proba(Xv)[:,1]
    rf_tr.append(log_loss(ytr[:sz], p_tr)); rf_val.append(log_loss(yv, p_v))
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(tr_sizes, rf_tr, 'o-', label='train logloss'); ax.plot(tr_sizes, rf_val, 'o-', label='val logloss')
ax.set_xlabel('training samples'); ax.set_ylabel('logloss'); ax.set_title('RF Learning Curve (AAPL 70/10/20)')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(os.path.join(PLOTS_DIR,'rf_learning_curve.png'), dpi=120, bbox_inches='tight')
plt.close(fig)
print('rf done')
# XGB
xgb_tr, xgb_val = [], []
for sz in tr_sizes:
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', use_label_encoder=False, random_state=RS, n_jobs=-1)
    m.fit(Xtr[:sz], ytr[:sz], eval_set=[(Xv,yv)], verbose=False)
    p_tr = m.predict_proba(Xtr[:sz])[:,1]; p_v = m.predict_proba(Xv)[:,1]
    xgb_tr.append(log_loss(ytr[:sz], p_tr)); xgb_val.append(log_loss(yv, p_v))
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(tr_sizes, xgb_tr, 'o-', label='train logloss'); ax.plot(tr_sizes, xgb_val, 'o-', label='val logloss')
ax.set_xlabel('training samples'); ax.set_ylabel('logloss'); ax.set_title('XGB Learning Curve (AAPL 70/10/20)')
ax.legend(); ax.grid(alpha=.2)
fig.savefig(os.path.join(PLOTS_DIR,'xgb_learning_curve.png'), dpi=120, bbox_inches='tight')
plt.close(fig)
print('xgb done')
