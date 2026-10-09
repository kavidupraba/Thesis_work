import os, warnings, math
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
os.makedirs(PLOTS_DIR, exist_ok=True)
RS=42; TEST_SIZE=.2; VAL_SIZE=.1; NBOOT=20; MAX_SZ=800
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
# LR bias-variance (noise floor approx)
sz = min(MAX_SZ, len(Xtr))
sc = StandardScaler().fit(Xtr)
Xtz, Xtez = sc.transform(Xtr[:sz]), sc.transform(Xte)
# fit on full (proxy for main prediction) - get average prediction across bootstraps
preds = []
for b in range(NBOOT):
    Xb, yb = resample(Xtr[:sz], ytr[:sz], random_state=RS+b)
    Xbz = sc.transform(Xb)  # transform using original fit on full train? or refit - keep simple
    m = LogisticRegression(C=1.0, max_iter=3000, random_state=RS+b)
    m.fit(Xbz, yb)
    preds.append(m.predict_proba(Xtez)[:,1])
Phat = np.mean(np.array(preds), axis=0)
# 0-1 bias^2 + variance approx: E[(y!=yhat_bar)^?] simpler: bias = mean(|Phat - yte_true|>0.5? but yte 0/1) approx
yhat_bar = (Phat>=0.5).astype(int)
bias = np.mean(yhat_bar != yte)
# variance: avg over bootstraps of |yhat_b - yhat_bar|
var = np.mean([np.mean(( (p>=.5).astype(int)) != yhat_bar) for p in preds])
print('LR','bias~',bias,'var~',var,'error~',bias+var)
# RF
preds = []
for b in range(NBOOT):
    Xb, yb = resample(Xtr[:sz], ytr[:sz], random_state=RS+b)
    m = RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=5,random_state=RS+b,n_jobs=-1)
    m.fit(Xb,yb)
    preds.append(m.predict_proba(Xte)[:,1])
Phat=np.mean(np.array(preds),axis=0); yhat_bar=(Phat>=.5).astype(int)
bias=np.mean(yhat_bar!=yte); var=np.mean([np.mean(((p>=.5).astype(int))!=yhat_bar) for p in preds])
print('RF','bias~',bias,'var~',var,'error~',bias+var)
# XGB
preds = []
for b in range(NBOOT):
    Xb, yb = resample(Xtr[:sz], ytr[:sz], random_state=RS+b)
    m = XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',use_label_encoder=False,random_state=RS+b,n_jobs=-1)
    m.fit(Xb,yb, verbose=False)
    preds.append(m.predict_proba(Xte)[:,1])
Phat=np.mean(np.array(preds),axis=0); yhat_bar=(Phat>=.5).astype(int)
bias=np.mean(yhat_bar!=yte); var=np.mean([np.mean(((p>=.5).astype(int))!=yhat_bar) for p in preds])
print('XGB','bias~',bias,'var~',var,'error~',bias+var)
