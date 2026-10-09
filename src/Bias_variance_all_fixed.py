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
