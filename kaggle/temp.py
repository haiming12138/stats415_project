import joblib

model = joblib.load('./models/xgb_full.sav')
params = model.get_params()
print(params['steps'][1][1])