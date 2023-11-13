import joblib

# model = joblib.load('./linear.sav')
# params = model.get_params()
# print(params['steps'][1][1].get_params()['regressor'])

model = joblib.load('./models/xgb.sav')
params = model.get_params()
print(params['steps'][1][1])