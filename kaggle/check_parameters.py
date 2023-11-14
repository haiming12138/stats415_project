import joblib

# model = joblib.load('./linear.sav')
# params = model.get_params()
# print(params['steps'][1][1].get_params()['regressor'])

# model = joblib.load('./models/xgb_curr_best.sav')
# params = model.get_params()
# print(params['steps'][0][1].get_params()['pca'])

model = joblib.load('./models/xgb_optim.sav')
params = model.get_params()
print(params['steps'][1][1].get_params())