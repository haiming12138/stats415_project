import pandas as pd
import numpy as np
import joblib
from process_data import get_train
from sklearn.metrics import r2_score
from nn_definition import MyModule
import matplotlib.pyplot as plt

xgb = pd.read_csv('./past_submissions/res.csv')
nn = pd.read_csv('./past_submissions/res_nn.csv')

ids = xgb['SEQN']

xgb_pred = xgb['y']
nn_pred = nn['y']

avg_pred = pd.Series(xgb_pred * 0.5 + nn_pred * 0.5)

res_df = pd.concat([xgb_pred, nn_pred, avg_pred], keys=['xgb', 'nn', 'avg'], axis=1)
res_df.to_csv('./past_submissions/mean_output.csv', index=False)

output = pd.concat([ids, avg_pred], keys=['SEQN', 'y'], axis=1)
output.to_csv('./res_mix.csv', index=False)

X, y = get_train()
model_xgb = joblib.load('./models/xgb.sav')
train_pred_xgb = model_xgb.predict(X)

model_nn = joblib.load('./models/nn.sav')
X = X.to_numpy().astype(np.float32)
train_pred_nn = model_nn.predict(X)

# ids = [*range(8000)]
# plt.figure(figsize=(50, 10))
# plt.scatter(ids, train_pred_xgb, color='green', label='xgb')
# plt.scatter(ids, train_pred_nn, color='blue', label='nn')
# plt.scatter(ids, y, color='red', label='true')
# plt.legend()

# plt.savefig('./visual.png', dpi=800)

# train_pred_nn = train_pred_nn.flatten()
# train_pred_xgb = train_pred_xgb.flatten()
# y = y.to_numpy().flatten()
# count1 = np.sum((train_pred_nn < y) & (y < train_pred_xgb))
# count2 = np.sum((train_pred_xgb < y) & (y < train_pred_nn))
# print(count1 + count2)

# print(f'XGB: {r2_score(y, train_pred_xgb)}')
# print(f'NN: {r2_score(y, train_pred_nn)}')

# mix_pred = train_pred_xgb * 0.36 + train_pred_nn * 0.64
# print(f'mix: {r2_score(y, mix_pred)}')