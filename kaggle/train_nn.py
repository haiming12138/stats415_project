import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from skorch import NeuralNetRegressor
from process_data import get_train, get_test
from nn_definition import MyModule

X, y = get_train()
X = X.to_numpy().astype(np.float32)
y = y.to_numpy().astype(np.float32)


net = NeuralNetRegressor(
    MyModule(),
    max_epochs=282,
    optimizer=torch.optim.Adam,
    optimizer__lr=0.00018,
    optimizer__weight_decay=0.01,
    optimizer__amsgrad=True,
    optimizer__foreach=True,
    optimizer__betas=(0.9, 0.999),
    optimizer__eps=1e-8,
    iterator_train__shuffle=True,
    batch_size=200,
    # train_split=False
)

pipe = Pipeline([('scaler', RobustScaler()), ('reg', net)])

net.fit(X, y)

joblib.dump(net, './models/nn.sav')
print(round(net.score(X,y), 7))


X_test, ids = get_test()
X_test = X_test.to_numpy().astype(np.float32)

pred = pd.Series(net.predict(X_test).flatten())
res = pd.concat([ids, pred], keys=['SEQN', 'y'], axis=1)
res.to_csv('./res_nn.csv', index=False)