import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.model_selection import cross_validate, StratifiedKFold
from process_data import get_test_data

model = joblib.load('./model.sav')

X, ids = get_test_data()
pred = pd.Series(model.predict(X))
res = pd.concat([ids, pred], keys=['PassengerId', 'Survived'], axis=1)
res.to_csv('./res.csv', index=False)
print(res)

