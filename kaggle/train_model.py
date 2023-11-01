import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, \
    RobustScaler, StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from process_data import get_train_data, NUM_COLS, CAT_COLS
from sklearn.model_selection import ParameterGrid

X, y = get_train_data()

transformer = ColumnTransformer(
    [
        ('encode', OneHotEncoder(drop='first'), CAT_COLS),
        ('transform', RobustScaler(), NUM_COLS)
    ],
    remainder='passthrough',
    n_jobs=-1,
    verbose_feature_names_out=False
)
pipe = Pipeline(
    [
        ('transform', transformer),
        ('clf', XGBClassifier(n_jobs=-1))
    ]
)

param = {
    'clf__n_estimators': np.arange(10, 31, 2),
    'clf__max_depth': np.arange(5, 15),
    'clf__min_child_weight': np.arange(0.0001, 0.5, 0.001),
    'clf__gamma': np.arange(0.0,20.0,0.005),
    'clf__learning_rate': np.arange(0.0005,0.3,0.0005),
    'clf__subsample': np.arange(0.5,0.9,0.05),
    'clf__reg_alpha': np.linspace(0, 100, 50),
    'clf__reg_lambda': np.linspace(0, 100, 50),
    'clf__colsample_bylevel': np.linspace(0.1, 0.8, 9),
    'clf__colsample_bynode': np.linspace(0.1, 0.8, 9),
    'clf__colsample_bytree': np.linspace(0.1, 0.8, 9),
}

grid = RandomizedSearchCV(
    estimator=pipe, 
    param_distributions=param,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=7, random_state=415, shuffle=True),
    n_jobs=-1,
    verbose=0,
    n_iter=1000000
)


grid.fit(X, y)
joblib.dump(grid.best_estimator_, 'model.sav')

def save_cv_metric(classifier, X, y):
    metrics = ['f1', 'balanced_accuracy', 'roc_auc']
    res = cross_validate(classifier, X, y, cv=7, scoring=metrics)

    output = ''
    for metric in metrics:
        perf = np.round(res[f'test_{metric}'].mean(), 3)
        output += f'cv_{metric}: {perf}\n'
    with open(f'./perf_log.txt', '+w') as file:
        file.write(output)

save_cv_metric(grid.best_estimator_, X, y)