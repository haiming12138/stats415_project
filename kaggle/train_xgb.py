import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from process_data import get_train, NUM_COLS, CAT_COLS
from train_model_utils import save_cv_metric

X, y = get_train()

transformer = ColumnTransformer(
    [
        ('encode', OneHotEncoder(drop='first'), CAT_COLS),
        ('transform', RobustScaler(), NUM_COLS),
        ('pca', PCA(n_components='mle'), NUM_COLS)
    ],
    remainder='passthrough',
    n_jobs=-1,
    verbose_feature_names_out=False
)

pipe = Pipeline(
    [
        ('transform', transformer),
        ('reg', XGBRegressor(booster='gbtree', n_jobs=-1))
    ]
)

param = {
    'reg__n_estimators': np.arange(100, 200, 1),
    'reg__max_depth': np.arange(5, 15),
    'reg__min_child_weight': np.arange(0.0001, 0.5, 0.001),
    'reg__gamma': np.arange(0.0,20.0,0.005),
    'reg__learning_rate': np.arange(0.05,0.3,0.001),
    'reg__subsample': np.arange(0.5,0.9,0.05),
    'reg__reg_alpha': np.geomspace(0.00001, 100, 100)
}

grid = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param,
    scoring='neg_mean_squared_error',
    cv=7,
    n_jobs=-1,
    verbose=3,
    n_iter=100000
)


grid.fit(X, y)
save_cv_metric(grid.best_estimator_, X, y, 'xgb')
joblib.dump(grid.best_estimator_, './models/xgb.sav')