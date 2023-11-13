import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.model_selection import KFold
from process_data import get_train, NUM_COLS, CAT_COLS
from train_model_utils import save_cv_metric

X, y = get_train()

transformer = ColumnTransformer(
    [
        ('encode', OneHotEncoder(drop='first'), CAT_COLS),
        ('transform', PowerTransformer(), NUM_COLS),
        ('pca', PCA(n_components='mle'), NUM_COLS)
    ],
    remainder='passthrough',
    n_jobs=-1,
    verbose_feature_names_out=False
)

pipe = Pipeline(
    [
        ('transform', transformer),
        ('reg', TransformedTargetRegressor(regressor=Ridge(max_iter=10000),
                                           transformer=PowerTransformer()))
    ]
)

# include 0 in alpha to see OLS performance
param = {
    'reg__regressor__alpha': np.append(np.geomspace(0.00001, 100, 1000000), 0)
}

grid = GridSearchCV(
    estimator=pipe, 
    param_grid=param,
    scoring='neg_mean_squared_error',
    cv=KFold(n_splits=7, random_state=415, shuffle=True),
    n_jobs=-1,
    verbose=0
)


grid.fit(X, y)
save_cv_metric(grid.best_estimator_, X, y, 'linear')
joblib.dump(grid.best_estimator_, './models/linear.sav')