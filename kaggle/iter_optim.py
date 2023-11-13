import joblib
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import GridSearchCV
from process_data import get_train
from train_model_utils import save_cv_metric

model = joblib.load('./models/xgb.sav')
X, y = get_train()
metrics = ['r2', 'neg_mean_squared_error', 'explained_variance']

params = {
    'reg__n_estimators': np.arange(230, 360, 1),
    'reg__max_depth': np.arange(3, 25),
    'reg__min_child_weight': np.geomspace(0.01, 0.5, 1000),
    'reg__gamma': np.geomspace(1.0,10.0,1000),
    'reg__learning_rate': np.geomspace(0.05,0.5,1000),
    'reg__subsample': np.arange(0.7,0.9,0.005),
    'reg__colsample_bylevel': np.arange(0.7,0.9,0.005),
    'reg__reg_alpha': np.geomspace(0.0001, 40, 1000),
    'reg__reg_lambda': np.geomspace(0.0001, 40, 1000)
}

curr_best = cross_validate(model, X, y, 
                         cv=KFold(
                             n_splits=10, 
                             shuffle=True, 
                             random_state=614
                            ), scoring=metrics
                        )

best_res = [np.mean(curr_best[f'test_{metric}']) for metric in metrics]

for param in params.items():
    print(f'Optimize with {param[0]}')
    param_grid = {param[0] : param[1]}
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=7,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X, y)
    curr_cv = cross_validate(grid.best_estimator_, 
                         X, y, 
                         cv=KFold(
                             n_splits=10, 
                             shuffle=True, 
                             random_state=614
                            ), 
                         scoring=metrics
                        )
    curr_res = [np.mean(curr_cv[f'test_{metric}']) for metric in metrics]

    if np.all(curr_res >= best_res):
        best_res = curr_res
        model = grid.best_estimator_
        print('Success')
    else:
        print('Fail')

joblib.dump(model, './models/xgb_optim.sav')
save_cv_metric(model, X, y, 'xgb_optim')