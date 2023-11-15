import joblib
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import GridSearchCV
from process_data import get_train
from train_model_utils import save_cv_metric

model = joblib.load('./models/xgb_curr_best.sav')
X, y = get_train()
metrics = ['r2', 'neg_mean_squared_error', 'explained_variance']

params = {
    'reg__n_estimators': np.arange(275, 400, 1),
    'reg__max_depth': np.arange(3, 20),
    'reg__min_child_weight': np.geomspace(0.01, 0.3, 1000),
    'reg__gamma': np.geomspace(1.0,10.0,1000),
    'reg__learning_rate': np.geomspace(0.05,0.5,1000),
    'reg__subsample': np.geomspace(0.75,0.9,500),
    'reg__colsample_bylevel': np.arange(0.8,0.95,0.0025),
    'reg__reg_alpha': np.geomspace(0.0001, 5, 1000),
    'reg__reg_lambda': np.geomspace(0.00001, 5, 1000)
}

curr_best = cross_validate(model, X, y, 
                         cv=KFold(
                             n_splits=10, 
                             shuffle=True, 
                             random_state=614
                            ), scoring=metrics
                        )

best_res = np.round([np.mean(curr_best[f'test_{metric}']) for metric in metrics], 7)

early_stop_counter = 0
for i in range(2):
    for param in params.items():
        print(f'Optimize with {param[0]}')
        param_grid = {param[0] : param[1]}
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=7,
            n_jobs=-1,
            verbose=3
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
        curr_res = np.round([np.mean(curr_cv[f'test_{metric}']) for metric in metrics], 7)

        if np.all(curr_res > best_res):
            best_res = curr_res
            model = grid.best_estimator_
            early_stop_counter = 0
            joblib.dump(model, './models/xgb_optim.sav')
            save_cv_metric(model, X, y, 'xgb_optim')
            print('Success')
        else:
            early_stop_counter += 1
            print('Fail')
        
        if early_stop_counter > 9:
            break