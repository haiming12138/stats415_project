import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

def save_cv_metric(classifier, X, y, name):
    metrics = ['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error',
               'neg_mean_absolute_percentage_error', 'explained_variance']
    res = cross_validate(classifier, X, y, cv=KFold(n_splits=10, shuffle=True, 
                                                    random_state=614), 
                                                    scoring=metrics, n_jobs=-1)

    output = ''
    for metric in metrics:
        perf = np.round(res[f'test_{metric}'].mean(), 6)
        output += f'cv_{metric}: {perf}\n'
    with open(f'./perf_log/{name}.txt', '+w') as file:
        file.write(output)