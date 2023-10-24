import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, \
    RobustScaler, StandardScaler
from read_data_util import NUM_COLS, CAT_COLS, \
    get_full_data, get_group_data


def xgb(X, y, params, name):
    """
    X: array-like, contains all features
    y: contains class labels
    params: dictionary, contains hyperparameter search grid
    """
    transformer = ColumnTransformer(
        [
            ('encode', OneHotEncoder(drop='first'), CAT_COLS),
            ('standardize', RobustScaler(), NUM_COLS)
        ],
        remainder='drop',
        n_jobs=-1,
        verbose_feature_names_out=False
    )
    
    pipe = Pipeline(
        [
            ('transform', transformer),
            ('clf', XGBClassifier()
            )
        ],
        memory='./.cache'
    )

    grid = RandomizedSearchCV(
        estimator=pipe, 
        param_distributions=params,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1,
        n_iter=1000000
    )

    grid.fit(X, y)
    joblib.dump(grid.best_estimator_, f'./models/{name}.sav')


def full_xgb(params):
    """
    Train XGB with full data
    """
    X, y = get_full_data()
    xgb(X, y, params, 'xgb_full')


def group_xgb(params):
    """
    Train 3 seperate XGB with grouped data
    """
    data = get_group_data()
    for idx, group in enumerate(['young', 'mid', 'old']):
        X, y = data[group]
        xgb(X, y, params[idx], f'xgb_{group}')