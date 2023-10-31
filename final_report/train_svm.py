import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, \
    RobustScaler, StandardScaler
from read_data_util import NUM_COLS, CAT_COLS, \
    get_full_data, get_group_data
from visualize_model import save_cv_metric


def svm(X, y, params, name):
    """
    X: array-like, contains all features
    y: contains class labels
    params: dictionary, contains hyperparameter search grid
    """

    transformer = ColumnTransformer(
        [('encode', OneHotEncoder(drop='first'), CAT_COLS),
        ('standardize', PowerTransformer(), NUM_COLS)],
        remainder='drop',
        n_jobs=-1,
        verbose_feature_names_out=False
    )

    pipe = Pipeline(
        [('transform', transformer),
        ('clf', SVC(kernel='rbf',
                    probability=True, 
                    class_weight='balanced'))
        ],
        memory='./.cache'
    )

    grid = GridSearchCV(
        estimator=pipe, 
        param_grid=params,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=3
    )

    grid.fit(X, y)
    joblib.dump(grid.best_estimator_, f'./models/{name}.sav')
    save_cv_metric(grid.best_estimator_, X, y, name)


def full_svm(params: dict):
    """
    Train SVM with full data
    """
    X, y = get_full_data()
    svm(X, y, params, 'svm_full')


def group_svm(params: list):
    """
    Train 3 seperate SVM with grouped data
    params: list of dictionary
    EX: params[0] is grid for young group
        params[1] is grid for middle age group
        params[2] is grid for elderly group
    """
    data = get_group_data()
    for idx, group in enumerate(['young', 'mid', 'old']):
        X, y = data[group]
        svm(X, y, params[idx], f'svm_{group}')

