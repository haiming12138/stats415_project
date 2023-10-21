import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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
    pass


def full_xgb(params):
    """
    Train XGB with full data
    """
    pass


def group_xgb(params):
    """
    Train 3 seperate XGB with grouped data
    """
    pass