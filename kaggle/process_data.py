import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv')
X_train.drop('SEQN', axis=1, inplace=True)
y_train.drop('SEQN', axis=1, inplace=True)

test = pd.read_csv('./data/X_test.csv')
X_test = test.drop('SEQN', axis=1)

# Get numeric and categorical column name
NUM_COLS = [c for c in X_train.columns if c not in ['district', 'SEQN', 'self_eval', 'teacher_eval']]
CAT_COLS = []

transformer = ColumnTransformer(
    [  
       ('scale', RobustScaler(), NUM_COLS),
       ('encode', OneHotEncoder(drop='first'), CAT_COLS)
    ],
    remainder='passthrough',
    n_jobs=-1,
    verbose_feature_names_out=False
)
transformer.fit(X_train)
X_train = pd.DataFrame(transformer.transform(X_train), columns=transformer.get_feature_names_out())
X_test = pd.DataFrame(transformer.transform(test), columns=transformer.get_feature_names_out())


def missingdata(data):
    '''
    Get columns with missing data , and percentage of missing data
    '''
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms = ms[ms['Percent'] > 0]
    return ms


def feature_egin(data: pd.DataFrame):
    srp_cols = np.array([f'SRP_{day}' for day in range(1, 51)])
    for i in range(1, 49, 7):
        sub_df = data[srp_cols[range(i, i+7)]]
        data[f'avg{i}'] = np.average(sub_df, axis=1)
        data[f'std{i}'] = np.std(sub_df, axis=1)
        data[f'min{i}'] = np.min(sub_df, axis=1)
        data[f'max{i}'] = np.max(sub_df, axis=1)
        data[f'iqr{i}'] = np.percentile(sub_df, 75, axis=1) - np.percentile(sub_df, 25, axis=1)
        data[f'trend{i}'] = sub_df.apply(lambda row: np.polyfit(range(len(row)), row, 1)[0], axis=1)
    
    sub_df = data[srp_cols]
    data['total_avg'] = np.average(sub_df, axis=1)
    data['total_std'] = np.std(sub_df, axis=1)
    data['total_iqr'] = np.percentile(sub_df, 75, axis=1) - np.percentile(sub_df, 25, axis=1)
    data['total_min'] = np.min(sub_df, axis=1)
    data['total_max'] = np.max(sub_df, axis=1)

    data['district'] = data['district'].apply(lambda x: 1 if x % 2 == 0 else 0)


def get_train():
    feature_egin(X_train)
    return X_train, y_train


def get_test():
    feature_egin(X_test)
    return X_test, test['SEQN']