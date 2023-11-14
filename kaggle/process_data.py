import numpy as np 
import pandas as pd

X_train = pd.read_csv('./data/X_train.csv')
y_train = pd.read_csv('./data/y_train.csv')

test = pd.read_csv('./data/X_test.csv')

# Get numeric and categorical column name
NUM_COLS = [c for c in X_train.columns if c not in ['district', 'SEQN']]
CAT_COLS = ['district']


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
    
    sub_df = data[srp_cols]
    data['total_avg'] = np.average(sub_df, axis=1)
    data['total_std'] = np.std(sub_df, axis=1)
    data['total_iqr'] = np.percentile(sub_df, 75, axis=1) - np.percentile(sub_df, 25, axis=1)
    data['total_min'] = np.min(sub_df, axis=1)
    data['total_max'] = np.max(sub_df, axis=1)

    data['district'] = data['district'].apply(lambda x: 'even' if x % 2 == 0 else 'odd')


def get_train():
    X_train.drop('SEQN', axis=1, inplace=True)
    feature_egin(X_train)
    y_train.drop('SEQN', axis=1, inplace=True)
    return X_train, y_train


def get_test():
    X_test = test.drop('SEQN', axis=1)
    feature_egin(X_test)
    return X_test, test['SEQN']