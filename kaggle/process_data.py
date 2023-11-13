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


def get_train():
    X_train.drop('SEQN', axis=1, inplace=True)
    y_train.drop('SEQN', axis=1, inplace=True)
    return X_train, y_train


def get_test():
    X_test = test.drop('SEQN', axis=1)
    return X_test, test['SEQN']

