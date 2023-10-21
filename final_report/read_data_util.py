import pandas as pd

NUM_COLS = ['BMXWT', 'BMXBMI', 'LBXBPB', 
            'LBXBCD', 'LBXTHG', 'LBXBSE', 
            'LBXBMN','RIDAGEYR', 'INDFMPIR']

CAT_COLS = ['RIAGENDR']

def get_raw_data(file: str):
    """ Read from file, and output data in format that correspond to sklearn """
    df = pd.read_csv(file)
    return df.iloc[:, 1:], df.iloc[:, 0]

def get_full_data():
    """
    return X, y
    """
    return get_raw_data('./datasets/data.csv')


def get_group_data():
    """
    return a dictionary containing X, y for each group
    Ex: dict['mid'][0] is the X for middle age group
    """
    res = {}
    for group in ['young', 'mid', 'old']:
        X, y = get_raw_data(f'./datasets/data_{group}.csv')
        res[group] = (X, y)
    return res
