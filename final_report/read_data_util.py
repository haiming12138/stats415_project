import pandas as pd

NUM_COLS = ['BMXWT', 'BMXBMI', 'LBXBPB', 
            'LBXBCD', 'LBXTHG', 'LBXBSE', 
            'LBXBMN','RIDAGEYR', 'INDFMPIR']

CAT_COLS = ['RIAGENDR']

def get_data(file: str):
    """ Read from file, and output data in format that correspond to sklearn"""
    df = pd.read_csv(file)
    return df.iloc[:, 1:], df.iloc[:, 0]