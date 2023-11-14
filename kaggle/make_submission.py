import joblib
import argparse
import pandas as pd
from process_data import get_test


parse = argparse.ArgumentParser()
parse.add_argument('-m', '--mode', type=str)


def make_sub(mode):
    model = joblib.load(f'./models/{mode}.sav')
    X, ids = get_test()
    pred = pd.Series(model.predict(X))
    res = pd.concat([ids, pred], keys=['SEQN', 'y'], axis=1)
    res.to_csv('./res.csv', index=False)

make_sub(parse.parse_args().mode)
