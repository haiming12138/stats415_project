import numpy as np
import pandas as pd
import argparse
import joblib
from train_svm import full_svm, group_svm, svm
from train_xgb import full_xgb, group_xgb
from visualize_model import run_all_visual


parse = argparse.ArgumentParser()
parse.add_argument('-m', '--mode', required=True, 
                   choices=['svm_full', 'svm_group',
                            'xgb_full', 'xgb_group'], type=str)


def main():
    args = parse.parse_args()
    
    if args.mode == 'svm_full':
        params= {
        'clf__C': np.linspace(100, 200, 100), 
        'clf__gamma': np.linspace(7 * pow(10, -4), 1 * pow(10, -3), 100)
        }
        full_svm(params)
    elif args.mode == 'svm_group':
        param = {
        'clf__C': np.linspace(100, 200, 100), 
        'clf__gamma': np.linspace(7 * pow(10, -4), 1 * pow(10, -3), 100)
        }
        group_svm([param, param, param])
    elif args.mode == 'xgb_full':
        params = {}
        full_xgb(params)
    else:
        params = {}
        group_xgb(params)
    
    run_all_visual(args.mode)

    


if __name__ == '__main__':
    main()

