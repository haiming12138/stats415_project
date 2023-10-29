import numpy as np
import argparse
from train_svm import full_svm, group_svm, svm
from train_xgb import full_xgb, group_xgb
from visualize_model import run_all_visual


parse = argparse.ArgumentParser()
parse.add_argument('-m', '--mode', required=True, 
                   choices=['svm_full', 'svm_group',
                            'xgb_full', 'xgb_group', 'visualize'], type=str)


def main():
    args = parse.parse_args()
    
    if args.mode == 'svm_full':
        params= {
        'clf__C': np.linspace(100, 200, 100), 
        'clf__gamma': np.linspace(7 * pow(10, -4), 1 * pow(10, -3), 100)
        }
        full_svm(params)
        run_all_visual(args.mode)
    elif args.mode == 'svm_group':
        param = {
        'clf__C': np.linspace(100, 200, 100), 
        'clf__gamma': np.linspace(7 * pow(10, -4), 1 * pow(10, -3), 100)
        }
        group_svm([param, param, param])
        run_all_visual(args.mode)
    elif args.mode == 'xgb_full':
        params = {
            'clf__n_estimators': np.arange(5, 35, 5),
            'clf__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'clf__min_child_weight': np.arange(0.0001, 0.5, 0.001),
            'clf__gamma': np.arange(0.0,40.0,0.005),
            'clf__learning_rate': np.arange(0.0005,0.3,0.0005),
            'clf__subsample': np.arange(0.3,0.8,0.05),
            'clf__reg_alpha': np.linspace(0, 100, 50),
            'clf__reg_lambda': np.linspace(0, 100, 50),
            'clf__scale_pos_weight': [3, 4, 5, 6]
        }
        full_xgb(params)
        run_all_visual(args.mode)
    elif args.mode == 'xgb_group':
        param = {
            'clf__n_estimators': np.arange(5, 35, 5),
            'clf__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'clf__min_child_weight': np.arange(0.0001, 0.5, 0.001),
            'clf__gamma': np.arange(0.0,40.0,0.005),
            'clf__learning_rate': np.arange(0.0005,0.3,0.0005),
            'clf__subsample': np.arange(0.3,0.8,0.05),
            'clf__reg_alpha': np.linspace(0, 100, 50),
            'clf__reg_lambda': np.linspace(0, 100, 50),
            'clf__scale_pos_weight': [3, 4, 5, 6]
        }
        params = [param, param, param]
        group_xgb(params)
        run_all_visual(args.mode)
    else:
        for mode in ['svm_group']:
            run_all_visual(mode)

    


if __name__ == '__main__':
    main()

